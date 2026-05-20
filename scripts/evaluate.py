"""
MoE-LoRA 消融实验评估脚本

评估已训练的多个模型检查点（不同专家数量），收集关键指标：
1. 语言模型困惑度 (Perplexity) — 验证集上
2. 专家利用率 (Expert Utilization) — 路由分布均匀性
3. 可训练参数量 — 与专家数量的关系
4. 推理速度 — tokens/sec

输出: ablation_results/evaluation_summary.json
"""

import json
import os
import time
import argparse
from typing import Dict, List
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.config import MoELoRAConfig
from src.moe_lora import MoELoRAModel


def load_trained_model(
    base_model_name: str,
    checkpoint_dir: str,
    moe_config_dict: dict,
):
    """加载已训练的 MoE-LoRA 模型

    Args:
        base_model_name: 基础模型名称
        checkpoint_dir: 检查点目录（包含 MoE-LoRA 权重）
        moe_config_dict: MoE-LoRA 配置字典

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_model.eval()

    moe_config = MoELoRAConfig(**moe_config_dict)
    model = MoELoRAModel(model=base_model, moe_lora_config=moe_config)

    # 查找并加载检查点
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if not os.path.exists(adapter_path):
        # 尝试在 checkpoint_dir 下递归查找 .bin 文件
        bin_files = list(Path(checkpoint_dir).rglob("*.bin"))
        bin_files += list(Path(checkpoint_dir).rglob("*.safetensors"))
        if bin_files:
            adapter_path = str(bin_files[0])
        else:
            print(f"  [警告] 未找到检查点文件于 {checkpoint_dir}，使用未训练的 MoE-LoRA")
            return model, tokenizer

    model.load_moe_lora(adapter_path)
    model.eval()
    return model, tokenizer


def compute_perplexity(
    model: MoELoRAModel,
    tokenizer,
    eval_texts: List[str],
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """计算验证集上的困惑度

    PPL = exp(cross_entropy_loss)
    越低越好
    """
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(
        range(0, len(eval_texts), batch_size),
        desc="  计算 PPL",
        leave=False,
    ):
        batch_texts = eval_texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(model.model.device)

        with torch.no_grad():
            outputs, _, _ = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                labels=encodings["input_ids"],
            )

        # outputs 是 HuggingFace 输出对象
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss.item()
            total_loss += loss * encodings["input_ids"].size(1)
            total_tokens += encodings["input_ids"].size(1)
        else:
            # 回退：用 logits 手动计算
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encodings["input_ids"][:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).item()
            total_loss += loss * shift_labels.size(1)
            total_tokens += shift_labels.size(1)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    return perplexity


def compute_expert_utilization(
    model: MoELoRAModel,
    tokenizer,
    eval_texts: List[str],
    max_length: int = 512,
) -> Dict[str, float]:
    """计算专家利用率

    通过前向传播捕获每层的路由决策，统计每个专家被选中的频率。
    返回：
        avg_utilization: 实际使用的专家占比 (used/total)
        entropy: 路由分布的熵（越高越均匀）
        max_min_ratio: 最热门/最冷门专家的使用比（越接近1越均匀）
    """
    hooks = []
    captured_routings = {}

    def make_hook(layer_name):
        def hook(module, input_tensor):
            if isinstance(input_tensor, tuple):
                hs = input_tensor[0]
            else:
                hs = input_tensor
            with torch.no_grad():
                rw, ei, rl = module.router(hs)
                captured_routings[layer_name] = {
                    "expert_indices": ei.cpu(),
                    "routing_weights": rw.cpu(),
                }
        return hook

    for name, layer in model.moe_lora_layers.items():
        h = layer.register_forward_pre_hook(make_hook(name))
        hooks.append(h)

    # 用少量文本触发路由
    sample_texts = eval_texts[:8]  # 8 条样本足以估计利用率
    for text in sample_texts:
        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.model.device)
        with torch.no_grad():
            _, _, _ = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )

    for h in hooks:
        h.remove()

    # 统计专家使用频率
    layer_metrics = {}
    for layer_name, data in captured_routings.items():
        expert_indices = data["expert_indices"]  # (B, S, top_k)
        num_experts = model.moe_lora_layers[layer_name].num_experts

        usage = torch.zeros(num_experts, dtype=torch.float32)
        for e in range(num_experts):
            usage[e] = (expert_indices == e).sum().float()

        total = usage.sum()
        if total == 0:
            continue

        probs = usage / total

        # 被使用的专家占比（概率 > 1/num_experts 的视为"被使用"）
        utilized = (probs > 1.0 / num_experts).sum().item()
        utilization_rate = utilized / num_experts

        # 熵：衡量分布均匀程度
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum().item()
        max_entropy = np.log(num_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 热门/冷门比
        max_prob = probs.max().item()
        min_prob = probs[probs > 0].min().item() if (probs > 0).any() else 0
        max_min_ratio = max_prob / max(min_prob, 1e-8)

        layer_metrics[layer_name] = {
            "utilization_rate": round(utilization_rate, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "max_min_ratio": round(max_min_ratio, 4),
            "usage_distribution": [round(p, 4) for p in probs.tolist()],
        }

    # 跨层平均
    if layer_metrics:
        avg_util = np.mean([m["utilization_rate"] for m in layer_metrics.values()])
        avg_entropy = np.mean([m["normalized_entropy"] for m in layer_metrics.values()])
        avg_ratio = np.mean([m["max_min_ratio"] for m in layer_metrics.values()])
    else:
        avg_util = avg_entropy = avg_ratio = 0.0

    return {
        "avg_utilization_rate": round(avg_util, 4),
        "avg_normalized_entropy": round(avg_entropy, 4),
        "avg_max_min_ratio": round(avg_ratio, 4),
        "per_layer": layer_metrics,
    }


def compute_inference_speed(
    model: MoELoRAModel,
    tokenizer,
    prompt: str = "The capital of France is",
    num_tokens: int = 128,
    num_runs: int = 5,
) -> float:
    """测量推理速度 (tokens/sec)"""
    encodings = tokenizer(prompt, return_tensors="pt").to(model.model.device)

    # 预热
    with torch.no_grad():
        for _ in range(2):
            _ = model.model.generate(
                **encodings,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # 正式测评
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.model.generate(
                **encodings,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    tokens_per_sec = (num_tokens * num_runs) / elapsed
    return tokens_per_sec


def count_trainable_params(model: MoELoRAModel) -> Dict:
    """统计可训练参数"""
    trainable = model.get_num_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "ratio": round(trainable / total * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="MoE-LoRA 消融实验评估")
    parser.add_argument(
        "--results_dir", type=str, default="./ablation_results",
        help="消融实验结果根目录"
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen1.5-0.5B",
        help="基础模型名称"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=200,
        help="评估样本数"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MoE-LoRA 消融实验评估")
    print("=" * 60)
    print(f"结果目录: {args.results_dir}")
    print(f"基础模型: {args.base_model}")
    print()

    # 加载评估数据
    print("加载评估数据...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(min(args.eval_samples, len(dataset))))

    def format_text(example):
        if example.get("input") and example["input"].strip():
            return f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"
        return f"Instruction: {example['instruction']}\nResponse: {example['output']}"

    eval_texts = [format_text(ex) for ex in dataset]
    print(f"评估样本数: {len(eval_texts)}")

    # 收集各实验的结果
    all_results = []

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[错误] 结果目录不存在: {args.results_dir}")
        print("请先运行 run_ablation.sh 进行训练")
        return

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("experts_"):
            continue

        try:
            num_experts = int(exp_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        print(f"\n{'=' * 60}")
        print(f"评估: num_experts={num_experts}")
        print(f"检查点: {exp_dir}")

        # 加载模型
        try:
            model, tokenizer = load_trained_model(
                base_model_name=args.base_model,
                checkpoint_dir=str(exp_dir),
                moe_config_dict={
                    "num_experts": num_experts,
                    "top_k": 2,
                    "r": 16,
                    "lora_alpha": 32,
                },
            )
        except Exception as e:
            print(f"  [错误] 加载模型失败: {e}")
            continue

        result = {"num_experts": num_experts}

        # 1. 困惑度
        print("  [1/4] 计算困惑度...")
        result["perplexity"] = round(
            compute_perplexity(model, tokenizer, eval_texts[:100]), 2
        )

        # 2. 专家利用率
        print("  [2/4] 分析专家利用率...")
        result["expert_utilization"] = compute_expert_utilization(
            model, tokenizer, eval_texts
        )

        # 3. 参数量
        print("  [3/4] 统计参数量...")
        result["params"] = count_trainable_params(model)

        # 4. 推理速度
        if torch.cuda.is_available():
            print("  [4/4] 测量推理速度...")
            result["inference_speed_tps"] = round(
                compute_inference_speed(model, tokenizer), 1
            )
        else:
            print("  [4/4] 跳过推理速度（无 GPU）")
            result["inference_speed_tps"] = None

        all_results.append(result)

        print(f"  困惑度: {result['perplexity']}")
        print(f"  专家利用率: {result['expert_utilization']['avg_utilization_rate']}")
        print(f"  可训练参数: {result['params']['trainable']:,}")
        if result["inference_speed_tps"]:
            print(f"  推理速度: {result['inference_speed_tps']:.1f} tokens/sec")

    # 保存结果
    output_path = results_dir / "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 60}")
    print(f"评估完成！结果保存至: {output_path}")
    print(f"下一步: python visualize.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()
