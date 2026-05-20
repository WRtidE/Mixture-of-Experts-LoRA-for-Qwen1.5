"""
MoE-LoRA Qwen1.5 推理示例

演示：
1. 加载微调后的 MoE-LoRA 模型
2. 文本生成
3. 可视化专家路由模式（哪个 token 激活了哪些专家）
4. 分析各专家的使用频率
"""

import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.config import MoELoRAConfig
from src.moe_lora import MoELoRAModel, MoELoRALinear


def load_model(
    base_model_name: str = "Qwen/Qwen1.5-7B",
    moe_lora_path: Optional[str] = None,
    moe_config_path: Optional[str] = None,
    use_4bit: bool = True,
):
    """加载带有 MoE-LoRA 权重的模型

    Args:
        base_model_name: 基础模型名称或路径
        moe_lora_path: MoE-LoRA 权重检查点路径
        moe_config_path: MoE-LoRA 配置文件路径
        use_4bit: 是否使用 4-bit 量化

    Returns:
        model, tokenizer
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    )

    # 加载 MoE-LoRA 配置
    if moe_config_path and os.path.exists(moe_config_path):
        with open(moe_config_path) as f:
            config_dict = json.load(f)
        moe_config = MoELoRAConfig(**config_dict)
    else:
        moe_config = MoELoRAConfig()

    # 应用 MoE-LoRA 结构
    model = MoELoRAModel(
        model=base_model,
        moe_lora_config=moe_config,
    )

    # 加载训练好的权重
    if moe_lora_path and os.path.exists(moe_lora_path):
        model.load_moe_lora(moe_lora_path)
        print(f"Loaded MoE-LoRA weights from {moe_lora_path}")

    model.eval()
    return model, tokenizer


def generate(
    model: MoELoRAModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """使用 MoE-LoRA 模型生成文本

    Args:
        model: MoE-LoRA 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus sampling 参数
        do_sample: 是否采样

    Returns:
        生成的文本
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.model.device)

    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text


def analyze_expert_routing(
    model: MoELoRAModel,
    tokenizer: AutoTokenizer,
    text: str,
) -> dict:
    """分析给定文本中每个 token 的专家路由情况

    通过注册 hook 捕获每个 MoE-LoRA 层的路由器输出，
    统计各专家的使用频率。

    Args:
        model: MoE-LoRA 模型
        tokenizer: 分词器
        text: 输入文本

    Returns:
        包含路由分析结果的字典:
            - expert_usage: {layer_name: {expert_id: count}}
            - token_routing: {layer_name: [(token, expert_indices, weights)]}
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # ---- 通过 pre-forward hook 捕获路由信息 ----
    # 策略：在每次前向传播前，拦截输入 hidden_states，
    #       手动调用路由器获取 routing_weights 和 expert_indices，
    #       存储到 captured_routings 中供后续分析。
    hooks = []
    captured_routings = {}

    def pre_forward_hook(layer_name):
        """创建闭包，捕获指定层的路由信息"""
        def hook(module, input_tensor):
            # input_tensor 是 (hidden_states,) 元组，取第一个元素
            if isinstance(input_tensor, tuple):
                hs = input_tensor[0]
            else:
                hs = input_tensor

            # 手动运行路由器（不触发完整 forward，仅获取路由决策）
            with torch.no_grad():
                rw, ei, rl = module.router(hs)
                captured_routings[layer_name] = {
                    "routing_weights": rw.cpu(),  # (1, S, top_k)
                    "expert_indices": ei.cpu(),   # (1, S, top_k)
                    "router_logits": rl.cpu(),    # (1, S, num_experts)
                }
        return hook

    # 为每一层 MoE-LoRA 注册 pre-forward hook
    for name, layer in model.moe_lora_layers.items():
        h = layer.register_forward_pre_hook(pre_forward_hook(name))
        hooks.append(h)

    # 运行一次前向传播，触发所有 hook
    with torch.no_grad():
        _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # 移除 hooks（避免影响后续推理）
    for h in hooks:
        h.remove()

    # ---- 整理分析结果 ----
    analysis = {
        "expert_usage": {},     # 每个专家被选中的次数统计
        "token_routing": {},    # 每层的 token 级别路由详情
        "expert_frequency": {}, # 每层各专家的使用频率（归一化）
    }

    for layer_name, data in captured_routings.items():
        expert_indices = data["expert_indices"]  # (1, S, top_k)
        routing_weights = data["routing_weights"]

        seq_len = expert_indices.shape[1]
        num_experts = model.moe_lora_layers[layer_name].num_experts

        # 统计每个专家被选中的总次数
        usage = torch.zeros(num_experts, dtype=torch.int32)
        token_routes = []

        for pos in range(seq_len):
            if pos >= len(tokens):
                break
            tok = tokens[pos]

            # 该 token 位置被选中的 top_k 专家 ID 和对应权重
            pos_indices = expert_indices[0, pos].tolist()
            pos_weights = routing_weights[0, pos].tolist()

            for eid in pos_indices:
                usage[eid] += 1

            token_routes.append({
                "token": tok,
                "expert_indices": pos_indices,
                "weights": [round(w, 4) for w in pos_weights],
            })

        # 原始计数
        analysis["expert_usage"][layer_name] = usage.tolist()
        # 归一化为频率（加 epsilon 防止除零）
        analysis["expert_frequency"][layer_name] = (
            usage.float() / (usage.sum() + 1e-8)
        ).tolist()
        analysis["token_routing"][layer_name] = token_routes

    return analysis


def print_routing_analysis(analysis: dict, tokens_to_show: int = 20):
    """打印路由分析结果"""
    print("\n" + "=" * 70)
    print("MoE 专家路由分析")
    print("=" * 70)

    for layer_name, token_routes in analysis["token_routing"].items():
        frequencies = analysis["expert_frequency"][layer_name]
        print(f"\n--- {layer_name} ---")
        print(f"专家使用频率: ", end="")
        for eid, freq in enumerate(frequencies):
            bar = "█" * int(freq * 50)
            print(f"\n  Expert {eid}: {freq*100:5.1f}% {bar}", end="")
        print()

        print(f"\nToken 级别路由 (前 {tokens_to_show} 个):")
        print(f"{'Token':<15} {'Experts':<25} {'Weights'}")
        print("-" * 60)
        for route in token_routes[:tokens_to_show]:
            token_str = route["token"].replace("Ġ", "_")
            experts_str = str(route["expert_indices"])
            weights_str = str(route["weights"])
            print(f"{token_str:<15} {experts_str:<25} {weights_str}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """演示 MoE-LoRA 推理功能"""
    print("MoE-LoRA Qwen1.5 推理演示\n")

    # 使用小模型做演示
    model_name = "Qwen/Qwen1.5-0.5B"  # 最小的 Qwen1.5，方便测试

    print(f"正在加载模型: {model_name}")
    print("注意：首次运行会下载模型文件\n")

    try:
        model, tokenizer = load_model(
            base_model_name=model_name,
            use_4bit=False,  # 小模型不需要量化
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已安装 transformers 并且网络可访问 HuggingFace")
        print("\n你可以通过设置 HF_ENDPOINT 环境变量使用镜像站:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        return

    print(f"可训练参数: {model.get_num_trainable_parameters():,}")

    # 测试生成
    prompt = "Explain the concept of machine learning in simple terms:"
    print(f"\n提示词: {prompt}")
    print("\n正在生成...")

    try:
        response = generate(model, tokenizer, prompt, max_new_tokens=128)
        print(f"\n生成结果:\n{response}")
    except Exception as e:
        print(f"生成失败: {e}")

    # 分析专家路由
    test_text = "What is the capital of France?"
    print(f"\n\n正在分析专家路由: '{test_text}'")

    try:
        analysis = analyze_expert_routing(model, tokenizer, test_text)
        print_routing_analysis(analysis)
    except Exception as e:
        print(f"路由分析失败: {e}")


if __name__ == "__main__":
    demo()
