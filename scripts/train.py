"""
MoE-LoRA Qwen1.5 微调训练脚本

使用方式:
    python train.py                          # 使用默认配置
    python train.py --num_experts 4 --top_k 2  # 自定义 MoE 参数
    python train.py --model_name Qwen/Qwen1.5-1.8B  # 使用更小的模型

依赖:
    pip install -r requirements.txt
"""

import os
import math
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset

from src.config import MoELoRAConfig, TrainingConfig
from src.moe_lora import MoELoRALinear, MoELoRAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义 Trainer：将 MoE 辅助损失合并到语言模型损失中
# ---------------------------------------------------------------------------

class MoELoRATrainer(Trainer):
    """支持 MoE-LoRA 辅助损失的自定义 Trainer

    在标准语言模型损失的基础上添加：
    1. 负载均衡损失 (load balance loss) — 鼓励 token 均匀分配到各专家
    2. Router Z-loss — 稳定路由器训练
    """

    def __init__(
        self,
        moe_lora_model: MoELoRAModel,
        load_balance_weight: float = 0.01,
        router_z_loss_weight: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__(model=moe_lora_model, *args, **kwargs)
        self.moe_lora_model = moe_lora_model
        self.load_balance_weight = load_balance_weight
        self.router_z_loss_weight = router_z_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss，将 MoE 辅助损失合并到总损失中。

        总损失 = LM损失 + α * 负载均衡损失 + β * router_z_loss
        """
        # 标准前向传播
        outputs, load_balance_loss, router_z_loss = model(**inputs)

        # 语言模型损失
        lm_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # 合并辅助损失
        total_loss = lm_loss
        if self.load_balance_weight > 0 and load_balance_loss is not None:
            total_loss = total_loss + self.load_balance_weight * load_balance_loss
        if self.router_z_loss_weight > 0 and router_z_loss is not None:
            total_loss = total_loss + self.router_z_loss_weight * router_z_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """添加 MoE 相关的日志"""
        # 收集所有 MoE-LoRA 层的损失用于日志
        lb_losses = []
        z_losses = []
        for layer in self.moe_lora_model.moe_lora_layers.values():
            lb, z = layer.get_last_losses()
            if lb is not None:
                lb_losses.append(lb.item())
            if z is not None:
                z_losses.append(z.item())

        if lb_losses:
            logs["moe/load_balance_loss"] = sum(lb_losses) / len(lb_losses)
        if z_losses:
            logs["moe/router_z_loss"] = sum(z_losses) / len(z_losses)

        super().log(logs, **kwargs)


# ---------------------------------------------------------------------------
# 数据预处理
# ---------------------------------------------------------------------------

def format_alpaca_instruction(example: dict) -> str:
    """将 Alpaca 格式数据格式化为指令文本"""
    if example.get("input") and example["input"].strip():
        text = (
            f"Below is an instruction that describes a task, paired with an input "
            f"that provides further context. Write a response that appropriately "
            f"completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return text


def preprocess_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
):
    """预处理数据集：格式化 + 分词
    
    将 Alpaca 格式的 instruction/input/output 三元组
    转换为模型可接受的 input_ids/attention_mask/labels 格式。
    labels 与 input_ids 相同，模型内部会自动 shift 处理。
    """

    def format_and_tokenize(example):
        """单条数据：格式化文本 -> 分词 -> 生成 labels"""
        text = format_alpaca_instruction(example)
        result = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,           # 在 collator 中统一 padding，这里不处理
            return_tensors=None,      # 返回 Python list 而非 tensor
        )
        # labels 与 input_ids 一致，模型内部自动处理 loss mask
        result["labels"] = result["input_ids"].copy()
        return result

    # 移除原始列，仅保留分词后的三列
    columns_to_remove = [
        col for col in dataset.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]

    tokenized = dataset.map(
        format_and_tokenize,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )

    return tokenized


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main():
    # 解析配置
    moe_config = MoELoRAConfig()
    train_config = TrainingConfig()

    # ---- 命令行参数解析 ----
    # 允许通过命令行覆盖默认配置，方便快速实验
    import argparse
    arg_parser = argparse.ArgumentParser(description="MoE-LoRA Fine-tuning for Qwen1.5")
    
    # 模型与 MoE 参数
    arg_parser.add_argument("--model_name", type=str, default=train_config.model_name)
    arg_parser.add_argument("--num_experts", type=int, default=moe_config.num_experts)
    arg_parser.add_argument("--top_k", type=int, default=moe_config.top_k)
    arg_parser.add_argument("--r", type=int, default=moe_config.r)
    arg_parser.add_argument("--lora_alpha", type=int, default=moe_config.lora_alpha)
    
    # 训练超参数
    arg_parser.add_argument("--num_epochs", type=int, default=train_config.num_epochs)
    arg_parser.add_argument("--batch_size", type=int, default=train_config.per_device_train_batch_size)
    arg_parser.add_argument("--learning_rate", type=float, default=train_config.learning_rate)
    arg_parser.add_argument("--max_length", type=int, default=train_config.max_length)
    
    # 输出与日志
    arg_parser.add_argument("--output_dir", type=str, default=train_config.output_dir)
    arg_parser.add_argument("--report_to", type=str, default=train_config.report_to)
    
    # 量化开关
    arg_parser.add_argument("--no_4bit", action="store_true",
                            help="禁用 4-bit 量化（需要更多显存）")
    args = arg_parser.parse_args()

    # 将命令行参数同步到配置对象
    train_config.model_name = args.model_name
    train_config.num_epochs = args.num_epochs
    train_config.per_device_train_batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    train_config.output_dir = args.output_dir
    train_config.max_length = args.max_length
    train_config.report_to = args.report_to
    train_config.use_4bit = not args.no_4bit
    moe_config.num_experts = args.num_experts
    moe_config.top_k = args.top_k
    moe_config.r = args.r
    moe_config.lora_alpha = args.lora_alpha

    set_seed(train_config.seed)

    # -----------------------------------------------------------------------
    # 加载分词器
    # -----------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {train_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name,
        trust_remote_code=train_config.trust_remote_code,
    )

    # Qwen 使用 pad_token_id 作为 eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # 加载模型
    # -----------------------------------------------------------------------
    logger.info(f"Loading model: {train_config.model_name}")
    model_kwargs = {
        "trust_remote_code": train_config.trust_remote_code,
        "torch_dtype": torch.bfloat16 if train_config.bf16 else torch.float16,
    }

    if train_config.use_4bit:
        logger.info("Using 4-bit quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, train_config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=train_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=train_config.use_nested_quant,
        )
        model_kwargs["quantization_config"] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        **model_kwargs,
    )

    # 启用 gradient checkpointing
    if train_config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False  # gradient checkpointing 需要关闭 cache

    # -----------------------------------------------------------------------
    # 应用 MoE-LoRA
    # -----------------------------------------------------------------------
    logger.info(
        f"Applying MoE-LoRA with r={moe_config.r}, "
        f"num_experts={moe_config.num_experts}, top_k={moe_config.top_k}"
    )
    model = MoELoRAModel(
        model=base_model,
        moe_lora_config=moe_config,
    )

    trainable_params = model.get_num_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # -----------------------------------------------------------------------
    # 加载数据
    # -----------------------------------------------------------------------
    logger.info(f"Loading dataset: {train_config.dataset_name}")
    dataset = load_dataset(train_config.dataset_name)

    # 处理 train/test split
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset

    # 划分训练/验证集（95%/5%）
    # 大数据集时截取前 10000 条用于快速实验
    if len(train_dataset) > 10000:
        train_dataset = train_dataset.select(range(10000))

    split = train_dataset.train_test_split(test_size=0.05, seed=train_config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ---- 数据预处理：格式化 + 分词 ----
    logger.info("Preprocessing dataset...")

    def format_and_tokenize(example):
        """单条数据处理：Alpaca 格式文本 -> 分词 -> 生成 labels"""
        text = format_alpaca_instruction(example)
        result = tokenizer(
            text,
            truncation=True,
            max_length=train_config.max_length,
            padding=False,           # collator 中统一 padding
            return_tensors=None,      # 返回 Python list
        )
        # 因果语言模型：labels 与 input_ids 相同，模型内部自动 shift
        result["labels"] = result["input_ids"].copy()
        return result

    # 仅保留分词后的三列
    columns_to_remove = [
        col for col in train_dataset.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]

    train_dataset = train_dataset.map(
        format_and_tokenize,
        remove_columns=columns_to_remove,
        desc="Tokenizing train",
    )
    eval_dataset = eval_dataset.map(
        format_and_tokenize,
        remove_columns=columns_to_remove,
        desc="Tokenizing eval",
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=base_model,
        padding=True,
    )

    # -----------------------------------------------------------------------
    # 训练参数
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        weight_decay=train_config.weight_decay,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=train_config.bf16,
        fp16=train_config.fp16,
        report_to=train_config.report_to if train_config.report_to != "none" else "none",
        ddp_find_unused_parameters=train_config.ddp_find_unused_parameters,
        save_total_limit=3,
        remove_unused_columns=False,  # 重要：保留 labels 列
        dataloader_num_workers=4,
        seed=train_config.seed,
    )

    # -----------------------------------------------------------------------
    # 创建 Trainer
    # -----------------------------------------------------------------------
    trainer = MoELoRATrainer(
        moe_lora_model=model,
        load_balance_weight=moe_config.load_balance_weight,
        router_z_loss_weight=moe_config.router_z_loss_weight,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # -----------------------------------------------------------------------
    # 开始训练
    # -----------------------------------------------------------------------
    logger.info("Starting training...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 保存模型
    # -----------------------------------------------------------------------
    logger.info(f"Saving model to {train_config.output_dir}")
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)

    # 保存 MoE-LoRA 配置
    import json
    from dataclasses import asdict
    with open(os.path.join(train_config.output_dir, "moe_lora_config.json"), "w") as f:
        json.dump(asdict(moe_config), f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
