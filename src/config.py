"""
MoE-LoRA 配置模块
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MoELoRAConfig:
    """Mixture of Experts LoRA 配置

    Attributes:
        r: LoRA 秩（低秩分解的中间维度）
        lora_alpha: LoRA 缩放因子
        num_experts: 专家数量 K
        top_k: 每个 token 激活的 top-k 个专家
        target_modules: 需要应用 MoE-LoRA 的模块名称列表
        lora_dropout: LoRA 层的 dropout 概率
        expert_dropout: 专家选择的 dropout（用于负载均衡探索）
        router_hidden_dim: 路由器隐藏层维度（None 表示单层线性路由器）
        use_noisy_router: 是否使用带噪声的 Top-k 门控
        noise_epsilon: 噪声项 epsilon（防止 log 爆炸）
        load_balance_weight: 负载均衡损失的权重系数
        router_z_loss_weight: router z-loss 权重（稳定训练）
        init_scale: 路由器初始化的标准差缩放
        fan_mode: fan_in 或 fan_out 初始化模式
        merge_weights: 推理时是否合并权重
        bias: LoRA 中是否使用 bias
        init_lora_weights: LoRA 权重初始化方式 ('gaussian', 'kaiming', 'pissa')
    """
    # LoRA 参数
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # MoE 参数
    num_experts: int = 8
    top_k: int = 2
    expert_dropout: float = 0.0

    # 路由器参数
    router_hidden_dim: Optional[int] = None
    use_noisy_router: bool = True
    noise_epsilon: float = 1e-2
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    # 初始化参数
    init_scale: float = 0.02
    fan_mode: str = "fan_in"

    # 目标模块
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 其他
    merge_weights: bool = False
    bias: str = "none"
    init_lora_weights: str = "gaussian"

    def __post_init__(self):
        """dataclass 初始化后自动校验参数合法性"""
        assert self.top_k <= self.num_experts, \
            f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
        valid_bias = ["none", "all", "lora_only"]
        assert self.bias in valid_bias, \
            f"bias must be one of {valid_bias}, got {self.bias}"
        valid_init = ["gaussian", "kaiming", "pissa"]
        assert self.init_lora_weights in valid_init, \
            f"init_lora_weights must be one of {valid_init}"
        valid_fan = ["fan_in", "fan_out"]
        assert self.fan_mode in valid_fan, \
            f"fan_mode must be one of {valid_fan}"


@dataclass
class TrainingConfig:
    """训练配置"""
    # ---- 模型 ----
    model_name: str = "Qwen/Qwen1.5-7B"
    trust_remote_code: bool = True

    # ---- 数据 ----
    dataset_name: str = "tatsu-lab/alpaca"
    max_length: int = 512

    # ---- 训练超参数 ----
    output_dir: str = "./qwen-moe-lora"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4     # 有效 batch = 4 * 4 = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03               # 前 3% 步数线性 warmup
    lr_scheduler_type: str = "cosine"        # 余弦退火调度
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # ---- 量化配置（bitsandbytes 4-bit） ----
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16" # 计算时的数据类型
    bnb_4bit_quant_type: str = "nf4"         # NormalFloat4 量化
    use_nested_quant: bool = False           # 双重量化（进一步压缩）

    # ---- 其他 ----
    seed: int = 42
    report_to: str = "none"  # 或 "wandb"
    fp16: bool = False
    bf16: bool = True                        # BFloat16 混合精度训练
    gradient_checkpointing: bool = True      # 用计算换显存
    ddp_find_unused_parameters: bool = False
