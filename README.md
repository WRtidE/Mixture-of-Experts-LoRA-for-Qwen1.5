# MoE-LoRA: Mixture of Experts LoRA for Qwen1.5

基于 PyTorch 和 HuggingFace Transformers 实现的自定义 **MoE-LoRA**（Mixture of Experts + Low-Rank Adaptation）微调方案，专为 Qwen1.5 系列模型设计。

> **项目背景**：大语言模型参数高效微调技术研究 — 通过冻结预训练参数、引入稀疏专家 LoRA 适配模块，在资源受限场景下实现高效下游任务适配。

## 项目结构

```
Qwen/
├── src/                         # 🔬 核心模块
│   ├── __init__.py              #   包初始化
│   ├── moe_lora.py              #   Router, MoELoRALinear, MoELoRAModel
│   └── config.py                #   MoELoRAConfig, TrainingConfig
│
├── scripts/                     # 🚀 运行脚本
│   ├── train.py                 #   训练脚本（含自定义 Trainer）
│   ├── inference.py             #   推理示例 & 专家路由可视化
│   ├── evaluate.py              #   消融实验评估
│   └── visualize.py             #   结果可视化（图表生成）
│
├── docs/                        # 📄 文档
│   ├── EXPERIMENT_PLAN.md       #   实验计划与分工
│   └── REPORT_OUTLINE.md        #   报告大纲（填空式模板）
│
├── run_ablation.sh              #   ⚡ 一键消融实验入口
├── requirements.txt
└── README.md
```

## 架构概览

```
Input (batch, seq_len, d_in)
    │
    ├──► Base Linear (frozen) ──────────────────────┐
    │                                                │
    ├──► Router (Gating Network) ──► Top-K 稀疏激活   │
    │         │                                      │
    │    ┌────┴────┬─────────┐                       │
    │    ▼         ▼         ▼                       │
    │  Expert₀  Expert₁  ... Expertₖ₋₁              │
    │  B₀@A₀@x  B₁@A₁@x     Bₖ₋₁@Aₖ₋₁@x            │
    │    │         │         │                       │
    │    └────┬────┴────┬────┘                       │
    │         ▼                                      │
    │    Σ( g_k · B_k @ A_k @ x ) · (α/r) ──────────┤
    │                                                │
    └────────────────────────────────────────────────┼──► Output
                                                     │
                                              base + MoE-LoRA
```

### 核心组件

| 组件 | 描述 |
|------|------|
| **Router (门控网络)** | 可学习的路由网络，根据 token hidden state 动态选择 top-k 个专家 |
| **Experts (专家)** | 每个专家是一对 LoRA 低秩矩阵 (Aₖ, Bₖ)，学习不同的知识模式 |
| **Load Balance Loss** | 辅助损失，鼓励 token 均匀分配到各专家，防止专家坍塌 |
| **Router Z-Loss** | 稳定路由器训练，防止 logits 发散 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install matplotlib seaborn   # 可视化依赖
```

### 2. 快速验证（单组小实验，约15分钟）

```bash
python scripts/train.py \
    --model_name Qwen/Qwen1.5-0.5B \
    --num_experts 4 --num_epochs 1 \
    --no_4bit \
    --output_dir ./test_run
```

### 3. 一键运行消融实验

```bash
./run_ablation.sh                # 串行训练 num_experts=1,2,4,8,16
./run_ablation.sh --dry-run      # 预览命令，不实际执行
./run_ablation.sh --expert 8     # 仅运行 num_experts=8 的单组
```

### 4. 评估与可视化

```bash
python scripts/evaluate.py --results_dir ./ablation_results
python scripts/visualize.py --results_dir ./ablation_results
```

### 5. 推理

```bash
python scripts/inference.py
```

## 训练命令参考

```bash
# 基础训练（Qwen1.5-7B，8 专家，top-2 路由）
python scripts/train.py

# 使用更小模型快速测试
python scripts/train.py --model_name Qwen/Qwen1.5-0.5B --no_4bit

# 自定义 MoE 参数
python scripts/train.py \
    --model_name Qwen/Qwen1.5-7B \
    --num_experts 4 --top_k 2 \
    --r 8 --lora_alpha 16 \
    --num_epochs 3 --batch_size 2 \
    --learning_rate 2e-4
```

## 配置说明

### MoELoRAConfig

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `r` | 16 | LoRA 秩 |
| `lora_alpha` | 32 | LoRA 缩放因子，实际缩放 = α/r |
| `num_experts` | 8 | 专家总数 |
| `top_k` | 2 | 每个 token 激活的专家数 |
| `lora_dropout` | 0.1 | LoRA dropout |
| `expert_dropout` | 0.0 | 专家级 dropout（训练时随机丢弃专家） |
| `use_noisy_router` | True | 训练时注入噪声以促进探索 |
| `load_balance_weight` | 0.01 | 负载均衡损失权重 |
| `router_z_loss_weight` | 0.001 | Router Z-loss 权重 |
| `target_modules` | q_proj,k_proj,v_proj,o_proj,... | 应用 MoE-LoRA 的模块 |
| `init_lora_weights` | gaussian | 初始化方式 (gaussian/kaiming/pissa) |

### TrainingConfig

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `model_name` | Qwen/Qwen1.5-7B | 基础模型 |
| `use_4bit` | True | 4-bit 量化（节省显存） |
| `learning_rate` | 2e-4 | 学习率 |
| `per_device_train_batch_size` | 4 | 每卡 batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |

## 消融实验设计

| 实验 | num_experts | 含义 |
|------|-------------|------|
| E1 | 1 | Baseline（退化到标准 LoRA） |
| E2 | 2 | 最少 MoE 配置 |
| E3 | 4 | 中小规模 MoE |
| E4 | 8 | 默认配置 |
| E5 | 16 | 大规模 MoE |

固定 `r=16, top_k=2, lora_alpha=32`，评估指标包括困惑度、专家利用率、路由熵、推理速度。

## 关键设计

### 路由器带噪声 Top-k 门控

训练时注入可学习的噪声 $\text{Softplus}(W_{noise} \cdot x) \cdot \mathcal{N}(0,1)$，增强探索能力。推理时使用纯净的 Top-k 选择。

### 负载均衡损失

$$\mathcal{L}_{balance} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是路由器分配给专家 $i$ 的平均概率。鼓励均匀分配。

### 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{LM} + \alpha \cdot \mathcal{L}_{balance} + \beta \cdot \mathcal{L}_{z}$$

## 内置权重初始化

| 方法 | 策略 | 特点 |
|------|------|------|
| **gaussian** (推荐) | A: kaiming uniform, B: 零 | 训练初期等价于原始模型 |
| **kaiming** | A, B 均 kaiming uniform | 标准初始化 |
| **pissa** | 正交初始化 | 保留主成分方向 |

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [MoE-LoRA (arxiv)](https://arxiv.org/abs/2404.11590)
- [Qwen1.5](https://huggingface.co/Qwen)
# Mixture-of-Experts-LoRA-for-Qwen1.5
