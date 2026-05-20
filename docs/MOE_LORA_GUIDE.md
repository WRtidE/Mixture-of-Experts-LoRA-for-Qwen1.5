# MoE-LoRA 模块详解

> 基于 Qwen 项目的 MoE-LoRA (Mixture of Experts + Low-Rank Adaptation) 实现，提供一份从宏观架构到微观实现的完整讲解。

---

## 目录

1. [整体架构](#1-整体架构)
2. [Router — 门控网络](#2-router--门控网络)
3. [MoELoRALinear — 专家增强线性层](#3-moeloralinear--专家增强线性层)
4. [MoELoRAModel — 模型包装器](#4-moeloramodel--模型包装器)
5. [训练流程](#5-训练流程)
6. [部署流程](#6-部署流程)

---

## 1. 整体架构

![040bc7968c76e3c0fbad31785545f9bb](https://azusa-img-1348009459.cos.ap-beijing.myqcloud.com/LoRAMoE.png)

### 1.1 三层设计

```
┌───────────────────────────────────────────────────┐
│                  MoELoRAModel                       │  ← 最外层包装器
│  ┌─────────────────────────────────────────────┐   │
│  │        HuggingFace 原始模型 (全部冻结)        │   │
│  │  ┌───────────────────────────────────────┐  │   │
│  │  │        MoELoRALinear (× N)            │  │   │  ← 替换后的线性层
│  │  │  ┌────────────┐ ┌──────────────────┐  │  │   │
│  │  │  │ base_linear │ │     Router       │  │  │   │  ← 底层组件
│  │  │  │   (冻结)    │ │                  │  │  │   │
│  │  │  └────────────┘ │  A_experts (K)   │  │  │   │
│  │  │                 │  B_experts (K)   │  │  │   │
│  │  │                 └──────────────────┘  │  │   │
│  │  └───────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────┘
```

### 1.2 核心公式

$$
\text{output} = Wx + \underbrace{\sum_{k \in \text{top-K}} g_k \cdot (B_k \cdot A_k \cdot x)}_{\text{MoE-LoRA 分支}} \cdot \frac{\alpha}{r}
$$

其中：
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ — 冻结的预训练权重
- $A_k \in \mathbb{R}^{r \times d_{\text{in}}}, B_k \in \mathbb{R}^{d_{\text{out}} \times r}$ — 第 $k$ 个专家的低秩矩阵
- $g_k$ — 路由器给第 $k$ 个专家的软权重
- $\alpha / r$ — LoRA 缩放因子，默认 $\alpha=32, r=16 \Rightarrow$ 缩放 = 2.0

### 1.3 数据流

```
hidden_states (B, S, d_in)
        │
        ├────────────────────────────────────────────┐
        │                                            │
        ▼                                            ▼
  base_linear (冻结 W)                        Router 门控网络
        │                                            │
        │                                routing_weights (B, S, top_k)
        │                                expert_indices  (B, S, top_k)
        │                                            │
        │                                            ▼
        │                                ┌──────────────────────┐
        │                                │   逐专家批量计算:      │
        │                                │                      │
        │                                │ Expert 0: B₀·A₀·x   │
        │                                │ Expert 1: B₁·A₁·x   │
        │                                │    ...               │
        │                                │ Expert K-1           │
        │                                │                      │
        │                                │ 按路由权重加权求和     │
        │                                └──────────────────────┘
        │                                            │
        └────────────────────┬───────────────────────┘
                             │
                             ▼
                   output = base + moe_output × (α/r) + bias
```

---

## 2. Router — 门控网络

> 文件位置: `src/moe_lora.py` 第 35 行
>
> 负责决定每个 token 应该被哪些专家处理。

### 2.1 输入输出

```
hidden_states (B, S, hidden_dim)
        │
        ▼
   ┌─────────┐
   │  Router  │
   └─────────┘
        │
        ├──> routing_weights: (B, S, top_k)    归一化后的专家权重
        ├──> expert_indices:  (B, S, top_k)    选中的专家编号
        └──> router_logits:   (B, S, num_experts)  原始 logits（用于负载均衡）
```

### 2.2 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | `int` | 必填 | 输入隐藏维度 |
| `num_experts` | `int` | 必填 | 专家总数 $E$ |
| `top_k` | `int` | 2 | 每个 token 激活的专家数 |
| `router_hidden_dim` | `int \| None` | `None` | `None`=单层 Linear，非 `None`=两层 MLP |
| `use_noisy_router` | `bool` | True | 训练时是否注入探索噪声 |
| `noise_epsilon` | `float` | 1e-2 | 噪声平滑系数 |
| `init_scale` | `float` | 0.02 | 权重初始化标准差 |

### 2.3 路由器架构

```
单层模式 (router_hidden_dim=None):
  Linear(hidden_dim, num_experts, bias=False)

两层模式 (router_hidden_dim 非 None):
  Linear(hidden_dim, router_hidden_dim) → ReLU → Linear(router_hidden_dim, num_experts)
```

### 2.4 前向传播算法

```
1. 计算干净 logits
   router_logits = self.router(hidden_states)   # (B, S, E)

2. [仅训练时] 注入噪声 (Noisy Top-k Gating)
   noise_logits = self.noise_router(hidden_states)
   noisy_logits = router_logits + N(0,1) × softplus(noise_logits)

3. Top-k 选择
   top_k_logits, expert_indices = torch.topk(noisy_logits, k=top_k)

4. Softmax 归一化
   routing_weights = softmax(top_k_logits)  # 仅在被选中的 top-k 中做
```

**噪声注入公式**：

$$
\text{noisy\_logits} = \text{logits} + \mathcal{N}(0,1) \cdot \text{softplus}(\text{noise\_logits})
$$

借鉴自 Shazeer et al. (2017) 的 Noisy Top-k Gating。训练时通过噪声鼓励路由器探索不同的专家组合，防止过早坍缩到少数专家。推理时不注入噪声。

### 2.5 负载均衡损失

MoE 训练中最关键的辅助损失，防止所有 token 涌向同一个专家：

$$
\mathcal{L}_{\text{lb}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

**Router Z-loss**（稳定训练用）：

$$
\mathcal{L}_{z} = \frac{1}{T} \sum \bigl( \text{logits} \bigr)^2
$$

| 符号 | 含义 |
|------|------|
| $E$ | 专家总数 |
| $f_i$ | 实际分配给专家 $i$ 的 token 比例（从 `expert_indices` 统计） |
| $P_i$ | 路由器 softmax 概率中专家 $i$ 的平均值 |
| $\mathcal{L}_{\text{lb}}$ | 理想最小值为 1.0（$f_i = P_i = 1/E$） |
| $\mathcal{L}_{z}$ | 惩罚过大的 logits 值 |

---

## 3. MoELoRALinear — 专家增强线性层

> 文件位置: `src/moe_lora.py` 第 174 行
>
> 整个项目的**核心模块**。将一个普通 `nn.Linear` 变为带多专家的 MoE-LoRA 层。

### 3.1 参数张量结构

| 参数 | 形状 | 可训练 | 说明 |
|------|------|--------|------|
| `base_linear.weight` | $(d_{\text{out}}, d_{\text{in}})$ | ❌ 冻结 | 原始预训练权重 |
| `A_experts` | $(K, r, d_{\text{in}})$ | ✅ | 每个专家的下投影矩阵 |
| `B_experts` | $(K, d_{\text{out}}, r)$ | ✅ | 每个专家的上投影矩阵 |
| `router.router` | 依赖于配置 | ✅ | 门控网络权重 |
| `bias` | $(d_{\text{out}},)$ | ✅ | 可选偏置 |

**可训练参数量估算**：

$$
N_{\text{params}} = K \times r \times (d_{\text{in}} + d_{\text{out}}) + N_{\text{router}}
$$

对于 Qwen-7B 的 attention 层 ($d_{\text{in}}=d_{\text{out}}=4096, r=16, K=8$)：
- 每个 MoELoRA 层: $8 \times 16 \times (4096+4096) \approx 1.05M$ 参数
- 对比原始全量微调参数量大幅降低

### 3.2 初始化策略

| 方法 | A 矩阵初始化 | B 矩阵初始化 | 适用场景 |
|------|-------------|-------------|---------|
| `gaussian` (默认) | kaiming uniform | **全零** | 训练初期 $\Delta W=0$，等价于原始模型，最安全 |
| `kaiming` | kaiming uniform | kaiming uniform | 从头训练风格，适合大幅度微调 |
| `pissa` | 正交初始化 | 正交初始化 | SVD 主成分保留，保持预训练知识 |

**默认 `gaussian` 的原因**：B=0 意味着 MoE-LoRA 分支初始对输出无贡献，模型从"原样"开始逐步学习。这是 LoRA 论文推荐的稳定策略。

### 3.3 双重 Dropout 机制

| 类型 | 默认值 | 作用对象 | 效果 |
|------|--------|----------|------|
| `lora_dropout` | 0.1 | hidden states | 标准 dropout，防过拟合 |
| `expert_dropout` | 0.0 | 整个专家 | 随机屏蔽专家，强制互补学习 |

### 3.4 前向传播详解

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 基础路径                                          │
│   base_output = base_linear(hidden_states)   # Wx        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: 路由                                              │
│   routing_weights, expert_indices, router_logits         │
│   = router(hidden_states)                                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: 展平 + Dropout                                    │
│   hidden_flat = hidden.view(-1, d_in)    # (B×S, d_in)   │
│   hidden_dropped = lora_dropout(hidden_flat)              │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: 专家级 Dropout (训练时)                            │
│   dropout_mask = rand(K) > expert_dropout                 │
│   被屏蔽的专家在循环中直接跳过                               │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: 逐专家批量计算 (核心循环)                           │
│                                                          │
│   for expert_idx in range(num_experts):                  │
│     (1) 找出路由到该专家的 token 位置                       │
│         expert_mask = (indices == expert_idx)             │
│         tokens = hidden_dropped[expert_mask.any(-1)]      │
│                                                          │
│     (2) 低秩双分解 (一次性批处理所有匹配 token)              │
│         A_out = tokens @ A[expert_idx]^T  # (N, r)       │
│         B_out = A_out @ B[expert_idx]^T  # (N, d_out)    │
│                                                          │
│     (3) 聚合路由权重                                       │
│         weights = sum(routing_weights × expert_mask)      │
│                                                          │
│     (4) in-place 加权累加                                  │
│         moe_output[matched] += B_out × weights            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: 缩放 + 合并                                       │
│   moe_output *= alpha / r                                │
│   output = base_output + moe_output + bias               │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 7: 辅助损失 (训练时)                                  │
│   self._last_losses = router.compute_load_balance_loss() │
└─────────────────────────────────────────────────────────┘
```

**逐专家计算的精巧之处**：不逐 token 计算，而是**按专家聚合后批量矩阵乘法**。当某专家被大量 token 选中时，GPU 可以高效利用。这是典型的"用遍历专家换取 GPU 批量吞吐"的 trade-off。

### 3.5 权重合并/分离

```python
merge_weights():   W = W + mean(B_k @ A_k) × α/r
unmerge_weights(): W = W - mean(B_k @ A_k) × α/r
```

> ⚠️ **局限性**：合并后所有专家被平均成一个 $\Delta W$，丢失了动态路由能力。仅用于需要导出为标准模型格式的简化部署场景。

---

## 4. MoELoRAModel — 模型包装器

> 文件位置: `src/moe_lora.py` 第 433 行
>
> 最外层的包装器，自动将 HuggingFace 模型中的目标 `nn.Linear` 层替换为 `MoELoRALinear`。

### 4.1 构造函数

```python
def __init__(self, model, moe_lora_config, target_modules=None):
```

| 参数 | 说明 |
|------|------|
| `model` | HuggingFace 预训练模型（如 `Qwen2ForCausalLM`） |
| `moe_lora_config` | `MoELoRAConfig` 数据类，包含所有超参数 |
| `target_modules` | 要替换的模块名列表，如 `["q_proj", "k_proj", "v_proj", "o_proj"]` |

### 4.2 `_apply_moe_lora()` — 非侵入式替换引擎

```
遍历 self.model.named_modules():
    │
    ├── "model.layers.5.self_attn.q_proj"
    │         │
    │         └── 最后一段 = "q_proj" ∈ target_modules? 是 nn.Linear?
    │                    │
    │                    YES:
    │                    ├── parent = "model.layers.5.self_attn"
    │                    ├── child  = "q_proj"
    │                    ├── 创建 MoELoRALinear(base_linear=原Linear)
    │                    ├── setattr(parent, child, moe_lora_layer)
    │                    └── 记录到 self.moe_lora_layers[name]
    │
    └── 跳过不匹配的模块
```

**为什么用 `setattr` 而非重新构建模型？**

HuggingFace 模型的 `forward` 中已写好 `self.q_proj(x)` 调用，直接替换属性后前向传播自动走 `MoELoRALinear.forward`，**无需修改任何原始模型代码**。

### 4.3 `forward()` — 辅助损失后收集

```python
def forward(self, input_ids, attention_mask, labels, **kwargs):
    # 1. 原始模型前向（内部自动调用各 MoELoRALinear.forward）
    outputs = self.model(input_ids, attention_mask, labels)

    # 2. 遍历所有 MoE-LoRA 层，收集辅助损失
    total_lb = sum(layer.get_last_losses()[0] for layer in self.moe_lora_layers.values())
    total_z  = sum(layer.get_last_losses()[1] for layer in self.moe_lora_layers.values())

    return outputs, total_lb, total_z
```

**后收集模式**：每个 `MoELoRALinear` 在 `forward` 时把辅助损失存到 `self._last_losses`，父级在模型返回后统一收集。这样**不污染原始模型的 forward 签名**。

### 4.4 保存/加载

| 方法 | 行为 | 文件大小对比 (Qwen-7B) |
|------|------|------------------------|
| `save_moe_lora(path)` | 只保存 `requires_grad=True` 的参数 | ~几百 MB |
| 全量保存 | 保存所有权重 | ~14 GB |
| `load_moe_lora(path)` | 用 `strict=False` 加载，兼容不同配置 | — |

### 4.5 `merge_and_unload()`

```python
def merge_and_unload(self):
    for layer in self.moe_lora_layers.values():
        layer.merge_weights()   # W += mean(B_k @ A_k) * α/r
    return self.model            # 返回纯净的原始模型
```

返回一个**不含任何 MoE 结构的普通模型**，可直接用于标准推理、导出 ONNX、TensorRT 等。

---

## 5. 训练流程

### 5.1 典型训练代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MoELoRAConfig
from src.moe_lora import MoELoRAModel

# 1. 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# 2. 配置 MoE-LoRA
config = MoELoRAConfig(
    r=16,
    lora_alpha=32,
    num_experts=8,
    top_k=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    expert_dropout=0.0,
    load_balance_weight=0.01,
    router_z_weight=0.001,
)

# 3. 包装模型
moe_model = MoELoRAModel(model, config)

# 4. 训练循环
optimizer = torch.optim.AdamW(moe_model.parameters(), lr=1e-4)

for batch in dataloader:
    outputs, lb_loss, z_loss = moe_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    total_loss = outputs.loss + 0.01 * lb_loss + 0.001 * z_loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5.2 损失组成

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{lb}} \cdot \mathcal{L}_{\text{lb}} + \lambda_{z} \cdot \mathcal{L}_{z}
$$

| 损失项 | 含义 | 推荐权重 |
|--------|------|----------|
| $\mathcal{L}_{\text{LM}}$ | 语言建模交叉熵损失 | $1.0$ |
| $\mathcal{L}_{\text{lb}}$ | 负载均衡损失 | $0.01$ |
| $\mathcal{L}_{z}$ | Router Z-loss | $0.001$ |

---

## 6. 部署流程

### 方案 A: 合并部署 (推荐用于生产)

```python
# 训练完后
moe_model.save_moe_lora("checkpoints/moe_lora_weights.pt")

# 部署时
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
moe_model = MoELoRAModel(model, config)
moe_model.load_moe_lora("checkpoints/moe_lora_weights.pt")

# 合并权重，得到普通模型
plain_model = moe_model.merge_and_unload()

# 用标准方式推理/导出
plain_model.generate(...)
```

### 方案 B: 保留 MoE 结构部署

```python
# 推理时保持动态路由
moe_model.eval()  # 自动关闭噪声注入

with torch.no_grad():
    outputs, _, _ = moe_model(input_ids, attention_mask)
```

---

## 附录 A: 关键设计决策一览

| 设计决策 | 采用方案 | 理由 |
|----------|----------|------|
| 路由器架构 | 单层/两层可切换 | 小模型单层够用，大模型两层更灵活 |
| 训练探索 | Noisy Top-k Gating | 防止路由坍塌到少数专家 |
| 初始化 | B=0, A=kaiming | 训练初期等价于原始模型，最稳定 |
| 计算策略 | 按专家聚合批量矩阵乘法 | 对 GPU 友好的 trade-off |
| 替换方式 | `setattr` 就地替换 | 无需修改原始模型代码 |
| 损失收集 | Post-hoc `_last_losses` | 不污染 forward 签名 |
| 检查点保存 | 仅保存可训练参数 | 文件小 100x+ |

## 附录 B: 参考文献

- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- **MoE**: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- **Switch Transformers**: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
