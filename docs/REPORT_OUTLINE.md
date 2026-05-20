# 大语言模型参数高效微调技术研究 — 报告大纲

## 论文标题

**基于混合专家低秩适配（MoE-LoRA）的大语言模型参数高效微调研究**

或英文：
*Parameter-Efficient Fine-Tuning of Large Language Models via Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA)*

---

## 摘要（Abstract）

> 填空式模板，实验完成后填充数据：

大语言模型的全量微调面临算力消耗巨大、存储成本高昂等挑战。本文研究一种结合混合专家（MoE）与低秩适配（LoRA）的参数高效微调方法——MoE-LoRA。该方法冻结预训练模型主体参数，通过多个 LoRA 专家模块和可学习路由器实现稀疏激活的增量学习。我们在 Qwen1.5 模型上对专家数量进行了消融实验（K=1/2/4/8/16），在 Alpaca 数据集上微调并在 MMLU/HellaSwag 上评估。实验结果表明：（1）MoE-LoRA 在专家数量 K=__ 时达到最优困惑度 ___；（2）相比标准 LoRA，MoE-LoRA 在相同参数量下困惑度降低 ___%；（3）专家路由呈现 ___ 分布特征。本研究验证了 MoE-LoRA 在资源受限场景下进行大模型高效适配的有效性。

---

## 第1章 绪论

### 1.1 研究背景
- 大语言模型（GPT、Qwen、LLaMA）的快速发展
- 全量微调面临的三大挑战：
  - 算力消耗巨大（175B 模型全量微调需数百GB显存）
  - 存储成本高昂（每微调一个下游任务需保存完整模型副本）
  - 灾难性遗忘风险

### 1.2 研究意义
- 参数高效微调（PEFT）的技术价值
- 资源受限场景（个人设备、中小企业）的大模型适配需求
- MoE-LoRA 的理论创新潜力

### 1.3 国内外研究现状
- LoRA 系列：LoRA, AdaLoRA, QLoRA, DoRA
- MoE 系列：Switch Transformer, GLaM, Mixtral
- MoE + PEFT 交叉：MoRA, MixLoRA, MoELoRA
- 文献对比表

### 1.4 本文主要工作
- 实现 MoE-LoRA 模块并集成到 Qwen1.5
- 设计专家数量消融实验（K=1,2,4,8,16）
- 多指标综合评估
- 论文结构安排

---

## 第2章 相关技术

### 2.1 Low-Rank Adaptation (LoRA)
- 核心思想：W' = W + ΔW = W + BA
- 低秩假设：下游任务适配的增量矩阵秩较低
- 公式推导：A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, r << min(d_in, d_out)
- 缩放因子：α/r

### 2.2 Mixture of Experts (MoE)
- 稀疏 MoE 架构
- Top-k 门控机制
- 负载均衡（Load Balancing Loss）
- Switch Transformers 的 auxiliary loss

### 2.3 MoE-LoRA 融合设计
- 架构图（输入 → 冻结的 W → + 路由器 → K 个 LoRA 专家 → 输出）
- 路由公式：p_k = softmax(top-k(W_r · h + ε·softplus(W_n·h)))
- 总损失：L = L_LM + α·L_lb + β·L_z
- 创新点：相比标准 LoRA 增加了专家选择的能力；相比标准 MoE，每个专家更轻量

---

## 第3章 实验设计

### 3.1 模型与数据集
- 基础模型：Qwen1.5-0.5B（快速消融）/ Qwen1.5-7B（最终验证）
- 训练数据：Alpaca（52K 指令微调数据）
- 评估数据：MMLU（多任务语言理解）、HellaSwag（常识推理）

### 3.2 实验设置
- 消融变量：num_experts ∈ {1, 2, 4, 8, 16}
- 固定参数：r=16, top_k=2, lora_alpha=32
- 训练配置：3 epochs, lr=2e-4, cosine scheduler, batch_size=16
- 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 3.3 评估指标
| 指标 | 公式/含义 |
|------|----------|
| Perplexity | exp(cross_entropy) |
| 专家利用率 | 实际使用专家数 / 总专家数 |
| 路由熵 | -Σ p_i log(p_i) / log(K) |
| 推理速度 | tokens/sec |

---

## 第4章 实验结果与分析

### 4.1 困惑度对比

> [插入图: ppl_vs_experts.png]

- 标准 LoRA (K=1) 困惑度：___
- MoE-LoRA 最优 (K=__) 困惑度：___
- 困惑度随专家数量变化的趋势分析
- 性能饱和点讨论

### 4.2 参数量与效率分析

> [插入图: params_speed_vs_experts.png]

- 可训练参数与专家数量的线性关系
- 每百万参数的困惑度增益（参数效率）
- 推理速度的影响

### 4.3 专家路由分析

> [插入图: utilization_heatmap.png]

- 专家使用分布：均匀 vs 坍缩
- 不同层的路由模式差异
- 负载均衡损失的有效性

### 4.4 综合对比

> [插入图: radar_comparison.png, summary_table.png]

- 多维度雷达图对比
- 最佳配置推荐

### 4.5 分析与讨论
- 为什么增加专家能提升性能？
- 为什么超过某个数量会饱和？
- 路由坍缩的原因及解决方案
- 与已有工作的对比（MoRA, MixLoRA）

---

## 第5章 总结与展望

### 5.1 工作总结
- 实现了完整的 MoE-LoRA 训练与评估流水线
- 通过消融实验验证了专家数量的影响
- 发现了 __（关键发现）

### 5.2 未来工作
- 动态专家数量：根据输入难度自适应激活
- 专家合并/剪枝：训练后移除冗余专家
- 更大规模验证：在 7B/14B 模型上实验
- 多任务场景下的专家专业化

---

## 参考文献

> [1] Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
> [2] Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models." JMLR 2022.
> [3] Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.
> [4] Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
> [5] Jiang, A. Q., et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
> [6] Zadouri, T., et al. "Pushing Mixture of Experts to the Limit." arXiv:2404.02575, 2024.

---

## 附录

### A. 代码仓库说明
- 项目结构
- 环境配置
- 复现步骤

### B. 完整实验数据
- 训练 loss 曲线
- 各层专家使用统计
- 原始评估输出
