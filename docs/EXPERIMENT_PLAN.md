# MoE-LoRA 消融实验 — 项目计划与分工

> **项目主题**：大语言模型参数高效微调技术研究 — 基于 MoE-LoRA 的专家数量消融实验  
> **时间节点**：2026年5月20日 → 5月底/6月初 （约2周）  
> **基础模型**：Qwen1.5-0.5B / Qwen1.5-7B  
> **数据集**：Alpaca (SFT) + MMLU / HellaSwag (评估)

---

## 一、项目分工（4人小组）

| 角色 | 成员 | 核心任务 | 产出物 |
|------|------|----------|--------|
| **A: 实验执行** | 成员A | 环境搭建、运行全部消融实验、记录训练日志 | 模型检查点、loss曲线、训练日志 |
| **B: 评估分析** | 成员B | 下游任务评估、专家路由分析、指标收集 | 评估指标表、路由分析报告 |
| **C: 可视化与报告** | 成员C | 图表绘制、论文初稿撰写、结论归纳 | 图表集、报告 LaTeX/Word |
| **D: 文献调研** | 成员D | 背景调研、相关工作整理、创新点讨论 | 文献综述、理论分析章节 |

---

## 二、时间线

```
Week 1 (5/20-5/25): 环境搭建 + 首次训练
├── D1: 全体 — 确认分工，讨论实验设计
├── D1-D2: A — 配置 conda 环境，下载 Qwen1.5-0.5B 模型
├── D2-D3: A — 运行小规模预实验（num_experts=4, epochs=1）
├── D3-D4: A — 启动完整消融实验（5组），后台运行
├── D4-D5: D — 完成文献调研初稿，共享给全组
└── D5-D7: A — 监控训练进度，收集 checkpoint

Week 2 (5/26-6/1): 评估 + 报告撰写
├── D1-D2: B — 对 5 组模型运行 evaluate.py，收集指标
├── D2-D3: C — 运行 visualize.py 生成所有图表
├── D3-D4: 全体 — 讨论结果，确认关键发现
├── D4-D6: C & D — 撰写报告（C 写实验部分，D 写背景/相关工作）
└── D6-D7: 全体 — 审阅、修改、定稿

Week 3 (6/2-6/5): 收尾 & 答辩准备
├── D1-D2: 最终修改、格式检查
└── D3-D5: 答辩 PPT 制作、演练
```

---

## 三、消融实验设计

### 3.1 实验变量

| 参数 | 取值 | 说明 |
|------|------|------|
| `num_experts` | **1, 2, 4, 8, 16** | 核心消融变量 |
| `top_k` | 2 (固定) | 每个 token 激活的 top-k 专家 |
| `r` (LoRA rank) | 16 (固定) | 低秩分解维度 |
| `lora_alpha` | 32 (固定) | 缩放系数 |

> `num_experts=1` 退化到标准 LoRA，作为 baseline

### 3.2 实验矩阵

| 实验编号 | num_experts | top_k | r | 预计参数量 | 预计训练时间(0.5B) |
|----------|-------------|-------|---|-----------|-------------------|
| E1 | 1 (baseline) | 1 | 16 | ~4M | ~1h |
| E2 | 2 | 2 | 16 | ~8M | ~1.2h |
| E3 | 4 | 2 | 16 | ~16M | ~1.5h |
| E4 | 8 | 2 | 16 | ~32M | ~2h |
| E5 | 16 | 2 | 16 | ~64M | ~3h |

### 3.3 评估指标

| 指标 | 含义 | 方向 |
|------|------|------|
| **Perplexity** | 验证集困惑度 | ↓ 越低越好 |
| **专家利用率** | 实际被使用的专家比例 | ↑ 越高越均匀 |
| **路由熵** | 路由分布的信息熵 | ↑ 越均匀 |
| **推理速度** | tokens/sec | ↑ 越快越好 |
| **可训练参数** | 新增参数总量 | 在效率和性能间平衡 |

---

## 四、执行命令

```bash
# 步骤1: 环境准备（成员A）
conda activate qwen
pip install -r requirements.txt
pip install matplotlib seaborn  # 用于可视化

# 步骤2: 快速验证脚本是否正常（单组小实验）
python train.py \
    --model_name Qwen/Qwen1.5-0.5B \
    --num_experts 4 --top_k 2 --r 16 \
    --num_epochs 1 --no_4bit \
    --output_dir ./test_run

# 步骤3: 运行全部消融实验（自动串行执行 5 组）
./run_ablation.sh

# 步骤4: 运行评估（成员B）
python evaluate.py --results_dir ./ablation_results

# 步骤5: 生成图表（成员C）
python visualize.py --results_dir ./ablation_results
```

---

## 五、文件结构

```
Qwen/
├── moe_lora.py              # MoE-LoRA 核心模块
├── train.py                 # 训练脚本
├── inference.py             # 推理与路由分析
├── evaluate.py              # 消融实验评估脚本  ✨新建
├── visualize.py             # 结果可视化脚本    ✨新建
├── config.py                # 配置文件
├── run_ablation.sh          # 一键消融实验      ✨新建
├── requirements.txt         # 依赖
├── README.md                # 项目说明
├── EXPERIMENT_PLAN.md       # 本文件
├── REPORT_OUTLINE.md        # 报告大纲
│
├── ablation_results/        # 消融实验结果（运行后生成）
│   ├── experts_1/           # num_experts=1 的检查点
│   ├── experts_2/
│   ├── experts_4/
│   ├── experts_8/
│   ├── experts_16/
│   ├── evaluation_summary.json
│   └── figures/             # 所有图表
└── test_run/                # 快速测试输出
```

---

## 六、预期发现与讨论方向

1. **性能饱和点**：超过某个专家数量后，困惑度改善趋于平缓
2. **路由坍缩**：少数专家可能被"废弃"（路由概率接近0），需要负载均衡损失来缓解
3. **参数效率**：专家增加带来的性能提升 vs 参数开销的 trade-off
4. **层间差异**：浅层 vs 深层的专家使用模式可能存在差异
5. **与标准 LoRA 对比**：MoE-LoRA 是否在相同参数量下优于标准 LoRA
