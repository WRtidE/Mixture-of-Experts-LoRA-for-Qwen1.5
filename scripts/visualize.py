"""
MoE-LoRA 消融实验可视化脚本

读取 evaluation_summary.json，生成论文级图表：
1. 专家数量 vs 困惑度 (柱状图)
2. 专家数量 vs 可训练参数量 (双轴图)
3. 专家利用率热力图
4. 综合对比雷达图

输出: ablation_results/figures/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np

# 尝试导入绘图库，缺失时给出安装提示
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，服务器友好
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[警告] 未安装 matplotlib，请运行: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ---- 中文字体设置 ----
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "SimHei", "DejaVu Sans", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False


def load_results(results_path: str) -> List[Dict]:
    """加载评估结果"""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_perplexity(results: List[Dict], output_dir: Path):
    """图1: 专家数量 vs 困惑度"""
    experts = [r["num_experts"] for r in results]
    ppls = [r["perplexity"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(experts)))
    bars = ax.bar(range(len(experts)), ppls, color=colors, edgecolor="white", linewidth=0.8)

    # 数值标注
    for bar, ppl in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ppls) * 0.01,
            f"{ppl:.1f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels([str(e) for e in experts])
    ax.set_xlabel("Number of Experts", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=13)
    ax.set_title("Impact of Expert Count on Perplexity\n(lower is better)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 标注最佳
    best_idx = np.argmin(ppls)
    ax.annotate(
        "Best", xy=(best_idx, ppls[best_idx]),
        xytext=(best_idx + 0.3, ppls[best_idx] + (max(ppls) - min(ppls)) * 0.15),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        fontsize=12, color="red", fontweight="bold",
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ppl_vs_experts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {output_dir / 'ppl_vs_experts.png'}")


def plot_params_and_speed(results: List[Dict], output_dir: Path):
    """图2: 专家数量 vs 参数量 + 推理速度 (双轴)"""
    experts = [r["num_experts"] for r in results]
    trainable_params = [r["params"]["trainable"] / 1e6 for r in results]  # 百万
    speeds = [r.get("inference_speed_tps", None) for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左轴: 可训练参数量
    color1 = "#2E86AB"
    ax1.set_xlabel("Number of Experts", fontsize=13)
    ax1.set_ylabel("Trainable Parameters (M)", color=color1, fontsize=13)
    line1 = ax1.plot(
        experts, trainable_params,
        "o-", color=color1, linewidth=2.5, markersize=10,
        label="Trainable Params (M)", zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.2, linestyle="--")

    # 标注参数值
    for e, p in zip(experts, trainable_params):
        ax1.annotate(
            f"{p:.1f}M", (e, p),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9, color=color1,
        )

    # 右轴: 推理速度
    if any(s is not None for s in speeds):
        ax2 = ax1.twinx()
        color2 = "#A23B72"
        ax2.set_ylabel("Inference Speed (tokens/sec)", color=color2, fontsize=13)
        valid_speeds = [(e, s) for e, s in zip(experts, speeds) if s is not None]
        if valid_speeds:
            e_vals, s_vals = zip(*valid_speeds)
            line2 = ax2.plot(
                e_vals, s_vals,
                "s--", color=color2, linewidth=2.5, markersize=10,
                label="Inference Speed (tok/s)", zorder=3,
            )
            ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Trainable Parameters & Inference Speed vs Expert Count", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "params_speed_vs_experts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {output_dir / 'params_speed_vs_experts.png'}")


def plot_utilization_heatmap(results: List[Dict], output_dir: Path):
    """图3: 专家利用率热力图"""
    # 收集每层每个专家的使用频率，并跨实验聚合
    # 取利用率最高的前 12 层展示
    all_layer_data = {}

    for r in results:
        num_experts = r["num_experts"]
        per_layer = r["expert_utilization"].get("per_layer", {})

        for layer_name, metrics in per_layer.items():
            if layer_name not in all_layer_data:
                all_layer_data[layer_name] = {}
            all_layer_data[layer_name][num_experts] = metrics["usage_distribution"]

    if not all_layer_data:
        print("  [跳过] 无利用率数据")
        return

    # 对于每个 num_experts 值，跨层平均专家分布
    experts_list = sorted(set(r["num_experts"] for r in results))
    max_experts = max(experts_list)

    # 构建矩阵: layers x experts
    heatmap_data = {}
    for layer_name, exp_data in all_layer_data.items():
        avg_dist = np.zeros(max_experts)
        count = 0
        for n_exp, dist in exp_data.items():
            if n_exp == max_experts:
                avg_dist = np.array(dist)
                count += 1
        if count > 0:
            heatmap_data[layer_name] = avg_dist

    if not heatmap_data:
        return

    # 取前12层
    sorted_layers = sorted(heatmap_data.keys())[:12]
    matrix = np.array([heatmap_data[l] for l in sorted_layers])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(max_experts))
    ax.set_xticklabels([f"E{i}" for i in range(max_experts)])
    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels([f"L{i+1}" for i in range(len(sorted_layers))], fontsize=8)
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"Expert Usage Distribution Heatmap\n(num_experts={max_experts})", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Usage Probability", fontsize=11)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "utilization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {output_dir / 'utilization_heatmap.png'}")


def plot_radar_comparison(results: List[Dict], output_dir: Path):
    """图4: 综合对比雷达图"""
    # 指标: 困惑度(逆), 利用率, 参数效率, 分布熵
    metrics_names = ["Perplexity⁻¹", "Utilization", "Param Efficiency", "Entropy"]
    num_metrics = len(metrics_names)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for r in results:
        num_experts = r["num_experts"]

        # 归一化各指标到 [0, 1]
        ppl = r["perplexity"]
        util = r["expert_utilization"]["avg_utilization_rate"]
        entropy = r["expert_utilization"]["avg_normalized_entropy"]
        param_ratio = r["params"]["ratio"] / 100.0

        # 困惑度取逆 (越低越好 -> 越高越好)
        # 用所有结果归一化
        all_ppls = [x["perplexity"] for x in results]
        ppl_inv = 1.0 - (ppl - min(all_ppls)) / (max(all_ppls) - min(all_ppls) + 1e-8)

        values = [ppl_inv, util, param_ratio, entropy]
        values += values[:1]  # 闭合

        ax.plot(angles, values, "o-", linewidth=2, label=f"E={num_experts}", markersize=6)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comprehensive Comparison Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {output_dir / 'radar_comparison.png'}")


def plot_summary_table(results: List[Dict], output_dir: Path):
    """图5: 汇总对比表格"""
    fig, ax = plt.subplots(figsize=(12, 3 + len(results) * 0.4))
    ax.axis("off")

    headers = ["Experts", "Perplexity ↓", "Utilization ↑", "Entropy ↑",
               "Trainable (M)", "Ratio (%)", "Speed (tok/s)"]
    rows = []

    for r in results:
        ppl = r["perplexity"]
        util = f"{r['expert_utilization']['avg_utilization_rate']:.3f}"
        entropy = f"{r['expert_utilization']['avg_normalized_entropy']:.3f}"
        params_m = r["params"]["trainable"] / 1e6
        ratio = r["params"]["ratio"]
        speed = f"{r['inference_speed_tps']:.1f}" if r["inference_speed_tps"] else "N/A"

        rows.append([
            str(r["num_experts"]),
            f"{ppl:.1f}",
            util,
            entropy,
            f"{params_m:.1f}",
            f"{ratio:.2f}",
            speed,
        ])

    # 找最佳值并加粗
    best_ppl_idx = min(range(len(results)), key=lambda i: results[i]["perplexity"])

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colColours=["#f0f0f0"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # 高亮最佳行
    for j in range(len(headers)):
        cell = table[best_ppl_idx + 1, j]  # +1 for header row
        cell.set_facecolor("#e8f5e9")

    ax.set_title("Ablation Study Results Summary\n(Green row = best perplexity)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {output_dir / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="MoE-LoRA 消融实验可视化")
    parser.add_argument(
        "--results_dir", type=str, default="./ablation_results",
        help="消融实验结果根目录"
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("[错误] matplotlib 未安装，无法生成图表")
        print("请运行: pip install matplotlib seaborn")
        return

    sns.set_style("whitegrid")

    results_path = Path(args.results_dir) / "evaluation_summary.json"
    if not results_path.exists():
        print(f"[错误] 评估结果不存在: {results_path}")
        print("请先运行: python evaluate.py --results_dir", args.results_dir)
        return

    print("加载评估结果...")
    results = load_results(str(results_path))
    print(f"共 {len(results)} 组实验结果\n")

    # 按专家数量排序
    results.sort(key=lambda x: x["num_experts"])

    output_dir = Path(args.results_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("生成图表...")
    plot_perplexity(results, output_dir)
    plot_params_and_speed(results, output_dir)
    plot_utilization_heatmap(results, output_dir)
    plot_radar_comparison(results, output_dir)
    plot_summary_table(results, output_dir)

    print(f"\n全部图表已保存至: {output_dir}/")
    print("可用于论文报告的图表:")
    print(f"  1. {output_dir / 'ppl_vs_experts.png'}         — 困惑度对比")
    print(f"  2. {output_dir / 'params_speed_vs_experts.png'} — 参数量与速度")
    print(f"  3. {output_dir / 'utilization_heatmap.png'}     — 专家利用热力图")
    print(f"  4. {output_dir / 'radar_comparison.png'}        — 综合雷达图")
    print(f"  5. {output_dir / 'summary_table.png'}           — 汇总表格")


if __name__ == "__main__":
    main()
