"""
plot_comparison.py
==================
从多个实验 JSON 结果文件中读取数据，生成对比箱线图。

用法示例
--------
# 比较两个实验
python plot_comparison.py results/result_A.json results/result_B.json

# 自定义输出目录、文件名
python plot_comparison.py results/result_A.json results/result_B.json \
    --output ./figures --filename comparison.png

# 自定义每组标签（顺序与 JSON 文件对应）
python plot_comparison.py results/result_A.json results/result_B.json \
    --labels "Model A" "Model B"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ──────────────────────────── 数据加载 ────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_label(data: dict, fallback: str) -> str:
    """从 meta 字段提取默认标签。"""
    meta = data.get("meta", {})
    model = os.path.splitext(os.path.basename(meta.get("model", fallback)))[0]
    data_name = os.path.splitext(os.path.basename(meta.get("data", "")))[0]
    return f"{model}\n({data_name})"


# ──────────────────────────── 绘图辅助 ────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _add_ci_annotation(ax, x_pos: float, stat: dict, color: str) -> None:
    """在箱线图 x_pos 处叠加均值菱形和 95% CI 误差线。"""
    mean   = stat["mean"]
    ci_lo  = stat["ci95_lo"]
    ci_hi  = stat["ci95_hi"]
    ax.plot(x_pos, mean, marker="D", color=color,
            markersize=6, zorder=5)
    ax.errorbar(x_pos, mean,
                yerr=[[mean - ci_lo], [ci_hi - mean]],
                fmt="none", color=color,
                capsize=5, linewidth=1.5, zorder=4)


def draw_comparison(datasets: list[dict],
                    labels: list[str],
                    metric_key: str,
                    title: str,
                    ylabel: str,
                    ax: plt.Axes) -> None:
    """
    在给定 Axes 上绘制多组的对比箱线图。

    Parameters
    ----------
    datasets   : list of JSON dict
    labels     : 每组的标签
    metric_key : JSON 中的指标键（"space_utilization" | "center_offset"）
    title, ylabel : 图标题与 y 轴标签
    ax         : matplotlib Axes
    """
    positions = list(range(1, len(datasets) + 1))
    all_values = [ds[metric_key]["values"] for ds in datasets]

    bp = ax.boxplot(
        all_values,
        positions=positions,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, linestyle="none",
                        markerfacecolor="gray", alpha=0.5),
    )

    legend_handles = []
    for i, (pos, ds, label) in enumerate(zip(positions, datasets, labels)):
        color = PALETTE[i % len(PALETTE)]
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.7)

        stat = ds[metric_key]
        _add_ci_annotation(ax, pos, stat, color)

        # 构造图例 patch
        patch = plt.matplotlib.patches.Patch(
            facecolor=color, alpha=0.7,
            label=(f"{label}\n"
                   f"mean={stat['mean']:.4f}  "
                   f"95%CI=[{stat['ci95_lo']:.4f}, {stat['ci95_hi']:.4f}]")
        )
        legend_handles.append(patch)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [lbl.split("\n")[0] for lbl in labels],   # 只显示模型名，避免 x 轴过挤
        fontsize=9, rotation=15, ha="right"
    )
    ax.legend(handles=legend_handles, fontsize=8,
              loc="best", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对比多个实验 JSON 结果，生成箱线图"
    )
    parser.add_argument(
        "json_files", nargs="+",
        help="一个或多个实验 JSON 文件路径（按顺序对应各组）"
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="每组的自定义标签（数量须与 json_files 一致）"
    )
    parser.add_argument(
        "--output", default="./results",
        help="输出目录（默认: ./results）"
    )
    parser.add_argument(
        "--filename", default=None,
        help="输出文件名（默认: comparison_<timestamp>.png）"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="图片 DPI（默认: 150）"
    )
    args = parser.parse_args()

    # 加载数据
    datasets = [load_json(p) for p in args.json_files]

    # 决定每组标签
    if args.labels:
        if len(args.labels) != len(datasets):
            parser.error("--labels 数量须与 json_files 数量一致")
        labels = args.labels
    else:
        labels = [
            extract_label(ds, os.path.basename(p))
            for ds, p in zip(datasets, args.json_files)
        ]

    # 校验所有 JSON 都含有所需字段
    required = {"space_utilization", "center_offset"}
    for path, ds in zip(args.json_files, datasets):
        missing = required - set(ds.keys())
        if missing:
            raise KeyError(f"{path} 缺少字段: {missing}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(6 + 2 * len(datasets), 6))

    draw_comparison(
        datasets, labels,
        metric_key="space_utilization",
        title="Space Utilization",
        ylabel="Ratio",
        ax=axes[0],
    )
    draw_comparison(
        datasets, labels,
        metric_key="center_offset",
        title="Center-of-Mass Offset",
        ylabel="Offset",
        ax=axes[1],
    )

    n_cases = datasets[0]["meta"].get("cases", "?")
    fig.suptitle(
        f"BPP Comparison — Box Plots with 95% CI  (n={n_cases} per group)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # 保存
    os.makedirs(args.output, exist_ok=True)
    if args.filename:
        out_path = os.path.join(args.output, args.filename)
    else:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output, f"comparison_{ts}.png")

    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"[Plot] 对比图已保存至: {out_path}")


if __name__ == "__main__":
    main()
