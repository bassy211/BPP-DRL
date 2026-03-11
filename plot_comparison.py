"""
plot_comparison.py
------------------
加载多个 unified_test 输出的 JSON 结果文件，
生成对比箱线图（空间利用率 & 质心稳定性），并标注 95% CI。

用法示例：
    python plot_comparison.py \
        log_eval/results_default_cut_2_20260309_143000.json \
        log_eval/results_rotation_cut_2_20260309_150000.json \
        --labels "No Rotation" "Rotation" \
        --output log_eval/comparison_20260309.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_result(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 对比箱线图
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict], labels: list[str], output: str):
    """
    results : list of result dicts (from unified_test JSON)
    labels  : display name for each result
    output  : output PNG path
    """
    assert len(results) == len(labels), "results 与 labels 数量不一致"

    n_groups = len(results)
    palette  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(6 + 3 * n_groups, 6))
    fig.suptitle('Model Comparison', fontsize=13, y=1.01)

    metrics = [
        ('space_utilization', 'Space Utilization'),
        ('center_offset',     'Center Offset'),
    ]

    for ax, (key, title) in zip(axes, metrics):
        all_data   = [r[key]['values'] for r in results]
        positions  = list(range(1, n_groups + 1))
        colors     = [palette[i % len(palette)] for i in range(n_groups)]

        # --- 箱线图 ---
        bps = ax.boxplot(
            all_data,
            positions=positions,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color='black', linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4,
                            markerfacecolor='gray', alpha=0.4),
        )
        for patch, color in zip(bps['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # --- 均值 & 95% CI ---
        for i, (r, pos, color) in enumerate(zip(results, positions, colors)):
            s      = r[key]
            ci_lo  = s['ci95_lo']
            ci_hi  = s['ci95_hi']
            mean   = s['mean']

            # CI 误差棒（独立绘制）
            ax.errorbar(pos, mean,
                        yerr=[[mean - ci_lo], [ci_hi - mean]],
                        fmt='D', color='crimson', capsize=5,
                        capthick=1.5, elinewidth=1.5, markersize=5,
                        zorder=6, label=f'{labels[i]} mean' if i == 0 else None)

            # CI 区域阴影
            ax.axvspan(pos - 0.25, pos + 0.25,
                       ymin=(ci_lo  - ax.get_ylim()[0] + 1e-9),
                       ymax=(ci_hi  - ax.get_ylim()[0] + 1e-9),
                       alpha=0.0)   # 仅用误差棒表示，不加额外阴影

            # 数值标签
            ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                    f"μ={mean:.4f}\nCI[{ci_lo:.4f},\n   {ci_hi:.4f}]",
                    ha='center', va='bottom', fontsize=7.5, family='monospace',
                    color='dimgray')

        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.45)

    # 统一 y 轴标注后重新绘制文字（需在 ax 范围稳定后）
    _add_value_labels(axes, results, labels)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_comparison] Saved -> {output}')


def _add_value_labels(axes, results, labels):
    """在 y 轴范围稳定后重新注释均值/CI 文字。"""
    metrics = ['space_utilization', 'center_offset']
    for ax, key in zip(axes, metrics):
        y_lo, y_hi = ax.get_ylim()
        y_range    = y_hi - y_lo
        for i, (r, label) in enumerate(zip(results, labels)):
            s    = r[key]
            pos  = i + 1
            text = (
                f"μ={s['mean']:.4f}\n"
                f"σ={s['std']:.4f}\n"
                f"CI [{s['ci95_lo']:.4f},\n"
                f"    {s['ci95_hi']:.4f}]"
            )
            ax.text(pos, y_hi + y_range * 0.01, text,
                    ha='center', va='bottom', fontsize=7.5,
                    family='monospace', color='dimgray')
        ax.set_ylim(y_lo, y_hi + y_range * 0.18)   # 为文字腾出空间


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='对比多个 unified_test JSON 结果，生成箱线对比图'
    )
    parser.add_argument(
        'json_files', nargs='+',
        help='一个或多个 unified_test 输出的 JSON 文件路径'
    )
    parser.add_argument(
        '--labels', nargs='*', default=None,
        help='每个文件的显示标签，顺序与 json_files 对应（默认用文件名）'
    )
    parser.add_argument(
        '--output', default=None,
        help='输出 PNG 路径（默认保存到第一个 JSON 所在目录）'
    )
    args = parser.parse_args()

    results = [load_result(p) for p in args.json_files]

    if args.labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.json_files]
    else:
        labels = args.labels

    if args.output is None:
        first_dir = os.path.dirname(os.path.abspath(args.json_files[0]))
        from datetime import datetime
        ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = os.path.join(first_dir, f'comparison_{ts}.png')
    else:
        output = args.output

    plot_comparison(results, labels, output)


if __name__ == '__main__':
    main()
