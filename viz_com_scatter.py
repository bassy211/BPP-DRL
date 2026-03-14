"""
viz_com_scatter.py
──────────────────
将 100 次装箱实验的质心落点绘制在容器俯视网格图上。

用法（在 unified_test.py 中解注释调用即可）：
    from viz_com_scatter import plot_com_scatter
    plot_com_scatter(com_xs, com_ys, container_size,
                     output_dir, model_tag, timestamp)

也可独立运行（从 JSON 读取）：
    python viz_com_scatter.py results/ours.json
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_com_scatter(com_xs, com_ys, container_size,
                     output_dir, model_tag, timestamp):
    """
    在 container_size[0] × container_size[1] 的网格图上绘制质心散点。

    Parameters
    ----------
    com_xs : array-like  质心 X 坐标列表（每个 case 一个值）
    com_ys : array-like  质心 Y 坐标列表
    container_size : tuple  (W, D, H)，只用 W × D 绘制俯视图
    output_dir : str
    model_tag  : str
    timestamp  : str
    """
    com_xs = np.asarray(com_xs, dtype=float)
    com_ys = np.asarray(com_ys, dtype=float)
    W, D   = float(container_size[0]), float(container_size[1])
    cx, cy = W / 2.0, D / 2.0          # 几何中心

    # ── 每个点的偏移距离（用于着色）──
    dists = np.sqrt((com_xs - cx)**2 + (com_ys - cy)**2)
    diagonal = np.sqrt(W**2 + D**2)

    # ── 画布 ──
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # ── 网格底色 ──
    ax.set_facecolor('#F8F8F8')

    # ── 主网格线（每格1单位）──
    for x in range(int(W) + 1):
        ax.axvline(x, color='#CCCCCC', linewidth=0.6, zorder=1)
    for y in range(int(D) + 1):
        ax.axhline(y, color='#CCCCCC', linewidth=0.6, zorder=1)

    # ── 容器边框 ──
    border = patches.Rectangle((0, 0), W, D,
                                linewidth=2, edgecolor='#333333',
                                facecolor='none', zorder=4)
    ax.add_patch(border)

    # ── 散点（按偏移距离着色）──
    norm   = Normalize(vmin=0, vmax=dists.max() * 1.05)
    cmap   = plt.cm.RdYlGn_r          # 绿→黄→红，偏移越大颜色越红
    colors = cmap(norm(dists))

    sc = ax.scatter(com_xs, com_ys,
                    c=dists, cmap=cmap, norm=norm,
                    s=55, alpha=0.75, zorder=5,
                    edgecolors='#444444', linewidths=0.4,
                    label=f'CoM per case (n={len(com_xs)})')

    # ── 几何中心（橙色 ★）──
    ax.plot(cx, cy, marker='*', color='#FF7F0E',
            markersize=12, markeredgewidth=0.6,
            markeredgecolor='#333333', zorder=6,
            linestyle='none',
            label=f'Geometric center ({cx:.1f}, {cy:.1f})')

    # ── Colorbar ──
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('CoM Offset Distance', fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    # ── 轴设置 ──
    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(-0.3, D + 0.3)
    ax.set_xticks(range(int(W) + 1))
    ax.set_yticks(range(int(D) + 1))
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right',
              framealpha=0.9, edgecolor='#BBBBBB')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'com_scatter_{model_tag}_{timestamp}.pdf')
    png_path = os.path.join(output_dir, f'com_scatter_{model_tag}_{timestamp}.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f'CoM scatter saved to:\n  PDF: {pdf_path}\n  PNG: {png_path}')
    print('----------------------------------------------')


# ─────────────────────────────────────────────────────
# 独立运行：python viz_com_scatter.py results/ours.json
# ─────────────────────────────────────────────────────
def _from_json(json_path):
    with open(json_path, 'r') as f:
        d = json.load(f)
    com_xs = d['center_of_mass'].get('xs')
    com_ys = d['center_of_mass'].get('ys')
    if com_xs is None or com_ys is None:
        print('ERROR: JSON does not contain center_of_mass.xs / .ys fields.')
        print('  Please re-run unified_test.py with CoM collection enabled.')
        sys.exit(1)
    container_size = d['meta'].get('container_size', [10, 10, 10])
    model_tag = os.path.splitext(os.path.basename(d['meta']['model']))[0]
    from datetime import datetime
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.dirname(os.path.abspath(json_path))
    plot_com_scatter(com_xs, com_ys, container_size,
                     output_dir, model_tag, timestamp)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python viz_com_scatter.py <path/to/result.json>')
        sys.exit(1)
    _from_json(sys.argv[1])
