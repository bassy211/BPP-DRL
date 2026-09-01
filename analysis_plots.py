"""
analysis_plots.py
─────────────────
置信区间计算、JSON 结果导出、单模型箱线图绘制。

用法一：在 unified_test.py 中调用
    from analysis_plots import run_full_analysis
    run_full_analysis(url, data_url, args, ratios, center_offsets,
                      avg_counter, avg_time, times)

用法二：对已有 JSON 文件重新生成图表
    python analysis_plots.py results/no_R_FH_20260309_160247.json
"""

import json
import os
import sys
import numpy as np
from datetime import datetime
from scipy import stats


# ─────────────────────────────────────────
# 1. 置信区间计算
# ─────────────────────────────────────────
def conf_interval_95(data):
    """返回 (mean, ci_lo, ci_hi)，基于 Student's t 分布。"""
    data = np.asarray(data)
    mean = np.mean(data)
    se   = stats.sem(data)
    ci   = stats.t.interval(0.95, df=len(data) - 1, loc=mean, scale=se)
    return float(mean), float(ci[0]), float(ci[1])


# ─────────────────────────────────────────
# 2. JSON 导出
# ─────────────────────────────────────────
def export_json(url, data_url, args, ratios, center_offsets,
                com_xs, com_ys, avg_counter, avg_time, times, output_dir, timestamp):
    """将测试结果序列化为 JSON，返回保存路径。"""
    ratios         = np.asarray(ratios)
    center_offsets = np.asarray(center_offsets)

    ratio_mean,  ratio_ci_lo,  ratio_ci_hi  = conf_interval_95(ratios)
    offset_mean, offset_ci_lo, offset_ci_hi = conf_interval_95(center_offsets)

    model_tag = f"{getattr(args, 'algorithm', 'heuristic')}" if url is None else os.path.splitext(os.path.basename(url))[0]
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'{model_tag}_{timestamp}.json')

    result_data = {
        "meta": {
            "model":           url,
            "data":            data_url,
            "cases":           times,
            "timestamp":       timestamp,
            "env_name":        args.env_name,
            "container_size":  args.container_size if hasattr(args, 'container_size') else [10, 10, 10],
            "enable_rotation": args.enable_rotation,
        },
        "space_utilization": {
            "values":   ratios.tolist(),
            "mean":     ratio_mean,
            "std":      float(np.std(ratios)),
            "var":      float(np.var(ratios)),
            "ci95_lo":  ratio_ci_lo,
            "ci95_hi":  ratio_ci_hi,
            "min":      float(np.min(ratios)),
            "max":      float(np.max(ratios)),
            "q25":      float(np.percentile(ratios, 25)),
            "median":   float(np.median(ratios)),
            "q75":      float(np.percentile(ratios, 75)),
        },
        "center_of_mass": {
            "xs": np.asarray(com_xs).tolist() if com_xs is not None else [],
            "ys": np.asarray(com_ys).tolist() if com_ys is not None else []
        },
        "center_offset": {
            "values":   center_offsets.tolist(),
            "mean":     offset_mean,
            "std":      float(np.std(center_offsets)),
            "var":      float(np.var(center_offsets)),
            "ci95_lo":  offset_ci_lo,
            "ci95_hi":  offset_ci_hi,
            "min":      float(np.min(center_offsets)),
            "max":      float(np.max(center_offsets)),
            "q25":      float(np.percentile(center_offsets, 25)),
            "median":   float(np.median(center_offsets)),
            "q75":      float(np.percentile(center_offsets, 75)),
        },
        "throughput": {
            "avg_items_per_case": float(avg_counter / times),
            "avg_time_per_case":  float(avg_time / times),
            "avg_time_per_item":  float(avg_time / avg_counter),
        },
    }
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Results saved to: {json_path}')
    return json_path


# ─────────────────────────────────────────
# 3. 单模型箱线图
# ─────────────────────────────────────────
def plot_single_boxplots(ratios, center_offsets, model_tag, timestamp, output_dir):
    """
    绘制空间利用率与质心稳定性的箱线图（含 95% CI 标注），
    适用于 SCI 论文插图风格（300 DPI）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _ci95(data):
        se = stats.sem(data)
        return stats.t.interval(0.95, df=len(data) - 1,
                                loc=np.mean(data), scale=se)

    PALETTE = {'box': '#4878CF', 'median': '#E24A33',
               'ci':  '#2CA02C', 'mean':   '#FF7F0E'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(f'Performance Evaluation — {model_tag}',
                 fontsize=14, fontweight='bold', y=1.01)

    for ax, data, ylabel, subtitle in zip(
        axes,
        [ratios, center_offsets],
        ['Space Utilization Ratio', 'Center-of-Mass Offset Ratio'],
        ['(a) Space Utilization',   '(b) CoM Stability'],
    ):
        ax.boxplot(
            data, patch_artist=True, widths=0.45, showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='gray',
                            markersize=3, linestyle='none', alpha=0.5),
            medianprops=dict(color=PALETTE['median'], linewidth=2.5),
            whiskerprops=dict(linestyle='--', linewidth=1.2, color='#555555'),
            capprops=dict(linewidth=1.8, color='#333333'),
            boxprops=dict(facecolor=PALETTE['box'], alpha=0.55,
                          linewidth=1.5, color='#2255AA'),
        )

        mean_val = np.mean(data)
        ci_lo, ci_hi = _ci95(data)

        ax.scatter([1], [mean_val], marker='^', color=PALETTE['mean'],
                   s=80, zorder=5, label=f'Mean = {mean_val:.4f}')
        ax.errorbar([1], [mean_val],
                    yerr=[[mean_val - ci_lo], [ci_hi - mean_val]],
                    fmt='none', ecolor=PALETTE['ci'],
                    elinewidth=2, capsize=8, capthick=2,
                    label=f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]')

        stats_text = (
            f'n = {len(data)}\n'
            f'Median = {np.median(data):.4f}\n'
            f'IQR = [{np.percentile(data,25):.4f}, {np.percentile(data,75):.4f}]\n'
            f'Std = {np.std(data):.4f}'
        )
        ax.text(1.30, np.percentile(data, 75), stats_text,
                fontsize=8.5, va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='#AAAAAA', alpha=0.9))

        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([model_tag], fontsize=9, rotation=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8.5, loc='upper right',
                  framealpha=0.85, edgecolor='#CCCCCC')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'boxplot_{model_tag}_{timestamp}.pdf')
    png_path = os.path.join(output_dir, f'boxplot_{model_tag}_{timestamp}.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f'Box plots saved to:\n  PDF: {pdf_path}\n  PNG: {png_path}')
    print('----------------------------------------------')


# ─────────────────────────────────────────
# 4. 一键调用入口（供 unified_test.py 使用）
# ─────────────────────────────────────────
def run_full_analysis(url, data_url, args, ratios, center_offsets,
                      avg_counter, avg_time, times, com_xs=None, com_ys=None):
    """
    计算 95% CI、打印统计摘要、导出 JSON、绘制单模型箱线图。
    在 unified_test.py 中解注释后直接调用即可。
    """
    ratios         = np.asarray(ratios)
    center_offsets = np.asarray(center_offsets)

    ratio_mean,  ratio_ci_lo,  ratio_ci_hi  = conf_interval_95(ratios)
    offset_mean, offset_ci_lo, offset_ci_hi = conf_interval_95(center_offsets)

    print('space utilization 95%% CI: [%.4f, %.4f]' % (ratio_ci_lo, ratio_ci_hi))
    print('space utilization variance: %.4f' % np.var(ratios))
    print('space utilization std dev:  %.4f' % np.std(ratios))
    print('----------------------------------------------')
    print('Stability Analysis:')
    print('center offset 95%% CI: [%.4f, %.4f]' % (offset_ci_lo, offset_ci_hi))
    print('center offset variance: %.4f' % np.var(center_offsets))
    print('center offset std dev:  %.4f' % np.std(center_offsets))
    print('----------------------------------------------')

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = f"{getattr(args, 'algorithm', 'heuristic')}" if url is None else os.path.splitext(os.path.basename(url))[0]

    export_json(url, data_url, args, ratios, center_offsets,
                com_xs, com_ys, avg_counter, avg_time, times, output_dir, timestamp)

    plot_single_boxplots(ratios, center_offsets, model_tag, timestamp, output_dir)


# ─────────────────────────────────────────
# 5. 独立运行：从已有 JSON 重新生成图表
#    python analysis_plots.py results/xxx.json
# ─────────────────────────────────────────
def analyze_json(json_path):
    """从已保存的 JSON 文件重新生成置信区间摘要和箱线图。"""
    with open(json_path, 'r') as f:
        d = json.load(f)

    ratios         = np.array(d['space_utilization']['values'])
    center_offsets = np.array(d['center_offset']['values'])
    model_tag      = os.path.splitext(os.path.basename(
                         d['meta']['model']))[0]
    timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir     = os.path.dirname(os.path.abspath(json_path))

    r_mean, r_lo, r_hi = conf_interval_95(ratios)
    o_mean, o_lo, o_hi = conf_interval_95(center_offsets)

    print(f'Model : {model_tag}')
    print(f'Space Utilization : mean={r_mean:.4f}  95%CI=[{r_lo:.4f},{r_hi:.4f}]'
          f'  std={np.std(ratios):.4f}')
    print(f'CoM Offset        : mean={o_mean:.4f}  95%CI=[{o_lo:.4f},{o_hi:.4f}]'
          f'  std={np.std(center_offsets):.4f}')
    print('----------------------------------------------')

    plot_single_boxplots(ratios, center_offsets, model_tag, timestamp, output_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python analysis_plots.py <path/to/result.json>')
        sys.exit(1)
    analyze_json(sys.argv[1])
