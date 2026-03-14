from time import perf_counter
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
import numpy as np
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下安全渲染
import matplotlib.pyplot as plt
from gym.envs.registration import register
from acktr.arguments import get_args
from viz_com_scatter import plot_com_scatter

# ---------------------------------------------------------------------------
# 统计 & 输出辅助函数
# ---------------------------------------------------------------------------

def compute_stats(values):
    """对一组标量列表计算统计描述指标，返回字典。"""
    arr = np.array(values, dtype=float)
    n   = len(arr)
    mean   = float(np.mean(arr))
    std    = float(np.std(arr, ddof=1))
    var    = float(np.var(arr, ddof=1))
    se     = std / np.sqrt(n)
    ci95_lo = float(mean - 1.96 * se)
    ci95_hi = float(mean + 1.96 * se)
    return {
        "values":  [round(float(v), 6) for v in arr],
        "mean":    round(mean,   4),
        "std":     round(std,    4),
        "var":     round(var,    6),
        "ci95_lo": round(ci95_lo, 4),
        "ci95_hi": round(ci95_hi, 4),
        "min":     round(float(np.min(arr)),          4),
        "max":     round(float(np.max(arr)),          4),
        "q25":     round(float(np.percentile(arr, 25)), 4),
        "median":  round(float(np.median(arr)),       4),
        "q75":     round(float(np.percentile(arr, 75)), 4),
    }


def save_results_json(url, args, ratios, center_offsets,
                      avg_counter, avg_time, timestamp,
                      com_xs=None, com_ys=None):
    """将测试结果按指定格式序列化为 JSON 文件，返回 (路径, 结果字典)。"""
    n = len(ratios)
    container_size = list(args.container_size) if hasattr(args, 'container_size') else [10, 10, 10]
    result = {
        "meta": {
            "model":           url,
            "data":            './dataset/' + args.data_name,
            "cases":           n,
            "timestamp":       timestamp,
            "env_name":        args.env_name,
            "enable_rotation": args.enable_rotation,
            "container_size":  container_size,
        },
        "space_utilization": compute_stats(ratios),
        "center_offset":     compute_stats(center_offsets),
        "throughput": {
            "avg_items_per_case": round(avg_counter / n,          2),
            "avg_time_per_case":  round(avg_time    / n,          4),
            "avg_time_per_item":  round(avg_time    / avg_counter, 4),
        },
    }
    # 写入质心坐标（供 viz_com_scatter.py 独立读取使用）
    if com_xs is not None and com_ys is not None:
        result['center_of_mass'] = {
            'xs': [round(float(v), 6) for v in com_xs],
            'ys': [round(float(v), 6) for v in com_ys],
        }
    model_stem = os.path.splitext(os.path.basename(url))[0]
    out_dir    = 'log_eval'
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, f'results_{model_stem}_{timestamp}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[save_results_json] Results saved -> {out_path}')
    return out_path, result


def plot_boxplots(result, save_dir='log_eval'):
    """绘制空间利用率 & 质心稳定性箱线图（标注 95% CI 和均值），保存为 PNG。"""
    model_stem = os.path.splitext(os.path.basename(result['meta']['model']))[0]
    timestamp  = result['meta']['timestamp']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"Model: {model_stem}   Data: {result['meta']['data']}   Cases: {result['meta']['cases']}",
        fontsize=11, y=1.01
    )

    metrics = [
        ('space_utilization', 'Space Utilization', '#4C72B0'),
        ('center_offset',     'Center Offset',     '#DD8452'),
    ]
    for ax, (key, label, color) in zip(axes, metrics):
        s    = result[key]
        data = s['values']

        # --- 箱线图主体 ---
        bp = ax.boxplot(
            data, patch_artist=True, widths=0.45,
            medianprops=dict(color='black', linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4,
                            markerfacecolor='gray', alpha=0.5),
            boxprops=dict(facecolor=color, alpha=0.55),
        )

        # --- 95% CI 水平虚线 ---
        ci_lo, ci_hi = s['ci95_lo'], s['ci95_hi']
        for y_val, ls in [(ci_lo, '--'), (ci_hi, '--')]:
            ax.axhline(y_val, color='red', linestyle=ls, linewidth=1.3, alpha=0.8)
        ax.axhspan(ci_lo, ci_hi, color='red', alpha=0.08,
                   label=f'95% CI\n[{ci_lo:.4f}, {ci_hi:.4f}]')

        # --- 均值散点 ---
        ax.scatter([1], [s['mean']], color='crimson', zorder=6,
                   s=55, marker='D', label=f'Mean: {s["mean"]:.4f}')

        # --- 右侧统计注释 ---
        stats_text = (
            f"mean   = {s['mean']:.4f}\n"
            f"std    = {s['std']:.4f}\n"
            f"median = {s['median']:.4f}\n"
            f"Q25    = {s['q25']:.4f}\n"
            f"Q75    = {s['q75']:.4f}\n"
            f"min    = {s['min']:.4f}\n"
            f"max    = {s['max']:.4f}"
        )
        ax.text(
            1.32, np.median(data), stats_text,
            transform=ax.get_xaxis_transform(),
            fontsize=8, va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='lightyellow', alpha=0.85, edgecolor='gray')
        )

        ax.set_title(label, fontsize=13, pad=8)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([label])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.45)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'boxplot_{model_stem}_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot_boxplots]     Box plot  saved -> {fig_path}')
    return fig_path


# ---------------------------------------------------------------------------

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = perf_counter()
    while True:
        box_list = env.box_creator.preview(preview_num)
        # print(box_list)
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])
        if done:
            end = perf_counter()
            print('Time cost:', end-start)
            print('Ratio:', info['ratio'])
            # 一次性获取质心坐标，再推导偏移量
            com = env.space.calculate_center_of_mass()
            if com is not None:
                com_x, com_y = float(com[0]), float(com[1])
                cx = float(env.space.plain_size[0]) / 2.0
                cy = float(env.space.plain_size[1]) / 2.0
                center_offset = float(np.sqrt((com_x - cx)**2 + (com_y - cy)**2))
            else:
                com_x = float(env.space.plain_size[0]) / 2.0
                com_y = float(env.space.plain_size[1]) / 2.0
                center_offset = 0.0
            return info['ratio'], info['counter'], end-start, default_counter/box_counter, center_offset, (com_x, com_y)
        box_counter += 1
        default_counter += int(default)

def unified_test(url,  args, pruning_threshold = 0.5):
    nmodel = nnModel(url, args)
    data_url = './dataset/' +args.data_name
    env = gym.make(args.env_name,
                    box_set=args.box_size_set,
                    container_size=args.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=args.enable_rotation,
                    data_type=args.data_type,
                    target_total=args.target_total)
    print('Env name: ', args.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', args.cases)
    print('pruning threshold: ', pruning_threshold)
    print('Known item number: ', args.preview)
    times = args.cases
    ratios = []
    center_offsets = []  # 收集质心偏移标量
    com_xs = []          # 质心 X 坐标
    com_ys = []          # 质心 Y 坐标
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate, center_offset, (cx, cy) = run_sequence(nmodel, env, args.preview, c_bound)
        avg_ratio += ratio
        ratios.append(ratio)
        center_offsets.append(center_offset)
        com_xs.append(cx)
        com_ys.append(cy)
        avg_counter += counter
        avg_time += time
        avg_drate += depen_rate

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(avg_ratio/times))
    print('average put item number: %.4f'%(avg_counter/times))
    print('average sequence time: %.4f'%(avg_time/times))
    print('average time per item: %.4f'%(avg_time/avg_counter))
    print('----------------------------------------------')
    
    # 计算并输出质心偏移量统计
    center_offsets = np.array(center_offsets)
    avg_offset = np.mean(center_offsets)
    var_offset = np.var(center_offsets)
    std_offset = np.std(center_offsets)
    min_offset = np.min(center_offsets)
    max_offset = np.max(center_offsets)
    
    print('---------- Center of Mass Statistics ----------')
    print('Average center offset: %.4f' % avg_offset)
    print('Variance of center offset: %.4f' % var_offset)
    print('Std deviation of center offset: %.4f' % std_offset)
    print('Min center offset: %.4f' % min_offset)
    print('Max center offset: %.4f' % max_offset)
    print('----------------------------------------------')

    # ---------- 保存 JSON & 箱线图 ----------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_stem = os.path.splitext(os.path.basename(url))[0]
    _, result = save_results_json(
        url, args, ratios, list(center_offsets),
        avg_counter, avg_time, timestamp,
        com_xs=com_xs, com_ys=com_ys
    )
    plot_boxplots(result, save_dir='log_eval')
    plot_com_scatter(
        com_xs, com_ys,
        container_size=args.container_size,
        output_dir='log_eval',
        model_tag=model_stem,
        timestamp=timestamp
    )

def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5  # pruning_threshold (default: 0.5)
    unified_test('pretrained_models/default_cut_2.pt', args, pruning_threshold)
    # args.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', args, pruning_threshold)

    