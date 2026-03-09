from time import perf_counter
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
import numpy as np
import json
import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from gym.envs.registration import register
from acktr.arguments import get_args


# ─────────────────────────── 统计辅助函数 ───────────────────────────

def compute_stats(values: list) -> dict:
    """计算一组数值的完整统计量，包含 95% 置信区间。"""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    var  = float(np.var(arr, ddof=1))
    se   = std / np.sqrt(n)
    # 双侧 95% t 置信区间
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo  = float(mean - t_crit * se)
    ci_hi  = float(mean + t_crit * se)
    return {
        "values":  [round(float(v), 6) for v in values],
        "mean":    round(mean, 6),
        "std":     round(std,  6),
        "var":     round(var,  6),
        "ci95_lo": round(ci_lo, 6),
        "ci95_hi": round(ci_hi, 6),
        "min":     round(float(np.min(arr)),            6),
        "max":     round(float(np.max(arr)),            6),
        "q25":     round(float(np.percentile(arr, 25)), 6),
        "median":  round(float(np.median(arr)),         6),
        "q75":     round(float(np.percentile(arr, 75)), 6),
    }


def save_results_json(url: str, args, ratios: list,
                      center_offsets: list,
                      avg_counter: float, avg_time: float,
                      output_dir: str = "./results") -> str:
    """将测试结果保存为 JSON 文件，返回文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(output_dir, f"result_{timestamp}.json")

    times = len(ratios)
    result = {
        "meta": {
            "model":           url,
            "data":            f"./dataset/{args.data_name}",
            "cases":           times,
            "timestamp":       timestamp,
            "env_name":        args.env_name,
            "enable_rotation": bool(args.enable_rotation),
        },
        "space_utilization": compute_stats(ratios),
        "center_offset":     compute_stats(center_offsets),
        "throughput": {
            "avg_items_per_case": round(avg_counter / times, 4),
            "avg_time_per_case":  round(avg_time    / times, 4),
            "avg_time_per_item":  round(avg_time    / avg_counter, 6),
        },
    }

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 结果已保存至: {fname}")
    return fname


def plot_boxplots(ratios: list, center_offsets: list,
                  output_dir: str = "./results",
                  tag: str = "") -> None:
    """绘制空间利用率与质心稳定性的箱线图（含 95% CI 标注）并保存。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        fname = os.path.join(output_dir, f"boxplot_{tag}_{timestamp}.png")
    else:
        fname = os.path.join(output_dir, f"boxplot_{timestamp}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def _draw_box(ax, data, title, ylabel, color):
        arr = np.array(data, dtype=float)
        n   = len(arr)
        mean = np.mean(arr)
        se   = np.std(arr, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean - t_crit * se, mean + t_crit * se

        bp = ax.boxplot(arr, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)

        # 均值点
        ax.plot(1, mean, marker="D", color="red", zorder=5, label=f"Mean={mean:.4f}")
        # 95% CI 误差线
        ax.errorbar(1, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt="none", color="red", capsize=6, linewidth=1.5,
                    label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    _draw_box(axes[0], ratios,
              "Space Utilization", "Ratio", "#4C72B0")
    _draw_box(axes[1], center_offsets,
              "Center-of-Mass Offset", "Offset", "#DD8452")

    fig.suptitle("BPP Evaluation — Box Plots with 95% CI", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Plot] 箱线图已保存至: {fname}")

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
            # 计算质心偏移量
            center_offset = env.space.get_center_offset()
            return info['ratio'], info['counter'], end-start, default_counter/box_counter, center_offset
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
    center_offsets = []  # 收集质心偏移量
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate, center_offset = run_sequence(nmodel, env, args.preview, c_bound)
        avg_ratio += ratio
        ratios.append(ratio)
        center_offsets.append(center_offset)
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
    co_arr     = np.array(center_offsets, dtype=float)
    avg_offset = float(np.mean(co_arr))
    std_offset = float(np.std(co_arr, ddof=1))

    print('---------- Center of Mass Statistics ----------')
    print('Average center offset: %.4f' % avg_offset)
    print('Std deviation of center offset: %.4f' % std_offset)
    print('----------------------------------------------')

    # ── 保存 JSON 结果文件 ──────────────────────────────────────────
    save_results_json(url, args, ratios, center_offsets,
                      avg_counter, avg_time, output_dir="./results")

    # ── 绘制箱线图 ─────────────────────────────────────────────────
    model_tag = os.path.splitext(os.path.basename(url))[0]
    plot_boxplots(ratios, center_offsets,
                  output_dir="./results", tag=model_tag)

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

    