# -*- coding: utf-8 -*-
"""
对预训练模型执行一次完整的装箱推理，并把装满盒子的容器可视化。

用法示例（best.pt 是 10x10 底面 + 启用旋转的模型，脚本默认即此配置）：

    python visualize_packing.py
    python visualize_packing.py --model pretrained_models/best.pt --seed 42
    python visualize_packing.py --model pretrained_models/best.pt --item-seq processed --show
    python visualize_packing.py --model pretrained_models/best.pt --pyvista   # 交互式 3D 窗口

输出：
    results/packing_<模型名>_<时间戳>.png     matplotlib 3D 可视化（单一等轴侧视图，柔和粉彩色系，每箱一色，无坐标轴，容器正视图框线位于箱子前方）
    results/packing_<模型名>_<时间戳>.json    放置结果明细（盒子尺寸/位置/质量/旋转）
"""

import os
import sys
import json
import copy
import random
import argparse
from time import perf_counter

# PyTorch(MKL) 与 matplotlib 同时加载时会触发 OpenMP 运行时重复初始化
# （OMP Error #15），必须在任何第三方库导入之前设置
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# 保证从任意工作目录运行时都能找到本仓库的包（acktr / envs / baselines stub）
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import gym
from gym.envs.registration import register


# --------------------------------------------------------------------------
# 1. 命令行参数
# --------------------------------------------------------------------------
def parse_my_args():
    p = argparse.ArgumentParser(description='Run one inference with a trained BPP model and visualize the packing')
    p.add_argument('--model', default='pretrained_models/best.pt',
                   help='path to the trained model checkpoint (default: pretrained_models/best.pt)')
    p.add_argument('--container-size', nargs=3, type=int, default=[10, 10, 10],
                   help='container size (width, length, height), default 10 10 10 (matches best.pt)')
    p.add_argument('--no-rotation', action='store_true',
                   help='disable Z-axis rotation (default: rotation enabled, matches best.pt)')
    p.add_argument('--item-seq', default='cut1', choices=['cut1', 'cut2', 'rs', 'processed'],
                   help='box sequence generator: cut1 (random cutting, default) | cut2 | rs | processed (dataset)')
    p.add_argument('--train-data-name', default='processed_train.pt',
                   help='dataset file for --item-seq processed (under ./dataset/)')
    p.add_argument('--target-total-boxes', type=int, default=80,
                   help='target boxes per trajectory for --item-seq processed (default: 80)')
    p.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    p.add_argument('--max-steps', type=int, default=500, help='safety cap on episode steps (default: 500)')
    p.add_argument('--out-dir', default='results', help='output directory (default: results)')
    p.add_argument('--labels', action='store_true', help='annotate every box with its placement index')
    p.add_argument('--show', action='store_true', help='open an interactive matplotlib window')
    p.add_argument('--pyvista', action='store_true', help='also open an interactive pyvista 3D window')
    return p.parse_args()


def build_acktr_args(my):
    """构造 acktr 的 args（复用 get_args 的完整后处理：硬掩码、box_size_set 等）。"""
    argv = ['visualize_packing.py']
    argv += ['--container_size', str(my.container_size[0]), str(my.container_size[1]), str(my.container_size[2])]
    if not my.no_rotation:
        argv += ['--enable-rotation']
    if torch.cuda.is_available():
        argv += ['--use-cuda']
    argv += ['--item-seq', my.item_seq]
    argv += ['--train_data_name', my.train_data_name]
    argv += ['--target_total_boxes', str(my.target_total_boxes)]
    sys.argv = argv

    from acktr.arguments import get_args
    args = get_args()

    # 参数一致性自检：观测维度 / 动作维度必须与 checkpoint 匹配（在加载时验证）
    args.checkpoint_url = my.model
    return args


# --------------------------------------------------------------------------
# 2. 环境注册 + 推理
# --------------------------------------------------------------------------
def registration_envs():
    """注册环境到Gym（已注册时忽略）"""
    try:
        register(id='Bpp-v0', entry_point='envs.bpp0:PackingGame')
    except gym.error.Error:
        print('  [info] Bpp-v0 已注册')


def make_env(args):
    data_name = None
    if args.data_type == 'processed':
        data_name = os.path.join('./dataset/', args.train_data_name)

    env = gym.make(args.env_name,
                   box_set=args.box_size_set,
                   container_size=args.container_size,
                   enable_rotation=args.enable_rotation,
                   data_type=args.data_type,
                   data_name=data_name,
                   target_total_boxes=args.target_total_boxes,
                   effective_container_size=args.effective_container_size,
                   infer=True)
    env.reset()
    return env


def run_episode(nmodel, env, max_steps=500):
    """贪心推理：每步选择模型概率最高的合法动作（与 unified_test.py 一致）。"""
    obs = env.cur_observation
    t0 = perf_counter()
    for step in range(max_steps):
        val, poss = nmodel.evaluate(obs, use_mask=True)
        act = int(poss.argmax())
        obs, _, done, info = env.step([act])
        if done:
            return info, perf_counter() - t0
    raise RuntimeError('episode did not terminate within %d steps' % max_steps)


# --------------------------------------------------------------------------
# 3. 可视化（matplotlib 3D，柔和粉彩色系）
# --------------------------------------------------------------------------
# 柔和粉彩色系（参考对比图中 EMSs / Zhao et al. / Ours 的柔和平滑配色）
PASTEL_PALETTE = [
    '#C9B8D8', '#B8CDB2', '#D9BFC6', '#E6DCC0', '#B9CFE0',
    '#BFDCC6', '#A9C9C9', '#D6C3A5', '#C5AFC2', '#BCC7D1',
    '#E8C9B8', '#E4D9A5', '#A99BC0', '#B5BC93', '#E3C2D0',
    '#B8DFD8', '#C4BCB0', '#B4B8DE', '#E0B8AC', '#CBD6B5',
    '#D9D0E0', '#A8C6B8', '#E2D3C0', '#C0C8E0', '#D8C8E8',
    '#B3C8C2', '#E8D8C8', '#C8B8A8',
]


def box_colors(n):
    """为每个箱子分配一个颜色；超出调色板长度时循环复用调色板颜色。"""
    palette = PASTEL_PALETTE
    return [palette[i % len(palette)] for i in range(n)]


# ---- 光照模型：模拟"右前上方"主光源的 Lambert 漫反射，让箱子的每个面明暗有别 ----
_LIGHT_DIR = np.array([0.45, -0.30, 0.84])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_AMBIENT = 0.42            # 环境光比例（避免背光面全黑）
_EDGE_FACTOR = 0.5         # 描边亮度（同色系深色，比纯黑更精致）


def _rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)],
                    dtype=float) / 255.0


def _shaded_cuboid(dx, dy, dz, lx, ly, lz, hex_color):
    """返回 (faces, face_colors, edge_colors)：6 个面按各自法线打光着色。

    同一箱子的顶面最亮、侧面次之、底面最暗，产生真实的体积感。
    """
    c = np.array([
        [lx, ly, lz], [lx + dx, ly, lz], [lx + dx, ly + dy, lz], [lx, ly + dy, lz],
        [lx, ly, lz + dz], [lx + dx, ly, lz + dz], [lx + dx, ly + dy, lz + dz], [lx, ly + dy, lz + dz],
    ], dtype=float)
    face_idx = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
    ]
    center = np.array([lx + dx / 2, ly + dy / 2, lz + dz / 2])
    base = _rgb(hex_color)

    faces, face_colors, edge_colors = [], [], []
    for idx in face_idx:
        v = [c[i] for i in idx]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        n = n / (np.linalg.norm(n) + 1e-12)
        face_center = np.mean(v, axis=0)
        if np.dot(n, face_center - center) < 0:  # 法线统一指向外侧
            n = -n
        diff = max(0.0, float(np.dot(n, _LIGHT_DIR)))
        intensity = _AMBIENT + (1.0 - _AMBIENT) * diff
        faces.append(v)
        face_colors.append(np.clip(base * intensity, 0.0, 1.0))
        edge_colors.append(np.clip(base * intensity * _EDGE_FACTOR, 0.0, 1.0))
    return faces, face_colors, edge_colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Arrow3D(FancyArrowPatch):
    """3D arrow (line + arrowhead) for coordinate axes."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def visualize_packing(container_size, boxes, title, out_path, labels=False, show=False,
                      effective_container_size=None):
    """boxes: list of (dx, dy, dz, lx, ly, lz, mass, density, rotation)"""
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt

    W, L, H = container_size
    if (effective_container_size is not None
            and tuple(int(v) for v in effective_container_size[:3]) != tuple(container_size)):
        fw, fl, fh = (int(v) for v in effective_container_size[:3])
    else:
        fw, fl, fh = W, L, H
    
    colors = box_colors(len(boxes))

    # 样式参数
    SIDE_ALPHA = 0.15
    TOP_ALPHA  = 0.20
    BOT_FILL   = 0.12
    GRID_LW    = 0.6
    EDGE_LW    = 1.4

    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(9.5, 8.5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置正交投影，取消透视效果
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=-58)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    # 1. 绘制容器底部网格与背景侧面 (zorder较低)
    xg = np.linspace(0, fw, int(fw)+1)
    yg = np.linspace(0, fl, int(fl)+1)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = np.zeros_like(Xg)
    
    # 底面填充与网格
    ax.plot_surface(Xg, Yg, Zg, color="#c9d3e0", alpha=BOT_FILL,
                    edgecolor="none", shade=False, zorder=1)
    ax.plot_wireframe(Xg, Yg, Zg, color="#2f3b4f", lw=GRID_LW, alpha=0.95, zorder=2)

    def face(X, Y, Z, alpha, color="#a7c0dd", zo=3):
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                        edgecolor="none", shade=False, zorder=zo)

    xx, yy, zz = np.linspace(0, fw, 2), np.linspace(0, fl, 2), np.linspace(0, fh, 2)
    Xf, Zf = np.meshgrid(xx, zz)
    Yf = np.zeros_like(Xf)
    Yf2, Zf2 = np.meshgrid(yy, zz)
    Xf2 = np.zeros_like(Yf2)
    
    # 绘制背面与左面
    face(Xf, Yf + fl, Zf, SIDE_ALPHA, zo=1)  # Back
    face(Xf2, Yf2, Zf2, SIDE_ALPHA, zo=1)    # Left

    # 2. 绘制装箱物体 (Poly3DCollection, zorder居中)
    all_faces, all_face_colors, all_edge_colors = [], [], []
    for i, (dx, dy, dz, lx, ly, lz, mass, density, rot) in enumerate(boxes):
        faces, face_colors, edge_colors = _shaded_cuboid(dx, dy, dz, lx, ly, lz, colors[i])
        all_faces += faces
        all_face_colors += face_colors
        all_edge_colors += edge_colors
        if labels:
            ax.text(lx + dx / 2, ly + dy / 2, lz + dz / 2, str(i + 1),
                    color='white', fontsize=7, ha='center', va='center', zorder=10)
            
    pc = Poly3DCollection(all_faces, facecolors=all_face_colors,
                          edgecolors=all_edge_colors, linewidths=0.6, alpha=1.0)
    pc.set_zorder(5)
    ax.add_collection3d(pc)

    # 3. 绘制容器前面板、顶面与所有实线边框 (zorder较高)
    face(Xf, Yf, Zf, SIDE_ALPHA, zo=6)       # Front
    face(Xf2 + fw, Yf2, Zf2, SIDE_ALPHA, zo=6) # Right
    Xt, Yt = np.meshgrid(xx, yy)
    face(Xt, Yt, np.full_like(Xt, fh), TOP_ALPHA, zo=6) # Top

    # 容器顶点与外边框
    # 去掉了与原本坐标系原点(0, fl, 0)相连的 3 条背面棱线：
    # 即 (2,3) 后下边缘, (3,0) 左下边缘, (3,7) 左后垂直边缘
    c = np.array([[0, 0, 0], [fw, 0, 0], [fw, fl, 0], [0, fl, 0],
                  [0, 0, fh], [fw, 0, fh], [fw, fl, fh], [0, fl, fh]])
    edges = [(0, 1), (1, 2), 
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6)]
             
    for i, j in edges:
        ax.plot([c[i, 0], c[j, 0]], [c[i, 1], c[j, 1]], [c[i, 2], c[j, 2]],
                color="#101826", lw=EDGE_LW, solid_capstyle="round", zorder=8)

    # 取消内置坐标系显示并设置比例
    ax.set_xlim(-2.6, fw + 4.2)
    ax.set_ylim(-4.8, fl + 1.2)
    ax.set_zlim(-1.0, fh + 4.0)
    ax.set_box_aspect((fw, fl, fh))
    ax.axis("off")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    print('  [OK] figure saved to: %s' % out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_pyvista(container_size, boxes):
    """可选的交互式 pyvista 窗口（需要图形环境，失败时给出提示）。"""
    import pyvista as pv
    W, L, H = container_size
    colors = box_colors(len(boxes))

    plotter = pv.Plotter(title='BPP Packing Result')
    plotter.add_mesh(pv.Box(bounds=(0, W, 0, L, 0, H)),
                     style='wireframe', color='black', line_width=3)
    for i, (dx, dy, dz, lx, ly, lz, mass, density, rot) in enumerate(boxes):
        plotter.add_mesh(pv.Box(bounds=(lx, lx + dx, ly, ly + dy, lz, lz + dz)),
                         color=colors[i], opacity=0.98, show_edges=True,
                         edge_color='black', line_width=1,
                         specular=0.4, specular_power=18, ambient=0.2)
    plotter.show_grid()
    plotter.show()
    plotter.close()


# --------------------------------------------------------------------------
# 4. 主流程
# --------------------------------------------------------------------------
def main():
    my = parse_my_args()
    random.seed(my.seed)
    np.random.seed(my.seed)
    torch.manual_seed(my.seed)

    registration_envs()
    args = build_acktr_args(my)
    print('  model :', my.model)
    print('  container :', tuple(args.container_size),
          ' rotation :', args.enable_rotation,
          ' item-seq :', args.item_seq,
          ' seed :', my.seed)

    # ---- 加载模型 ----
    from acktr.model_loader import nnModel
    nmodel = nnModel(my.model, args)

    # ---- 环境 + 推理 ----
    env = make_env(args)
    info, elapsed = run_episode(nmodel, env, max_steps=my.max_steps)

    ratio = info.get('ratio', 0.0)
    counter = info.get('counter', len(env.space.boxes))
    center_offset = info.get('center_offset', 0.0)
    center_offset_raw = info.get('center_offset_raw', 0.0)
    com_x = info.get('com_x', 0.0)
    com_y = info.get('com_y', 0.0)

    print()
    print('  ======== inference result ========')
    print('  space utilization : %.4f (%.1f%%)' % (ratio, ratio * 100))
    print('  boxes placed      : %d' % counter)
    print('  episode time      : %.3f s' % elapsed)
    print('  center offset     : %.2f%% (raw %.4f)' % (center_offset * 100, center_offset_raw))
    print('  center of mass    : (%.2f, %.2f)' % (com_x, com_y))
    print('  ===================================')

    # ---- 收集放置结果 ----
    boxes = []
    for i, box in enumerate(env.space.boxes):
        rot = env.space.flags[i] if i < len(env.space.flags) else 0
        boxes.append((box.x, box.y, box.z, box.lx, box.ly, box.lz,
                      box.mass, box.density, rot))
        print('  box %3d: size=(%d,%d,%d) pos=(%d,%d,%d) mass=%.2f density=%.2f rot=%d'
              % (i + 1, box.x, box.y, box.z, box.lx, box.ly, box.lz,
                 box.mass, box.density, rot))

    # ---- 保存结果 JSON ----
    model_tag = os.path.splitext(os.path.basename(my.model))[0]
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(my.out_dir, exist_ok=True)
    json_path = os.path.join(my.out_dir, 'packing_%s_%s.json' % (model_tag, ts))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': my.model,
            'container_size': [int(v) for v in args.container_size],
            'enable_rotation': bool(args.enable_rotation),
            'item_seq': args.item_seq,
            'seed': my.seed,
            'ratio': float(ratio),
            'boxes_placed': int(counter),
            'episode_time_s': float(elapsed),
            'center_offset': float(center_offset),
            'center_offset_raw': float(center_offset_raw),
            'center_of_mass': [float(com_x), float(com_y)],
            'boxes': [
                {'size': [int(b[0]), int(b[1]), int(b[2])],
                 'position': [int(b[3]), int(b[4]), int(b[5])],
                 'mass': float(b[6]), 'density': float(b[7]), 'rotation': int(b[8])}
                for b in boxes
            ],
        }, f, ensure_ascii=False, indent=2)
    print('  [OK] result json saved to: %s' % json_path)

    # ---- 可视化 ----
    title = ('%s | container %dx%dx%d | ratio %.1f%% | %d boxes | center offset %.2f%%'
             % (model_tag, args.container_size[0], args.container_size[1],
                args.container_size[2], ratio * 100, counter, center_offset * 100))
    png_path = os.path.join(my.out_dir, 'packing_%s_%s.png' % (model_tag, ts))
    visualize_packing(args.container_size, boxes, title, png_path,
                      labels=my.labels, show=my.show,
                      effective_container_size=args.effective_container_size)

    if my.pyvista:
        try:
            visualize_pyvista(args.container_size, boxes)
        except Exception as e:
            print('  [warn] pyvista visualization failed: %s' % e)


if __name__ == '__main__':
    main()
