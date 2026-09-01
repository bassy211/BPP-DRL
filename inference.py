# python inference.py --load-name yang_bin1208.pt --data-name processed_test.pt --enable-rotation --target-total 60 --random-trajectory
# python inference.py --load-name yang_bin1210.pt --data-name processed_test.pt --enable-rotation --target-total 80 --random-trajectory
# python inference.py --data-name cut_1.pt --load-name yang.pt --enable-rotation --random-trajectory



import os
# PyTorch(MKL) 与 matplotlib 同时加载时会触发 OpenMP 运行时重复初始化
# （OMP Error #15），必须在任何第三方库导入之前设置
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from time import perf_counter
from acktr.model_loader import nnModel, normalize_state_dict
from acktr.reorder import ReorderTree
import gym
from gym.envs.registration import register
import copy
import torch
from acktr.arguments import get_args
import random
# 复用 visualize_packing.py 中的顶刊风格 matplotlib 3D 可视化
import Visualization as vp

# 模型配置
DEFAULT_MODEL_PATH = 'pretrained_models/best.pt'

def registration_envs():
    """注册环境到Gym"""
    try:
        register(
            id='Bpp-v0',
            entry_point='envs.bpp0:PackingGame',
        )
        print("环境注册成功")
    except gym.error.Error:
        print("环境已经注册")

def generate_real_time_box():
    depth = random.randint(2, 5)
    width = random.randint(2, 5)
    height = random.randint(2, 5)
    return [depth, width, height]

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = perf_counter()
    while True:
        box_list = env.box_creator.preview(preview_num)
        
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])

        box_counter += 1
        default_counter += int(default)
        if done:
            end = perf_counter()
            print(f'\n打包完成!')
            print(f'  总耗时: {end - start:.4f} 秒')
            print(f'  空间利用率: {info["ratio"]:.1%}')
            print(f'  成功放置: {info["counter"]} 个盒子')
            
            # 获取质心偏移量（稳定性指标）
            center_offset = info.get('center_offset', 0.0)
            center_offset_raw = info.get('center_offset_raw', 0.0)  # 原始绝对偏移量（调试参考）
            
            # 将盒子数据转换为 visualize_packing 需要的格式
            # (dx, dy, dz, lx, ly, lz, mass, density, rotation)
            boxes_for_vis = []
            for i, box in enumerate(env.space.boxes):
                rot = env.space.flags[i] if i < len(env.space.flags) else 0
                mass = getattr(box, 'mass', 0.0)
                density = getattr(box, 'density', 0.0)
                boxes_for_vis.append((box.x, box.y, box.z, box.lx, box.ly, box.lz, mass, density, rot))
            
            # 输出所有盒子的信息及其放置位置
            print(f"\n 最终放置结果 (共{len(env.space.boxes)}个盒子):")
            for i, box in enumerate(env.space.boxes):
                # 检查盒子是否有质量和密度属性
                if hasattr(box, 'mass') and hasattr(box, 'density'):
                    print(f"  盒子 {i+1}: 尺寸=({box.x},{box.y},{box.z}), 位置=({box.lx},{box.ly},{box.lz}), 质量={box.mass:.2f}, 密度={box.density:.2f}")
                else:
                    print(f"  盒子 {i+1}: 尺寸=({box.x},{box.y},{box.z}), 位置=({box.lx},{box.ly},{box.lz})")
            
            # 输出稳定性分析
            print(f"\n 稳定性分析:")
            print(f"  质心偏移量: {center_offset * 100:.2f}%  (原始绝对值: {center_offset_raw:.4f})")
            if center_offset < 0.05:
                stability_level = "优秀 ✓"
            elif center_offset < 0.10:
                stability_level = "良好"
            elif center_offset < 0.20:
                stability_level = "一般"
            else:
                stability_level = "较差 ✗"
            print(f"  稳定性等级: {stability_level}")
            
            # 尝试可视化（复用 visualize_packing.py 的顶刊风格 matplotlib 3D 可视化）
            try:
                container_size = tuple(env.bin_size)
                title = ('BPP Packing Result | container %dx%dx%d | ratio %.1f%% | %d boxes | center offset %.2f%%'
                         % (container_size[0], container_size[1], container_size[2],
                            info['ratio'] * 100, len(env.space.boxes), center_offset * 100))
                os.makedirs('results', exist_ok=True)
                from datetime import datetime
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                vp.visualize_packing(
                    container_size=container_size,
                    boxes=boxes_for_vis,
                    title=title,
                    out_path=os.path.join('results', 'packing_inference_%s.png' % ts),
                    labels=False,
                    show=False,
                    effective_container_size=env.effective_container_size,
                )
            except Exception as vis_error:
                print(f" 可视化失败: {vis_error}")
            
            return info['ratio'], info['counter'], end - start, default_counter / box_counter, center_offset, center_offset_raw
        
def check_model_compat(model_path, args):
    """校验模型所需的容器尺寸/旋转配置与当前命令行参数是否一致"""
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        sd = ckpt[0] if isinstance(ckpt, (list, tuple)) else ckpt
        sd = normalize_state_dict(sd)

        alen = sd['dist.linear.weight'].shape[0]      # 模型动作空间大小
        mask_in = sd['base.mask.3.weight'].shape[1]   # mask 输入通道 = 8 * pallet^2
        pallet = int(round((mask_in / 8) ** 0.5))
        area = pallet * pallet
        needs_rot = (alen == area * 2)

        cur_area = args.container_size[0] * args.container_size[1]
        if cur_area != area:
            print(f'  [警告] 模型要求容器面积 {area} ({pallet}x{pallet})，'
                  f'当前 --container-size {tuple(args.container_size)} 面积 {cur_area}。'
                  f'请使用 --container-size {pallet} {pallet} {args.container_size[2]}')
        if needs_rot and not args.enable_rotation:
            print('  [警告] 该模型启用了旋转，请加上 --enable-rotation')
        if not needs_rot and args.enable_rotation:
            print(f'  [警告] 该模型未启用旋转（动作空间 {alen}），而当前开启了 --enable-rotation'
                  f'（会把动作空间扩为 {cur_area * 2}），加载将失败。'
                  f'请去掉 --enable-rotation，或换用支持旋转的模型（如 --load-name cut1.pt）。')
        return area, pallet, needs_rot
    except Exception as e:
        print(f'  [提示] 无法校验模型配置（将按当前参数尝试加载）: {e}')
        return None, None, None

def inference(url, args, pruning_threshold=0.5):
    check_model_compat(url, args)
    nmodel = nnModel(url, args)
    
    # 使用预生成的数据集文件（默认 cut_1.pt，可用 --data-name 切换 cut_2.pt / rs.pt / processed_test.pt 等）
    data_name = os.path.join('./dataset/', args.data_name)
        
    env = gym.make(args.env_name,
                   box_set=args.box_size_set,
                   container_size=args.container_size,
                   enable_rotation=args.enable_rotation,
                   data_type=args.data_type,
                   data_name=data_name,
                   target_total=args.target_total,
                   test=True)

    # 环境本身不存储 effective_container_size（被构造器 **kwargs 吞掉），
    # 这里显式挂到 env 上，供 run_sequence 可视化时使用
    env.effective_container_size = args.effective_container_size

    print('  环境名称:', args.env_name)
    print('  模型路径:', url)
    print('  数据集文件:', data_name)
    # print('  剪枝阈值:', pruning_threshold)
    print('  预览盒子数量:', args.preview)
    print('  容器尺寸:', args.container_size)
    print('  启用旋转:', args.enable_rotation)
    c_bound = pruning_threshold
    env.reset()
    
    # 可选：随机选取一条轨迹进行推理
    # 默认测试模式按顺序取数据集第一条轨迹，导致每次推理配置/结果相同；
    # 加上 --random-trajectory 后每次随机抽取一条轨迹（不同运行结果不同）。
    if args.random_trajectory:
        traj_nums = getattr(env.box_creator, 'traj_nums', 1)
        traj_idx = random.randrange(traj_nums)
        print('  随机选取轨迹: %d / %d' % (traj_idx + 1, traj_nums))
        env.box_creator.reset(index=traj_idx)
        env.box_creator.generate_box_size()
    
    try:
        ratio, counter, time, depen_rate, center_offset, center_offset_raw = run_sequence(nmodel, env, args.preview, c_bound)

        print()
        print('----------------------------------------------')
        print('  推理结果统计:')
        print('  空间利用率: %.4f (%.1f%%)' % (ratio, ratio * 100))
        print('  成功放置盒子数: %d' % counter)
        print('  总耗时: %.4f 秒' % time)
        print('  平均每个盒子耗时: %.4f 秒' % (time / counter if counter > 0 else 0))
        print('----------------------------------------------')
        print('  稳定性指标:')
        print('  质心偏移量: %.2f%%  (原始绝对值: %.4f)' % (center_offset * 100, center_offset_raw))
        print('----------------------------------------------')
    except Exception as e:
        print(f" 出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5
    # 模型路径由 --load-dir / --load-name 控制（默认 ./pretrained_models/best.pt）
    model_path = os.path.join(args.load_dir, args.load_name)
    inference(model_path, args, pruning_threshold)