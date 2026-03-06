from time import perf_counter
from acktr.model_loader import nnModel
import gym
from gym.envs.registration import register
import copy
import numpy as np
from acktr.arguments import get_args
import random
import Visualization

# 模型配置
DEFAULT_MODEL_PATH = 'pretrained_models/default_cut_1.pt'

def registration_envs():
    register(
        id='Bpp-v0',
        entry_point='envs.bpp0:PackingGame', 
    )

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
        value, poss = nmodel.evaluate(obs, use_mask=True)
        act = int(np.argmax(poss))
        default = 0
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
            
            # 将盒子数据转换为可视化函数需要的格式
            boxes_for_vis = []
            for box in env.space.boxes:
                boxes_for_vis.append((box.x, box.y, box.z, box.lx, box.ly, box.lz))
            
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
            
            # 计算质心位置
            com = env.space.calculate_center_of_mass()
            if com:
                com_x, com_y = com
                container_center_x = env.space.plain_size[0] / 2.0
                container_center_y = env.space.plain_size[1] / 2.0
                
                print(f"  容器几何中心: ({container_center_x:.2f}, {container_center_y:.2f})")
                print(f"  实际质心位置: ({com_x:.2f}, {com_y:.2f})")
                print(f"  质心偏移量: {center_offset:.4f}")
                print(f"  相对偏移率: {(center_offset / container_center_x * 100):.2f}%")
                
                # 计算质量分布统计
                total_mass = sum(box.mass for box in env.space.boxes)
                avg_mass = total_mass / len(env.space.boxes)
                mass_std = np.std([box.mass for box in env.space.boxes])
                print(f"\n 质量分布:")
                print(f"  总质量: {total_mass:.2f}")
                print(f"  平均质量: {avg_mass:.2f}")
                print(f"  质量标准差: {mass_std:.2f}")
            
            # 稳定性评级
            print(f"\n 稳定性评级:")
            if center_offset < 0.3:
                stability_level = "优秀 ✓"
                stability_desc = "质心位置非常接近几何中心，稳定性极佳"
            elif center_offset < 0.5:
                stability_level = "良好"
                stability_desc = "质心位置较为居中，稳定性良好"
            elif center_offset < 0.7:
                stability_level = "一般"
                stability_desc = "质心有一定偏移，建议优化布局"
            else:
                stability_level = "较差 ✗"
                stability_desc = "质心偏移较大，需要重新优化布局"
            print(f"  等级: {stability_level}")
            print(f"  说明: {stability_desc}")
            
            # 可选：打印完整的稳定性报告（取消注释以启用）
            # env.space.print_stability_report()
            
            # 尝试可视化
            try:
                Visualization.visualize_boxes_enhanced(
                    container_size=(10, 10, 10), 
                    boxes=boxes_for_vis, 
                    method='pyvista',
                    color_style='modern'
                )
            except Exception as vis_error:
                print(f" 可视化失败: {vis_error}")
            
            return info['ratio'], info['counter'], end - start, default_counter / box_counter, center_offset
        
def inference(url, args, pruning_threshold=0.5):
    nmodel = nnModel(url, args)
        
    env = gym.make(args.env_name,
                   box_set=args.box_size_set,
                   container_size=args.container_size,
                   enable_rotation=args.enable_rotation,
                   data_type=args.data_type, 
                   infer=True)           

    print('  环境名称:', args.env_name)
    print('  模型路径:', url)
    # print('  剪枝阈值:', pruning_threshold)
    print('  预览盒子数量:', args.preview)
    print('  容器尺寸:', args.container_size)
    print('  启用旋转:', args.enable_rotation)
    c_bound = pruning_threshold
    env.reset()
    
    try:
        ratio, counter, time, depen_rate, center_offset = run_sequence(nmodel, env, args.preview, c_bound)

        print()
        print('----------------------------------------------')
        print('  推理结果统计:')
        print('  空间利用率: %.4f (%.1f%%)' % (ratio, ratio * 100))
        print('  成功放置盒子数: %d' % counter)
        print('  总耗时: %.4f 秒' % time)
        print('  平均每个盒子耗时: %.4f 秒' % (time / counter if counter > 0 else 0))
        print('----------------------------------------------')
        print('  稳定性指标:')
        print('  质心偏移量: %.4f' % center_offset)
        
        # 计算稳定性得分 (0-100分)
        max_offset = np.sqrt(2) * env.space.plain_size[0] / 2.0  # 理论最大偏移
        stability_score = max(0, 100 * (1 - center_offset / max_offset))
        print('  稳定性得分: %.2f/100' % stability_score)
        
        if center_offset < 0.3:
            print('  稳定性评级: 优秀 ✓')
        elif center_offset < 0.5:
            print('  稳定性评级: 良好')
        elif center_offset < 0.7:
            print('  稳定性评级: 一般')
        else:
            print('  稳定性评级: 较差 ✗')
        print('----------------------------------------------')
    except Exception as e:
        print(f" 出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5
    # 使用配置的默认模型路径
    inference(DEFAULT_MODEL_PATH, args, pruning_threshold)
