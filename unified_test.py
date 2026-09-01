from time import perf_counter
from acktr.model_loader import nnModel
from acktr.heuristic_baselines import HeuristicModel
from acktr.reorder import ReorderTree
import gym
import copy
import numpy as np
from gym.envs.registration import register
from acktr.arguments import get_args
from analysis_plots import run_full_analysis

def run_sequence(nmodel, raw_env, preview_num, c_bound, reorder_times=100, reorder_pos_topk=1):
    env = raw_env.clone_for_search() if hasattr(raw_env, 'clone_for_search') else copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = perf_counter()
    while True:
        box_list = env.box_creator.preview(preview_num)
        # print(box_list)
        tree = ReorderTree(
            nmodel,
            box_list,
            env,
            p_bound=c_bound,
            times=reorder_times,
            pos_topk=reorder_pos_topk,
        )
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])
        if done:
            end = perf_counter()
            print('Time cost:', end-start)
            print('Ratio:', info['ratio'])
            # 计算质心偏移量（归一化比值 0~1，便于跨容器比较）
            center_offset = (env.space.get_relative_offset_ratio()
                             if hasattr(env.space, 'get_relative_offset_ratio')
                             else env.space.get_center_offset())
            com = env.space.calculate_center_of_mass()
            com_x, com_y = com if com else (0.0, 0.0)
            return info['ratio'], info['counter'], end-start, default_counter/box_counter, center_offset, com_x, com_y
        box_counter += 1
        default_counter += int(default)

def unified_test(url,  args, pruning_threshold = 0.5):
    if args.algorithm in ['random', 'first_fit', 'best_fit', 'corner_point', 'extreme_point', 'ems', 'macs', 'layer_building', 'f53']:
        nmodel = HeuristicModel(args.algorithm, args) if args.algorithm not in ['layer_building', 'f53'] else None
    else:
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
    counters = []
    times_list = []
    center_offsets = []  # 收集质心偏移量
    com_xs = []
    com_ys = []
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = getattr(args, 'reorder_p_bound', pruning_threshold)
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)

        if args.algorithm == 'layer_building':
            # Offline MILP: solve globally for all items at once
            from acktr.milp_heuristic import milp_two_phase_pack
            ratio, counter, elapsed, depen_rate, center_offset, com_x, com_y = \
                milp_two_phase_pack(env)
            time_val = elapsed
        elif args.algorithm == 'f53':
            # Online EMS heuristic (Ali et al., 2024)
            from acktr.f53_heuristic import f53_online_pack
            ratio, counter, elapsed, depen_rate, center_offset, com_x, com_y = \
                f53_online_pack(env)
            time_val = elapsed
        else:
            ratio, counter, time_val, depen_rate, center_offset, com_x, com_y = run_sequence(
                nmodel,
                env,
                args.preview,
                c_bound,
                reorder_times=args.reorder_times,
                reorder_pos_topk=args.reorder_pos_topk,
            )
        avg_ratio += ratio
        ratios.append(ratio)
        center_offsets.append(center_offset)
        com_xs.append(com_x)
        com_ys.append(com_y)
        avg_counter += counter
        counters.append(counter)
        avg_time += time_val
        times_list.append(time_val)
        avg_drate += depen_rate

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    
    # 计算各指标的标准差
    ratios = np.array(ratios)
    counters = np.array(counters)
    times_list = np.array(times_list)
    
    std_ratio = np.std(ratios)
    std_counter = np.std(counters)
    std_time = np.std(times_list)
    
    avg_ratio_val = avg_ratio / times
    avg_counter_val = avg_counter / times
    avg_time_val = avg_time / times
    avg_time_per_item = avg_time / avg_counter
    std_time_per_item = std_time / avg_counter_val  # 近似计算
    
    print('average space utilization: %.4f ± %.4f' % (avg_ratio_val, std_ratio))
    print('average put item number: %.4f ± %.4f' % (avg_counter_val, std_counter))
    print('average sequence time: %.4f ± %.4f' % (avg_time_val, std_time))
    print('average time per item: %.4f ± %.4f' % (avg_time_per_item, std_time_per_item))
    print('----------------------------------------------')
    
    # 计算并输出质心偏移量统计
    center_offsets = np.array(center_offsets)
    avg_offset = np.mean(center_offsets)
    var_offset = np.var(center_offsets)
    std_offset = np.std(center_offsets)
    min_offset = np.min(center_offsets)
    max_offset = np.max(center_offsets)
    
    print('---------- Center of Mass Statistics ----------')
    print('Average center offset: %.4f ± %.4f' % (avg_offset, std_offset))
    print('Variance of center offset: %.4f' % var_offset)
    print('Min center offset: %.4f' % min_offset)
    print('Max center offset: %.4f' % max_offset)
    print('----------------------------------------------')
    
    # # 导出 JSON 并生成箱线图
    # print('\nExporting results and generating plots...')
    # run_full_analysis(url, data_url, args, ratios, center_offsets,
    #                   avg_counter, avg_time, times, com_xs, com_ys)

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

    