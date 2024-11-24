from time import perf_counter
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
from acktr.arguments import get_args
import random

def generate_real_time_box():
    depth = random.randint(2, 5)
    width = random.randint(2, 5)
    height = random.randint(2, 5)
    set = [depth, width, height]
    return set

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = perf_counter()
    while True:
        box_list = env.box_creator.preview(preview_num)
        print("\nBox list:", box_list)
        
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])
        
        # 在每次放置后输出最新放置的盒子信息
        if env.space.boxes:  # 确保有盒子被放置
            latest_box = env.space.boxes[-1]
            print(f"Just placed box: x={latest_box.x}, y={latest_box.y}, z={latest_box.z}, "
                  f"lx={latest_box.lx}, ly={latest_box.ly}, lz={latest_box.lz}")

        box_counter += 1
        default_counter += int(default)
        if done:
            end = perf_counter()
            print('\nFinal state:')
            print('Time cost:', end - start)
            print('Ratio:', info['ratio'])
            
            # 将盒子数据转换为可视化函数需要的格式
            boxes_for_vis = []
            for box in env.space.boxes:
                boxes_for_vis.append((box.x, box.y, box.z, box.lx, box.ly, box.lz))
            
            # 调用可视化函数
            import Visualization
            Visualization.visualize_boxes(container_size=(10, 10, 10), boxes=boxes_for_vis)
            
            return info['ratio'], info['counter'], end - start, default_counter / box_counter
        
def inference(url, args, pruning_threshold=0.5):
    nmodel = nnModel(url, args)
    box_set = []
    num_boxes = 1
    
    for _ in range(num_boxes):
        box = generate_real_time_box()
        box_set.append(tuple(box))
        
    env = gym.make(args.env_name,
                   box_set=box_set,
                   container_size=args.container_size,
                   enable_rotation=args.enable_rotation,
                   infer=True,data_type='rs')           

    print('Env name: ', args.env_name)
    print('Model url: ', url)
    print('pruning threshold: ', pruning_threshold)
    print('Known item number: ', args.preview)
    print('Generated box set: ', box_set)
    c_bound = pruning_threshold
    env.reset()
    #env.box_creator.preview(500)
    ratio, counter, time, depen_rate = run_sequence(nmodel, env, args.preview, c_bound)

    print()
    print('----------------------------------------------')
    print('space utilization: %.4f' % ratio)
    print('put item number: %.4f' % counter)
    print('sequence time: %.4f' % time)
    print('time per item: %.4f' % (time / counter))
    print('----------------------------------------------')

if __name__ == '__main__':
    args = get_args()
    pruning_threshold = 0.5
    inference('pretrained_models/default_cut_2.pt', args, pruning_threshold)
