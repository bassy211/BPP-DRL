from .space import Space
import numpy as np
import copy
import gym
from .cutCreator import CuttingBoxCreator
from .mdCreator  import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator, TrajectoryBoxCreator

class PackingGame(gym.Env):
    def __init__(self, box_creator=None, container_size = (20, 20, 20),
                 box_set = None, data_name = None, test = False,
                 data_type = 'cut1', enable_rotation=False, target_total=60, use_pusnet=False,
                 reward_alpha=1.0, reward_beta=1.0, reward_sigma=0.8, reward_tau=0.8, **kwags):
        self.box_creator = box_creator
        self.bin_size = container_size
        # 物理容器尺寸
        self.physical_width = container_size[0]
        self.physical_length = container_size[1]
        # 正方形网格尺寸：取长宽最大值，形成 pallet_size×pallet_size 的网格
        self.pallet_size = max(container_size[0], container_size[1])
        # 动作空间面积 = 正方形网格的单元格数
        self.area = int(self.pallet_size * self.pallet_size)
        # 创建空间：内部网格为 pallet_size×pallet_size，物理边界为实际容器尺寸
        self.space = Space(self.pallet_size, self.pallet_size, container_size[2],
                          physical_width=self.physical_width, physical_length=self.physical_length)
        self.can_rotate = enable_rotation
        self.use_pusnet = use_pusnet
        self.buffer = []
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_sigma = reward_sigma
        self.reward_tau = reward_tau
        self.max_episode_steps = target_total * 4
        self.current_step = 0

        if not test and box_creator is None:
            assert box_set is not None or data_name is not None
            if data_type == 'rs':
                print('using random data')
                self.box_creator = RandomBoxCreator(box_set)
            elif data_type == 'cut1':
                low = list(box_set[0])
                up = list(box_set[-1])
                low.extend(up)
                print(low)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            elif data_type == 'cut2':
                print('using md data')
                self.box_creator = MDlayerBoxCreator(container_size, [box_set[0][0], box_set[-1][0]])
            elif data_type == 'trajectory':
                print('using trajectory data from processed_train.pt')
                self.box_creator = TrajectoryBoxCreator(data_name, target_total=target_total)
            assert isinstance(self.box_creator, BoxCreator)

        if test:
            self.box_creator = LoadBoxCreator(data_name, target_total=target_total)

        self.act_len = self.area * 2 if self.use_pusnet else self.area * (1 + self.can_rotate)
        if self.use_pusnet:
            self.obs_len = self.area * self.bin_size[2] + self.area * 3
        else:
            self.obs_len = self.area * (1+3)
        self.action_space = gym.spaces.Discrete(self.act_len)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height, shape=(self.obs_len,))

    def clone_for_search(self):
        # Use a lightweight clone to avoid deepcopying huge dataset objects.
        new_env = copy.copy(self)
        new_env.space = self.space.clone() if hasattr(self.space, 'clone') else copy.deepcopy(self.space)
        new_env.buffer = copy.deepcopy(self.buffer)

        src_creator = self.box_creator
        new_creator = copy.copy(src_creator)
        for key, value in src_creator.__dict__.items():
            if key == 'data':
                # Dataset content is treated as read-only and shared across clones.
                setattr(new_creator, key, value)
                continue
            try:
                setattr(new_creator, key, copy.deepcopy(value))
            except Exception:
                setattr(new_creator, key, value)
        new_env.box_creator = new_creator
        return new_env
        

    def get_box_ratio(self):
        coming_box = self.next_box
        physical_volume = float(self.physical_width * self.physical_length * self.bin_size[2])
        return (coming_box[0] * coming_box[1] * coming_box[2]) / physical_volume


    def get_box_plain(self):
        x_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[0]
        y_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[1]
        z_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[2]
        return (x_plain, y_plain, z_plain)

    def reset(self):
        self.box_creator.reset()
        self.space = Space(self.pallet_size, self.pallet_size, self.bin_size[2],
                          physical_width=self.physical_width, physical_length=self.physical_length)
        self.buffer = []
        self.current_step = 0
        self.box_creator.generate_box_size()
        return self.cur_observation

    @property
    def cur_observation(self):
        size = self.get_box_plain()
        if self.use_pusnet:
            voxel = self.space.get_weighted_voxel_grid()
            return np.concatenate((voxel.reshape(-1), np.reshape(np.stack(size), newshape=(-1,))))
        hmap = self.space.plain
        return np.reshape(np.stack((hmap,  *size)), newshape=(-1,))

    @property
    def next_box(self):
        if self.use_pusnet and len(self.buffer) > 0:
            return self.buffer[-1]
        return self.box_creator.preview(1)[0]

    def _consume_current_item(self):
        if self.use_pusnet and len(self.buffer) > 0:
            self.buffer.pop()
            return
        self.box_creator.drop_box()
        self.box_creator.generate_box_size()

    def _compute_reward(self, wasted_before, wasted_after):
        total_vol = float(self.physical_width * self.physical_length * self.bin_size[2])
        cur = self.next_box
        r_v = float(cur[0] * cur[1] * cur[2]) / total_vol
        packed_vol = sum([b.x * b.y * b.z for b in self.space.boxes])
        r_sv = float(packed_vol) / total_vol
        r_w = float(wasted_after) / total_vol
        r_cw = float(wasted_after - wasted_before) / total_vol
        reward = self.reward_alpha * r_v + self.reward_beta * r_sv - self.reward_sigma * r_w - self.reward_tau * r_cw
        return reward

    def get_possible_position(self, plain=None):
        x = self.next_box[0]
        y = self.next_box[1]
        z = self.next_box[2]

        if plain is None:
            plain = self.space.plain

        width = self.pallet_size
        length = self.pallet_size

        action_mask = np.zeros(shape=(width, length), dtype=np.int32)
        
        for i in range(width-x+1):
            for j in range(length-y+1):
                if self.space.check_box(plain, x, y, i, j, z) >= 0:
                    action_mask[i, j] = 1

        # 硬掩码：将物理上不可放置的区域（如 12×10 容器中 y≥10 的部分）直接置0
        action_mask[:, self.physical_length:] = 0

        if action_mask.sum() == 0:
            action_mask[:, :] = 1
            # 但仍然保持硬掩码区域无效
            action_mask[:, self.physical_length:] = 0
        
        return action_mask

    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            idx = action[0]
        else:
            idx = action
        self.current_step += 1
        wasted_before = self.space.get_wasted_volume() if self.use_pusnet else 0

        if self.use_pusnet:
            is_unpack = idx >= self.area
            cell_idx = idx - self.area if is_unpack else idx
            succeeded = False
            unpack_check = None

            if not is_unpack:
                succeeded = self.space.drop_box(self.next_box, cell_idx, False)
                if succeeded:
                    self._consume_current_item()
            else:
                removed, unpack_check = self.space.unpack_box_at_constrained(cell_idx, self.next_box)
                if removed is not None:
                    self.buffer.append(removed)
                    succeeded = True

            wasted_after = self.space.get_wasted_volume()
            reward = self._compute_reward(wasted_before, wasted_after)
            if not succeeded:
                reward -= 0.3

            done = False
            if self.current_step >= self.max_episode_steps:
                done = True

            info = {
                'counter': len(self.space.boxes),
                'ratio': self.space.get_ratio(),
                'center_offset': self.space.get_center_offset(),
                'buffer_size': len(self.buffer),
                'succeeded': succeeded,
                'unpack_check': unpack_check,
            }
            return self.cur_observation, reward, done, info

        flag = False
        if idx >= self.area:
            assert self.can_rotate
            idx = idx - self.area
            flag = True
        succeeded = self.space.drop_box(self.next_box, idx, flag)

        if not succeeded:
            reward = 0.0
            done = True
            info = {'counter':len(self.space.boxes), 'ratio':self.space.get_ratio(), 'center_offset':self.space.get_center_offset(), 'mask':np.ones(shape=self.act_len)}
            return self.cur_observation, reward, done, info

        box_ratio = self.get_box_ratio()

        self.box_creator.drop_box()
        self.box_creator.generate_box_size()

        reward = box_ratio * 10
        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['ratio'] = self.space.get_ratio()
        info['center_offset'] = self.space.get_center_offset()
        return self.cur_observation, reward, done, info

