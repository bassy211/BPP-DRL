import numpy as np
from functools import reduce
import copy, time


class Box(object):
    def __init__(self, x, y, z, lx, ly, lz, density=None):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz
        # 如果没有指定密度，从[0.5, 1.0]均匀采样
        if density is None:
            self.density = np.random.uniform(0.5, 1.0)
        else:
            self.density = density
        # 计算质量：密度 × 体积
        self.mass = self.density * self.x * self.y * self.z

    def get_center_xy(self):
        """获取盒子在XY平面的几何中心（质心）"""
        center_x = self.lx + self.x / 2.0
        center_y = self.ly + self.y / 2.0
        return center_x, center_y

    def standardize(self):
        return tuple([self.x, self.y, self.z, self.lx, self.ly, self.lz])

    def clone(self):
        return Box(
            int(self.x), int(self.y), int(self.z),
            int(self.lx), int(self.ly), int(self.lz),
            density=float(self.density)
        )


class Space(object):
    def __init__(self, width=10, length=10, height=10, physical_width=None, physical_length=None):
        # plain_size: 内部网格表示的尺寸（正方形，如12×12）
        self.plain_size = np.array([width, length, height])
        # 物理容器的实际宽/长（如12×10），用于check_box边界检查和体积计算
        self.physical_width = physical_width if physical_width is not None else width
        self.physical_length = physical_length if physical_length is not None else length
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.occupancy = np.zeros(shape=(width, length, height), dtype=np.int8)
        self.boxes = []
        self.flags = [] # record rotation information
        self.height = height
        self.last_wasted_volume = 0

    def print_height_graph(self):
        print(self.plain)

    def clone(self):
        new_space = Space(int(self.plain_size[0]), int(self.plain_size[1]), int(self.plain_size[2]),
                         physical_width=int(self.physical_width), physical_length=int(self.physical_length))
        new_space.plain = np.array(self.plain, copy=True)
        new_space.occupancy = np.array(self.occupancy, copy=True)
        new_space.boxes = [box.clone() for box in self.boxes]
        new_space.flags = [bool(f) for f in self.flags]
        new_space.height = int(self.height)
        new_space.last_wasted_volume = int(self.last_wasted_volume)
        return new_space

    def get_height_graph(self):
        plain = np.zeros(shape=self.plain_size[:2], dtype=np.int32)
        for box in self.boxes:
            plain = self.update_height_graph(plain, box)
        return plain

    @staticmethod
    def update_height_graph(plain, box):
        plain = copy.deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.lz + box.z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    def get_plain(self):
        return copy.deepcopy(self.plain)

    def get_action_space(self):
        return self.plain_size[0] * self.plain_size[1]

    def rebuild_state(self):
        width, length, height = self.plain_size
        self.occupancy = np.zeros((width, length, height), dtype=np.int8)
        self.plain = np.zeros((width, length), dtype=np.int32)
        for box in self.boxes:
            x0, x1 = box.lx, box.lx + box.x
            y0, y1 = box.ly, box.ly + box.y
            z0, z1 = box.lz, box.lz + box.z
            self.occupancy[x0:x1, y0:y1, z0:z1] = 1
            self.plain = self.update_height_graph(self.plain, box)

    def get_top_box_at(self, lx, ly):
        candidates = []
        for idx, box in enumerate(self.boxes):
            if box.lx <= lx < box.lx + box.x and box.ly <= ly < box.ly + box.y:
                candidates.append((idx, box))
        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[1].lz + item[1].z, reverse=True)
        top_idx, top_box = candidates[0]

        top_surface = top_box.lz + top_box.z
        for idx, other in enumerate(self.boxes):
            if idx == top_idx:
                continue
            overlap_x = not (other.lx + other.x <= top_box.lx or top_box.lx + top_box.x <= other.lx)
            overlap_y = not (other.ly + other.y <= top_box.ly or top_box.ly + top_box.y <= other.ly)
            if overlap_x and overlap_y and other.lz >= top_surface:
                return None, None
        return top_idx, top_box

    def unpack_box_at(self, idx):
        lx, ly = self.idx_to_position(idx)
        top_idx, top_box = self.get_top_box_at(lx, ly)
        if top_box is None:
            return None
        removed = self.boxes.pop(top_idx)
        self.flags.pop(top_idx)
        self.rebuild_state()
        self.last_wasted_volume = self.get_wasted_volume()
        return (removed.x, removed.y, removed.z)

    def _simulate_wasted_after_removal(self, top_idx):
        if top_idx < 0 or top_idx >= len(self.boxes):
            return None
        removed_box = self.boxes.pop(top_idx)
        removed_flag = self.flags.pop(top_idx)
        self.rebuild_state()
        wasted_after = self.get_wasted_volume()

        self.boxes.insert(top_idx, removed_box)
        self.flags.insert(top_idx, removed_flag)
        self.rebuild_state()
        return wasted_after

    def evaluate_unpack_candidate(self, idx, current_item):
        lx, ly = self.idx_to_position(idx)
        top_idx, top_box = self.get_top_box_at(lx, ly)
        if top_box is None:
            return {
                'valid': False,
                'reason': 'no_top_item',
                'top_idx': None,
                'removed_item': None,
                'wasted_improved': False,
            }

        wasted_before = self.get_wasted_volume()
        wasted_after = self._simulate_wasted_after_removal(top_idx)
        if wasted_after is None:
            return {
                'valid': False,
                'reason': 'invalid_simulation',
                'top_idx': None,
                'removed_item': None,
                'wasted_improved': False,
            }

        removed_vol = int(top_box.x * top_box.y * top_box.z)
        cur_vol = int(current_item[0] * current_item[1] * current_item[2])
        wasted_improved = (wasted_after < wasted_before)

        # Rule 1: only top-layer items can be unpacked (enforced by get_top_box_at)
        # Rule 2: avoid unpacking larger-than-current item unless it improves wasted space
        size_allowed = removed_vol <= cur_vol
        valid = size_allowed or wasted_improved

        return {
            'valid': bool(valid),
            'reason': 'ok' if valid else 'size_constraint',
            'top_idx': int(top_idx),
            'removed_item': (int(top_box.x), int(top_box.y), int(top_box.z)),
            'wasted_improved': bool(wasted_improved),
        }

    def unpack_box_at_constrained(self, idx, current_item):
        check = self.evaluate_unpack_candidate(idx, current_item)
        if not check['valid']:
            return None, check
        top_idx = check['top_idx']
        removed = self.boxes.pop(top_idx)
        self.flags.pop(top_idx)
        self.rebuild_state()
        self.last_wasted_volume = self.get_wasted_volume()
        return (removed.x, removed.y, removed.z), check

    def get_unpack_mask(self):
        width, length, _ = self.plain_size
        mask = np.zeros((width, length), dtype=np.int32)
        for i in range(width):
            for j in range(length):
                top_idx, top_box = self.get_top_box_at(i, j)
                if top_box is not None:
                    mask[i, j] = 1
        return mask

    def get_wasted_volume(self):
        width, length, height = self.plain_size
        wasted = 0
        min_free_height = 2
        for i in range(width):
            for j in range(length):
                col = self.occupancy[i, j]
                free_z = np.where(col == 0)[0]
                if free_z.size == 0:
                    continue
                top_occ = np.where(col == 1)[0]
                top_h = -1 if top_occ.size == 0 else int(top_occ[-1])
                for z in free_z:
                    if z <= top_h:
                        wasted += 1
                        continue
                    if height - z < min_free_height:
                        wasted += 1
        return wasted

    def get_weighted_voxel_grid(self):
        width, length, height = self.plain_size
        weighted = np.ones((width, length, height), dtype=np.int32)
        weighted[self.occupancy == 1] = 2

        min_free_height = 2
        for i in range(width):
            for j in range(length):
                col = self.occupancy[i, j]
                occ_idx = np.where(col == 1)[0]
                top_h = -1 if occ_idx.size == 0 else int(occ_idx[-1])
                for z in range(height):
                    if weighted[i, j, z] == 2:
                        continue
                    if z <= top_h:
                        weighted[i, j, z] = 0
                    elif height - z < min_free_height:
                        weighted[i, j, z] = 0
        return weighted

    def calculate_center_of_mass(self):
        """
        计算容器整体质心（忽略Z轴，只考虑XY平面）
        使用杠杆原理：质心 = Σ(质量i × 中心i) / Σ(质量i)
        返回: (center_x, center_y) 质心坐标
        """
        if len(self.boxes) == 0:
            return None
        
        weighted_x = 0.0
        weighted_y = 0.0
        total_mass = 0.0
        
        for box in self.boxes:
            box_center_x, box_center_y = box.get_center_xy()
            weighted_x += box.mass * box_center_x
            weighted_y += box.mass * box_center_y
            total_mass += box.mass
        
        center_x = weighted_x / total_mass
        center_y = weighted_y / total_mass
        
        return center_x, center_y
    
    def get_center_offset(self):
        """
        计算容器质心点相对于几何中心的偏移量
        返回: offset (欧氏距离)
        """
        com = self.calculate_center_of_mass()
        if com is None:
            return 0.0
        
        com_x, com_y = com
        # 容器几何中心（使用物理容器尺寸）
        container_center_x = self.physical_width / 2.0
        container_center_y = self.physical_length / 2.0
        
        # 计算欧氏距离
        offset = np.sqrt((com_x - container_center_x)**2 + (com_y - container_center_y)**2)
        
        return offset
    
    def get_stability_metrics(self):
        """
        获取详细的稳定性指标
        返回: dict 包含多个稳定性相关指标
        """
        metrics = {}
        
        # 基本信息
        metrics['box_count'] = len(self.boxes)
        
        if len(self.boxes) == 0:
            return metrics
        
        # 质心信息
        com = self.calculate_center_of_mass()
        if com:
            metrics['center_of_mass'] = com
            metrics['center_offset'] = self.get_center_offset()
            
            # 几何中心
            container_center = (self.physical_width / 2.0, self.physical_length / 2.0)
            metrics['container_center'] = container_center
            
            # 相对偏移率
            max_offset = np.sqrt((self.physical_width/2)**2 + (self.physical_length/2)**2)
            metrics['relative_offset_ratio'] = metrics['center_offset'] / max_offset
        
        # 质量分布统计
        masses = [box.mass for box in self.boxes]
        metrics['total_mass'] = sum(masses)
        metrics['avg_mass'] = np.mean(masses)
        metrics['mass_std'] = np.std(masses)
        metrics['mass_min'] = np.min(masses)
        metrics['mass_max'] = np.max(masses)
        
        # 密度分布统计
        densities = [box.density for box in self.boxes]
        metrics['avg_density'] = np.mean(densities)
        metrics['density_std'] = np.std(densities)
        
        # 空间利用率
        metrics['space_utilization'] = self.get_ratio()
        
        return metrics
    
    def print_stability_report(self):
        """
        打印详细的稳定性报告
        """
        metrics = self.get_stability_metrics()
        
        print("\n" + "="*50)
        print("        稳定性分析报告")
        print("="*50)
        
        print(f"\n[基本信息]")
        print(f"  盒子数量: {metrics.get('box_count', 0)}")
        print(f"  空间利用率: {metrics.get('space_utilization', 0):.2%}")
        
        if 'center_of_mass' in metrics:
            com_x, com_y = metrics['center_of_mass']
            cc_x, cc_y = metrics['container_center']
            
            print(f"\n[质心分析]")
            print(f"  容器几何中心: ({cc_x:.2f}, {cc_y:.2f})")
            print(f"  实际质心位置: ({com_x:.2f}, {com_y:.2f})")
            print(f"  质心偏移量: {metrics['center_offset']:.4f}")
            print(f"  相对偏移率: {metrics['relative_offset_ratio']:.2%}")
            
            # 稳定性评级
            offset = metrics['center_offset']
            if offset < 0.3:
                grade = "优秀 ★★★★★"
            elif offset < 0.5:
                grade = "良好 ★★★★"
            elif offset < 0.7:
                grade = "一般 ★★★"
            else:
                grade = "较差 ★★"
            print(f"  稳定性评级: {grade}")
        
        print(f"\n[质量分布]")
        print(f"  总质量: {metrics.get('total_mass', 0):.2f}")
        print(f"  平均质量: {metrics.get('avg_mass', 0):.2f}")
        print(f"  质量标准差: {metrics.get('mass_std', 0):.2f}")
        print(f"  质量范围: [{metrics.get('mass_min', 0):.2f}, {metrics.get('mass_max', 0):.2f}]")
        
        print(f"\n[密度分布]")
        print(f"  平均密度: {metrics.get('avg_density', 0):.3f}")
        print(f"  密度标准差: {metrics.get('density_std', 0):.3f}")
        
        print("="*50 + "\n")

    # def get_corners(self):
    #     width = self.plain_size[0]
    #     length = self.plain_size[1]
    #     guad = [list() for _ in range(4)]

    #     guad[0].append((width, 0))
    #     guad[1].append((width, length))
    #     guad[2].append((0, length))
    #     guad[3].append((0, 0))

    #     for i in range(1, width):
    #         if self.plain[i, 0] != self.plain[i-1, 0]:
    #             guad[0].append((i, 0))
    #             guad[3].append((i, 0))

    #     for i in range(1, width):
    #         if self.plain[i, length-1] != self.plain[i-1, length-1]:
    #             guad[1].append((i, length))
    #             guad[2].append((i, length))

    #     for j in range(1, length):
    #         if self.plain[0, j] != self.plain[0, j-1]:
    #             guad[2].append((0, j))
    #             guad[3].append((0, j))

    #     for j in range(1, length):
    #         if self.plain[width-1, j] != self.plain[width-1, j]:
    #             guad[0].append((width, j))
    #             guad[1].append((width, j))

    #     for i in range(1, width):
    #         for j in range(1, length):
    #             grid_0 = self.plain[i-1, j]
    #             grid_1 = self.plain[i-1, j-1]
    #             grid_2 = self.plain[i, j-1]
    #             grid_3 = self.plain[i, j]
    #             if grid_0 == grid_1 and grid_2 == grid_3:
    #                 continue
    #             if grid_0 == grid_3 and grid_1 == grid_2:
    #                 continue
    #             if grid_0 != grid_3 or grid_0 != grid_1:
    #                 guad[0].append((i, j))
    #             if grid_1 != grid_0 or grid_1 != grid_2:
    #                 guad[1].append((i, j))
    #             if grid_2 != grid_1 or grid_2 != grid_3:
    #                 guad[2].append((i, j))
    #             if grid_3 != grid_2 or grid_3 != grid_0:
    #                 guad[3].append((i, j))

    #     return guad

    def check_box(self, plain, x, y, lx, ly, z):
        # 检查物理容器边界
        if lx+x > self.physical_width or ly+y > self.physical_length:
            return -1
        if lx < 0 or ly < 0:
            return -1

        rec = plain[lx:lx+x, ly:ly+y]
        r00 = rec[0,0]
        r10 = rec[x-1,0]
        r01 = rec[0,y-1]
        r11 = rec[x-1,y-1]
        rm = max(r00,r10,r01,r11)
        sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
        if sc < 3:
            return -1
        # get the max height
        max_h = np.max(rec)
        # check area and corner
        max_area = np.sum(rec==max_h)
        area = x * y

        # check boundary
        assert max_h >= 0
        if max_h + z > self.height:
            return -1
     
        if max_area/area > 0.95:
            return max_h
        if rm == max_h and sc == 3 and max_area/area > 0.85:
            return max_h
        if rm == max_h and sc == 4 and max_area/area > 0.50:
            return max_h

        return -1

    def get_ratio(self):
        vo = reduce(lambda x, y: x+y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        physical_volume = self.physical_width * self.physical_length * self.plain_size[2]
        ratio = vo / physical_volume
        assert ratio <= 1.0
        return ratio

    def idx_to_position(self, idx):
        lx = idx // self.plain_size[1]
        ly = idx % self.plain_size[1]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.plain_size[0] and position[1] < self.plain_size[1]
        return position[0] * self.plain_size[1] + position[1]

    def drop_box(self, box_size, idx, flag):
        lx, ly = self.idx_to_position(idx)
        if not flag:
            x = box_size[0]
            y = box_size[1]
        else:
            x = box_size[1]
            y = box_size[0]
        z = box_size[2]
        plain = self.plain
        new_h = self.check_box(plain, x, y, lx, ly, z)
        if new_h != -1:
            self.boxes.append(Box(x, y, z, lx, ly, new_h)) # record rotated box
            self.flags.append(flag)
            self.plain = self.update_height_graph(plain, self.boxes[-1])
            self.occupancy[lx:lx + x, ly:ly + y, new_h:new_h + z] = 1
            self.height = max(self.height, new_h + z)
            self.last_wasted_volume = self.get_wasted_volume()
            return True
        return False


