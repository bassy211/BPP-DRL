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


class Space(object):
    def __init__(self, width=10, length=10, height=10):
        self.plain_size = np.array([width, length, height])
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.boxes = []
        self.flags = [] # record rotation information
        self.height = height

    def print_height_graph(self):
        print(self.plain)

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
        # 容器几何中心
        container_center_x = self.plain_size[0] / 2.0
        container_center_y = self.plain_size[1] / 2.0
        
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
            container_center = (self.plain_size[0] / 2.0, self.plain_size[1] / 2.0)
            metrics['container_center'] = container_center
            
            # 相对偏移率
            max_offset = np.sqrt((self.plain_size[0]/2)**2 + (self.plain_size[1]/2)**2)
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
        """
        基于重心的稳定性检查
        盒子视为具有均匀质量分布，几何中心点必须受支撑才稳定
        
        Args:
            plain: 高度图 (width x length)
            x, y: 盒子尺寸 (x方向长度, y方向长度)
            lx, ly: 放置位置 (左下角坐标)
            z: 盒子高度
        
        Returns:
            放置高度 (成功) 或 -1 (失败)
        """
        # 边界检查
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return -1
        if lx < 0 or ly < 0:
            return -1
        
        # 获取放置区域的高度分布
        rec = plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)
        
        # 高度边界检查
        assert max_h >= 0
        if max_h + z > self.height:
            return -1
        
        # 计算几何中心点在底面的投影位置 (浮点坐标)
        center_x = lx + x / 2.0  # 在plain坐标系中的x坐标
        center_y = ly + y / 2.0  # 在plain坐标系中的y坐标
        
        # 将中心点转换为网格索引
        # 使用floor找到中心点所在的网格单元
        cx_floor = int(np.floor(center_x))
        cy_floor = int(np.floor(center_y))
        
        # 确保索引在有效范围内
        cx_floor = min(cx_floor, lx + x - 1)
        cy_floor = min(cy_floor, ly + y - 1)
        
        # 条件1: 检查几何中心点是否直接由最高点支撑
        # 中心点所在网格的高度等于最大高度
        center_height = rec[cx_floor - lx, cy_floor - ly]
        
        # 即使中心点在最高点，也需要检查支撑是否足够
        # 找到所有最高点的位置
        max_height_mask = (rec == max_h)
        max_height_points = np.argwhere(max_height_mask)
        
        if len(max_height_points) == 0:
            return -1
        
        # 额外的合理性检查：支撑面积不能太小
        # 至少需要盒子底面积的一定比例作为支撑
        support_area = len(max_height_points)
        box_area = x * y
        min_support_ratio = 0.15  # 至少15%的支撑面积
        
        if support_area / box_area < min_support_ratio:
            # 支撑面积太小，不稳定
            return -1
        
        # 如果中心点直接在最高点上，还需要检查支撑点的分布
        if center_height == max_h:
            # 对于小盒子（只有1或2个格子）或支撑充分的情况，直接接受
            if box_area <= 2 or support_area / box_area > 0.35:  # 更严格：只有1x1或1x2盒子可以直接通过
                return max_h
            # 否则需要检查支撑点的分布是否合理（继续后续检查）
        
        # 条件2: 检查几何中心点是否被支撑点的凸包包围
        # 特殊情况：如果只有一个支撑点，要求盒子必须很小且中心几乎在支撑点上
        if len(max_height_points) == 1:
            px, py = max_height_points[0]
            # 转换回实际坐标
            point_x = lx + px + 0.5  # 网格中心
            point_y = ly + py + 0.5
            # 单点支撑不稳定，只接受很小的盒子(1x1或2x2)且中心非常接近支撑点
            if x <= 2 and y <= 2:
                dist_sq = (center_x - point_x)**2 + (center_y - point_y)**2
                # 对于2x2盒子，要求距离平方 < 0.5（不包括边界情况）
                # 对于1x1或1x2盒子，要求距离平方 <= 0.5
                threshold = 0.4 if (x == 2 and y == 2) else 0.5
                if dist_sq <= threshold:
                    return max_h
            return -1
        
        # 如果有多个支撑点，检查中心是否在凸包内
        # 将网格索引转换为实际坐标 (网格中心点)
        support_points = []
        for px, py in max_height_points:
            point_x = lx + px + 0.5
            point_y = ly + py + 0.5
            support_points.append([point_x, point_y])
        support_points = np.array(support_points)
        
        # 使用叉积方法检查点是否在凸包内
        # 对于2D情况，我们可以使用简化的方法
        if len(max_height_points) == 2:
            # 两个点的情况：检查中心是否在两点连线附近
            p1, p2 = support_points[0], support_points[1]
            # 计算中心点到线段的距离
            line_vec = p2 - p1
            point_vec = np.array([center_x, center_y]) - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq < 1e-6:
                # 两点重合
                dist_sq = np.dot(point_vec, point_vec)
                return max_h if dist_sq <= 0.5 else -1
            
            # 投影参数
            t = np.dot(point_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)  # 限制在线段上
            projection = p1 + t * line_vec
            dist_sq = np.sum((np.array([center_x, center_y]) - projection)**2)
            return max_h if dist_sq <= 0.5 else -1
        
        # 三个或更多点：使用凸包检查
        # 简化版本：检查中心点是否在支撑点的边界框内，并且足够接近支撑区域
        min_x = support_points[:, 0].min()
        max_x = support_points[:, 0].max()
        min_y = support_points[:, 1].min()
        max_y = support_points[:, 1].max()
        
        # 额外检查：支撑点的分布范围（相对于盒子尺寸）
        # 如果支撑点太集中，即使中心在其边界框内也不稳定
        support_span_x = max_x - min_x
        support_span_y = max_y - min_y
        
        # 要求支撑点的跨度至少覆盖盒子尺寸的一定比例
        min_span_ratio = 0.4  # 支撑点至少横跨盒子40%的长度
        if support_span_x < x * min_span_ratio and support_span_y < y * min_span_ratio:
            # 支撑点分布太集中，不稳定
            return -1
        
        # 如果中心在支撑点的边界框内，认为是稳定的
        if min_x <= center_x <= max_x and min_y <= center_y <= max_y:
            return max_h
        
        # 如果中心在边界框外但很接近，也接受
        # 计算到边界框的最小距离
        dx = max(0, min_x - center_x, center_x - max_x)
        dy = max(0, min_y - center_y, center_y - max_y)
        dist_sq = dx**2 + dy**2
        
        if dist_sq <= 0.5:  # 允许中心点距离边界框不超过0.5个单元
            return max_h
        
        return -1

    def get_ratio(self):
        vo = reduce(lambda x, y: x+y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
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
            self.height = max(self.height, new_h + z)
            return True
        return False


