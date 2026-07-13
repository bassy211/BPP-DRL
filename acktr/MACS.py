import numpy as np
import copy

class Box2D:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0

class Heightmap_MACS_Packer:
    def __init__(self, container_width, container_max_height):
        """
        初始化装箱器
        :param container_width: 容器宽度 (离散化后的网格数)
        :param container_max_height: 容器最大高度 (用于计算上方可用空间限制)
        """
        self.width = container_width
        self.max_height = container_max_height
        # 初始化高度图为 0，长度为容器宽度
        self.heightmap = np.zeros(self.width, dtype=int)
        self.placed_boxes = []

    def find_candidate_locations(self, box_width):
        """
        基于高度图寻找合法的候选放置位置 (X坐标)
        为了满足盒子宽度，我们需要确保从 x 到 x+box_width 不越界。
        """
        candidates = []
        for x in range(self.width - box_width + 1):
            # 在基于高度图的启发式中，盒子放入 x 位置后，其底部的实际支撑高度
            # 取决于该区间内高度图的最大值 (重力下落直到碰到最高点)
            support_y = np.max(self.heightmap[x : x + box_width])
            
            # 检查是否超出容器总高度限制
            if support_y + box_width <= self.max_height:
                candidates.append((x, support_y))
                
        return candidates

    def simulate_placement(self, current_heightmap, x, box_width, box_height):
        """
        模拟放置盒子并返回更新后的高度图
        """
        new_heightmap = np.copy(current_heightmap)
        # 计算该区域的支撑高度
        support_y = np.max(new_heightmap[x : x + box_width])
        # 更新该区域的高度图
        new_heightmap[x : x + box_width] = support_y + box_height
        return new_heightmap

    def evaluate_convex_space(self, current_heightmap):
        """
        核心逻辑：评估当前高度图上方的最大可访问凸空间 (最大空余矩形面积)
        实现思路：对于每一个可能的宽度 w (从 1 到 container_width)，
        我们都可以找到一个最高点，其上方直到 container_max_height 都是连续的矩形凸空间。
        """
        max_convex_area = 0
        
        # 遍历所有可能的区间起点 i 和终点 j
        for i in range(self.width):
            for j in range(i, self.width):
                w = j - i + 1
                # 该区间内高度图的最高点，即凸空间的底部
                floor_y = np.max(current_heightmap[i : j + 1])
                
                # 如果超出了最大高度，可用空间为 0
                if floor_y >= self.max_height:
                    continue
                
                # 计算该区间上方能形成的矩形凸空间面积
                h = self.max_height - floor_y
                area = w * h
                
                if area > max_convex_area:
                    max_convex_area = area
                    
        return max_convex_area

    def place_box_macs(self, box):
        """
        使用 MACS 策略执行单个盒子的放置
        """
        candidate_locations = self.find_candidate_locations(box.width)
        
        if not candidate_locations:
            return False # 无合法位置，放置失败

        best_x = -1
        best_y = -1
        max_remaining_space = -1

        # 遍历所有候选位置
        for (x, y) in candidate_locations:
            # 1. 模拟放置
            simulated_heightmap = self.simulate_placement(self.heightmap, x, box.width, box.height)
            
            # 2. 评估放置后的剩余最大可访问凸空间
            remaining_space = self.evaluate_convex_space(simulated_heightmap)
            
            # 3. 贪心选择最大剩余空间的位置
            if remaining_space > max_remaining_space:
                max_remaining_space = remaining_space
                best_x = x
                best_y = y

        # 实际执行最优放置
        if best_x != -1:
            box.x = best_x
            box.y = best_y
            self.placed_boxes.append(box)
            # 更新真实高度图
            self.heightmap = self.simulate_placement(self.heightmap, best_x, box.width, box.height)
            return True
        
        return False