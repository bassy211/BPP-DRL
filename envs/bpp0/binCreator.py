import numpy as np
import copy
import torch
import math

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])


class TrajectoryBoxCreator(BoxCreator):
    """
    从processed_train.pt加载训练数据的BoxCreator
    数据格式: {"Data": {"DN号": [{"weight":, "depth":, "width":, "height":, "number":}, ...]}}
    - 随机选取一个轨迹(DN组)
    - 提取物料的长宽高以及质量信息
    - number表示此物料有多少个，放完后才轮到下一个物料
    - 每个轨迹里的物料随机排序
    - 对每种物料等比例缩小到总数为60
    """
    def __init__(self, data_name=None, target_total=60):
        super().__init__()
        self.data_name = data_name
        self.target_total = target_total  # 目标总数
        self.data = None
        self.trajectory_keys = []
        self.box_index = 0
        self.box_set = []
        self.weights = []  # 存储物料质量信息
        self.recorder = []
        
        # 加载数据
        if data_name is not None:
            self._load_data()
    
    def _load_data(self):
        """加载训练数据"""
        raw_data = torch.load(self.data_name)
        if isinstance(raw_data, dict) and "Data" in raw_data:
            self.data = raw_data["Data"]
        else:
            self.data = raw_data
        self.trajectory_keys = list(self.data.keys())
        print(f"成功加载训练数据，共有 {len(self.trajectory_keys)} 个轨迹(DN组)")
    
    def _scale_to_target(self, items, target_total):
        """
        对每种物料等比例缩小到总数为target_total
        items: [{"depth":, "width":, "height":, "weight":, "number":}, ...]
        返回: 缩放后的物料列表，每个元素为 (depth, width, height, weight, scaled_number)
        """
        # 计算当前总数
        current_total = sum(item.get("number", 1) for item in items)
        
        if current_total == 0:
            return []
        
        # 计算缩放比例
        scale_ratio = target_total / current_total
        
        scaled_items = []
        remaining_total = target_total
        
        for i, item in enumerate(items):
            original_number = item.get("number", 1)
            depth = item.get("depth", 1)
            width = item.get("width", 1)
            height = item.get("height", 1)
            weight = item.get("weight", 0.0)
            
            if i == len(items) - 1:
                # 最后一个物料，使用剩余数量以确保总数精确为target_total
                scaled_number = remaining_total
            else:
                # 等比例缩放，使用四舍五入
                scaled_number = max(1, round(original_number * scale_ratio))
                # 确保不会超过剩余数量
                scaled_number = min(scaled_number, remaining_total - (len(items) - i - 1))
            
            if scaled_number > 0:
                scaled_items.append({
                    "depth": depth,
                    "width": width,
                    "height": height,
                    "weight": weight,
                    "number": scaled_number
                })
                remaining_total -= scaled_number
        
        return scaled_items
    
    def _expand_items_to_boxes(self, items):
        """
        将物料列表展开为单个箱子列表
        items: [{"depth":, "width":, "height":, "weight":, "number":}, ...]
        返回: [(depth, width, height), ...], [weight, ...]
        """
        boxes = []
        weights = []
        for item in items:
            depth = item.get("depth", 1)
            width = item.get("width", 1)
            height = item.get("height", 1)
            weight = item.get("weight", 0.0)
            number = item.get("number", 1)
            
            for _ in range(number):
                boxes.append((depth, width, height))
                weights.append(weight)
        
        return boxes, weights
    
    def reset(self, index=None):
        """重置并选择一个新的轨迹"""
        self.box_list.clear()
        self.recorder = []
        self.box_index = 0
        
        # 随机选取一个轨迹
        if index is None:
            traj_idx = np.random.randint(0, len(self.trajectory_keys))
        else:
            traj_idx = index % len(self.trajectory_keys)
        
        traj_key = self.trajectory_keys[traj_idx]
        items = self.data[traj_key]
        
        # 统计原始总数
        original_total = sum(item.get("number", 1) for item in items)
        
        # 等比例缩放到目标总数
        scaled_items = self._scale_to_target(items, self.target_total)
        
        # 随机打乱物料顺序
        np.random.shuffle(scaled_items)
        
        # 展开为单个箱子列表
        self.box_set, self.weights = self._expand_items_to_boxes(scaled_items)
        
        # 添加一个终止箱子
        self.box_set.append((10, 10, 10))
        self.weights.append(0.0)
    
    def generate_box_size(self, **kwargs):
        """生成下一个箱子"""
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
    
    def get_current_weight(self):
        """获取当前箱子的质量"""
        if self.box_index > 0 and self.box_index <= len(self.weights):
            return self.weights[self.box_index - 1]
        return 0.0


class LoadBoxCreator(BoxCreator):
    """
    从processed_test.pt加载测试数据的BoxCreator
    数据格式: {"Data": {"DN号": [{"weight":, "depth":, "width":, "height":, "number":}, ...]}}
    或者旧格式: [[[depth, width, height], ...], ...]
    - 每个轨迹里的物料随机排序
    - 对每种物料等比例缩小到总数为60
    """
    def __init__(self, data_name=None, target_total=60):
        super().__init__()
        self.data_name = data_name
        self.target_total = target_total
        self.index = 0
        self.box_index = 0
        self.recorder = []
        self.weights = []
        self.box_set = []
        self.data = None
        self.trajectory_keys = []
        self.is_new_format = False
        
        if data_name is not None:
            self._load_data()
    
    def _load_data(self):
        """加载测试数据"""
        raw_data = torch.load(self.data_name)
        
        # 检测数据格式
        if isinstance(raw_data, dict) and "Data" in raw_data:
            # 新格式: {"Data": {"DN号": [...]}}
            self.data = raw_data["Data"]
            self.trajectory_keys = list(self.data.keys())
            self.is_new_format = True
            self.traj_nums = len(self.trajectory_keys)
            print(f"成功加载新格式测试数据，共有 {self.traj_nums} 个轨迹(DN组)")
        elif isinstance(raw_data, dict):
            # 新格式但没有"Data"键
            self.data = raw_data
            self.trajectory_keys = list(self.data.keys())
            self.is_new_format = True
            self.traj_nums = len(self.trajectory_keys)
            print(f"成功加载新格式测试数据，共有 {self.traj_nums} 个轨迹(DN组)")
        else:
            # 旧格式: [[[depth, width, height], ...], ...]
            self.data = raw_data
            self.is_new_format = False
            self.traj_nums = len(self.data)
            print(f"成功加载旧格式测试数据，共有 {self.traj_nums} 个轨迹")
    
    def _scale_to_target(self, items, target_total):
        """
        对每种物料等比例缩小到总数为target_total
        """
        current_total = sum(item.get("number", 1) for item in items)
        
        if current_total == 0:
            return []
        
        scale_ratio = target_total / current_total
        
        scaled_items = []
        remaining_total = target_total
        
        for i, item in enumerate(items):
            original_number = item.get("number", 1)
            depth = item.get("depth", 1)
            width = item.get("width", 1)
            height = item.get("height", 1)
            weight = item.get("weight", 0.0)
            
            if i == len(items) - 1:
                scaled_number = remaining_total
            else:
                scaled_number = max(1, round(original_number * scale_ratio))
                scaled_number = min(scaled_number, remaining_total - (len(items) - i - 1))
            
            if scaled_number > 0:
                scaled_items.append({
                    "depth": depth,
                    "width": width,
                    "height": height,
                    "weight": weight,
                    "number": scaled_number
                })
                remaining_total -= scaled_number
        
        return scaled_items
    
    def _expand_items_to_boxes(self, items):
        """将物料列表展开为单个箱子列表"""
        boxes = []
        weights = []
        for item in items:
            depth = item.get("depth", 1)
            width = item.get("width", 1)
            height = item.get("height", 1)
            weight = item.get("weight", 0.0)
            number = item.get("number", 1)
            
            for _ in range(number):
                boxes.append((depth, width, height))
                weights.append(weight)
        
        return boxes, weights

    def reset(self, index=None):
        """重置并加载指定轨迹"""
        self.box_list.clear()
        self.recorder = []
        self.box_index = 0
        
        if index is None:
            self.index += 1
        else:
            self.index = index
        
        if self.is_new_format:
            # 新格式数据处理
            traj_idx = self.index % len(self.trajectory_keys)
            traj_key = self.trajectory_keys[traj_idx]
            items = self.data[traj_key]
            
            # 统计原始总数
            original_total = sum(item.get("number", 1) for item in items)
            
            # 等比例缩放到目标总数
            scaled_items = self._scale_to_target(items, self.target_total)
            
            # 随机打乱物料顺序
            np.random.shuffle(scaled_items)
            
            # 展开为单个箱子列表
            self.box_set, self.weights = self._expand_items_to_boxes(scaled_items)
        else:
            # 旧格式数据处理
            self.boxes = self.data[self.index % len(self.data)]
            self.box_set = list(self.boxes)
            self.weights = [0.0] * len(self.box_set)
        
        # 添加终止箱子
        self.box_set.append([10, 10, 10])
        self.weights.append(0.0)

    def generate_box_size(self, **kwargs):
        """生成下一个箱子"""
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
    
    def get_current_weight(self):
        """获取当前箱子的质量"""
        if self.box_index > 0 and self.box_index <= len(self.weights):
            return self.weights[self.box_index - 1]
        return 0.0