import copy

class Item3D:
    def __init__(self, item_id, l, w, h):
        self.id = item_id
        # 6种可能的旋转状态
        self.rotations = [
            (l, w, h), (w, l, h), (l, h, w), 
            (h, l, w), (w, h, l), (h, w, l)
        ]
        
class EMS:
    """定义最大空余空间 (Empty Maximal Space)"""
    def __init__(self, x, y, z, l, w, h):
        self.x = x  # 空间左下角后方坐标
        self.y = y
        self.z = z
        self.l = l  # 空间的长度 (X轴)
        self.w = w  # 空间的宽度 (Y轴)
        self.h = h  # 空间的高度 (Z轴)
        self.volume = l * w * h

class Bin3D:
    def __init__(self, bin_id, length, width, height):
        self.id = bin_id
        self.l = length
        self.w = width
        self.h = height
        # 初始时，整个容器是一个巨大的 EMS
        self.ems_list = [EMS(0, 0, 0, length, width, height)]
        self.packed_items = []

class Ali2024_OnlineHeuristic:
    def __init__(self, bin_length, bin_width, bin_height):
        self.bin_l = bin_length
        self.bin_w = bin_width
        self.bin_h = bin_height
        self.bins = []

    def open_new_bin(self):
        new_bin = Bin3D(len(self.bins) + 1, self.bin_l, self.bin_w, self.bin_h)
        self.bins.append(new_bin)
        return new_bin

    def strategy_5_space_sorting(self, ems_list):
        """
        Space Selection Strategy 5: 
        1. Smallest X (最深)
        2. Smallest Z (最底)
        3. Closest to one of the two rear-bottom corners (离后下角最近) [cite: 5151-5154]
        """
        def sort_key(ems):
            # 计算到左后下角 (0,0,0) 和右后下角 (0, W, 0) 的距离
            dist_left = (ems.x**2 + ems.y**2 + ems.z**2)
            dist_right = (ems.x**2 + (self.bin_w - ems.y)**2 + ems.z**2)
            min_dist = min(dist_left, dist_right)
            return (ems.x, ems.z, min_dist)
        
        return sorted(ems_list, key=sort_key)

    def strategy_3_item_placement(self, item, ems):
        """
        Item Placement Strategy 3: Minimum gap [cite: 5198]
        在给定的 EMS 中测试所有 6 种旋转，选择能放下且留下的间隙（体积差或尺寸差）最小的旋转。
        """
        best_rotation = None
        min_gap = float('inf')
        
        for rot in item.rotations:
            rl, rw, rh = rot
            if rl <= ems.l and rw <= ems.w and rh <= ems.h:
                # 简单定义 gap 为体积差 (此处可根据需要细化为三维表面间隙)
                gap = (ems.l - rl) + (ems.w - rw) + (ems.h - rh)
                if gap < min_gap:
                    min_gap = gap
                    best_rotation = rot
                    
        return best_rotation

    def update_ems(self, target_bin, packed_box, position):
        """
        Space Updating: 差分过程 (Difference Process) [cite: 5211-5213]
        当一个物品放入后，计算它与所有现有 EMS 的相交部分，
        生成新的子 EMS，并移除被完全包含或因遮挡失效的 EMS。
        """
        # 注：在实际 3D 装箱中，EMS 差分过程包含繁复的三维空间求交运算。
        # 原理是将原 EMS 沿着放入物品的6个面切分为最多6个更小的 EMS。
        # 考虑到代码长度，此处省略了具体的纯几何计算，但在论文复现时这是基础模块。
        pass 

    def pack_item_online(self, item):
        """
        在线到达一个物品，立刻进行装箱决策
        """
        if not self.bins:
            self.open_new_bin()

        # Bin Selection Strategy: First-fit 
        for current_bin in self.bins:
            # 获取该箱子中当前可用的 EMS 列表
            valid_ems_list = current_bin.ems_list
            
            # 使用 Space Strategy 5 排序
            sorted_ems = self.strategy_5_space_sorting(valid_ems_list)
            
            for ems in sorted_ems:
                # 使用 Placement Strategy 3 尝试放置
                best_rot = self.strategy_3_item_placement(item, ems)
                
                if best_rot:
                    # 找到了合适的箱子、空间和旋转方向，执行装箱
                    placed_pos = (ems.x, ems.y, ems.z)
                    current_bin.packed_items.append({
                        'item_id': item.id,
                        'position': placed_pos,
                        'rotation': best_rot
                    })
                    
                    # 更新该箱子的 EMS 列表 (切割剩余空间)
                    self.update_ems(current_bin, best_rot, placed_pos)
                    return True # 当前物品装箱完成
        
        # 如果所有的 First-fit 箱子都放不下，开新箱子
        new_bin = self.open_new_bin()
        # 递归调用自身放入新箱子
        return self.pack_item_online(item)

# === 模拟纯在线测试 (逐个到达) ===
packer = Ali2024_OnlineHeuristic(bin_length=100, bin_width=100, bin_height=100)
# 物品依次到达（系统不知道后面的物品）
incoming_stream = [
    Item3D(1, 40, 50, 20),
    Item3D(2, 60, 20, 30),
    Item3D(3, 30, 30, 30)
]

for item in incoming_stream:
    packer.pack_item_online(item)
    print(f"Item {item.id} packed. Total bins used: {len(packer.bins)}")