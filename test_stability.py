"""
质心稳定性评估测试脚本
用于验证质心计算和稳定性分析功能
"""

import numpy as np
from envs.bpp0.space import Space, Box

def test_center_of_mass():
    """测试质心计算功能"""
    print("\n" + "="*60)
    print("测试 1: 质心计算功能")
    print("="*60)
    
    # 创建一个10x10x10的容器
    space = Space(width=10, length=10, height=10)
    
    # 测试场景1：单个盒子在中心
    print("\n场景1: 单个盒子放置在容器中心")
    box1 = Box(x=2, y=2, z=2, lx=4, ly=4, lz=0, density=0.75)
    space.boxes.append(box1)
    
    com = space.calculate_center_of_mass()
    offset = space.get_center_offset()
    print(f"  盒子位置: (4, 4), 尺寸: (2, 2, 2)")
    print(f"  盒子质心: {box1.get_center_xy()}")
    print(f"  容器质心: {com}")
    print(f"  质心偏移量: {offset:.4f}")
    print(f"  预期: 偏移量应该接近0（盒子在中心）")
    
    # 测试场景2：添加角落的盒子
    print("\n场景2: 添加一个角落的盒子")
    space.boxes = []
    box2 = Box(x=2, y=2, z=2, lx=0, ly=0, lz=0, density=0.75)
    box3 = Box(x=2, y=2, z=2, lx=8, ly=8, lz=0, density=0.75)
    space.boxes.extend([box2, box3])
    
    com = space.calculate_center_of_mass()
    offset = space.get_center_offset()
    print(f"  盒子1位置: (0, 0), 盒子2位置: (8, 8)")
    print(f"  容器质心: {com}")
    print(f"  质心偏移量: {offset:.4f}")
    print(f"  预期: 两个对角线盒子，质心应该在中心附近")
    
    # 测试场景3：不对称分布
    print("\n场景3: 不对称分布（多个盒子在一侧）")
    space.boxes = []
    # 在左侧放置多个盒子
    for i in range(3):
        box = Box(x=2, y=2, z=2, lx=0, ly=i*2, lz=0, density=0.8)
        space.boxes.append(box)
    
    com = space.calculate_center_of_mass()
    offset = space.get_center_offset()
    print(f"  3个盒子都在左侧 (x=0)")
    print(f"  容器质心: {com}")
    print(f"  质心偏移量: {offset:.4f}")
    print(f"  预期: 偏移量应该较大（不对称）")

def test_stability_metrics():
    """测试稳定性指标计算"""
    print("\n" + "="*60)
    print("测试 2: 稳定性指标计算")
    print("="*60)
    
    space = Space(width=10, length=10, height=10)
    
    # 添加多个不同密度的盒子
    boxes_data = [
        (2, 2, 2, 1, 1, 0, 0.5),
        (3, 3, 3, 5, 5, 0, 0.8),
        (2, 2, 2, 8, 1, 0, 0.6),
        (1, 1, 1, 2, 8, 0, 1.0),
    ]
    
    for x, y, z, lx, ly, lz, density in boxes_data:
        box = Box(x, y, z, lx, ly, lz, density)
        space.boxes.append(box)
    
    # 获取稳定性指标
    metrics = space.get_stability_metrics()
    
    print(f"\n稳定性指标:")
    print(f"  盒子数量: {metrics['box_count']}")
    print(f"  总质量: {metrics['total_mass']:.2f}")
    print(f"  平均质量: {metrics['avg_mass']:.2f}")
    print(f"  质量标准差: {metrics['mass_std']:.2f}")
    print(f"  平均密度: {metrics['avg_density']:.3f}")
    print(f"  质心偏移量: {metrics['center_offset']:.4f}")
    print(f"  相对偏移率: {metrics['relative_offset_ratio']:.2%}")
    
    # 打印完整报告
    space.print_stability_report()

def test_mass_calculation():
    """测试质量计算"""
    print("\n" + "="*60)
    print("测试 3: 质量计算")
    print("="*60)
    
    print("\n创建不同密度的盒子:")
    for i, density in enumerate([0.5, 0.75, 1.0]):
        box = Box(x=2, y=2, z=2, lx=0, ly=0, lz=0, density=density)
        volume = box.x * box.y * box.z
        print(f"  盒子{i+1}: 体积={volume}, 密度={box.density:.2f}, 质量={box.mass:.2f}")
        expected_mass = density * volume
        print(f"    验证: {box.mass:.2f} == {expected_mass:.2f} ? {abs(box.mass - expected_mass) < 0.01}")

def test_random_density():
    """测试随机密度生成"""
    print("\n" + "="*60)
    print("测试 4: 随机密度生成")
    print("="*60)
    
    print("\n生成10个随机密度的盒子:")
    densities = []
    for i in range(10):
        box = Box(x=1, y=1, z=1, lx=0, ly=0, lz=0)  # 不指定密度，使用随机值
        densities.append(box.density)
        print(f"  盒子{i+1}: 密度={box.density:.3f}, 质量={box.mass:.3f}")
    
    print(f"\n密度统计:")
    print(f"  最小值: {min(densities):.3f}")
    print(f"  最大值: {max(densities):.3f}")
    print(f"  平均值: {np.mean(densities):.3f}")
    print(f"  标准差: {np.std(densities):.3f}")
    print(f"  所有密度都在[0.5, 1.0]范围内: {all(0.5 <= d <= 1.0 for d in densities)}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("       质心稳定性评估功能测试")
    print("="*60)
    
    try:
        test_center_of_mass()
        test_stability_metrics()
        test_mass_calculation()
        test_random_density()
        
        print("\n" + "="*60)
        print("       所有测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
