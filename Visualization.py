from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import namedtuple
import pyvista as pv

def get_color_palette(style='modern', num_colors=18):
    """
    获取不同风格的颜色调色板
    
    Args:
        style: 颜色风格 ('modern', 'pastel', 'vibrant', 'cool', 'warm')
        num_colors: 需要的颜色数量
    
    Returns:
        颜色列表
    """
    palettes = {
        'modern': [
            '#FF6B6B',  # 珊瑚红
            '#4ECDC4',  # 青绿色
            '#45B7D1',  # 天蓝色
            '#96CEB4',  # 薄荷绿
            '#FFEAA7',  # 淡黄色
            '#DDA0DD',  # 梅花色
            '#98D8C8',  # 浅绿色
            '#F7DC6F',  # 金黄色
            '#BB8FCE',  # 淡紫色
            '#85C1E9',  # 浅蓝色
            '#F8C471',  # 桃色
            '#82E0AA',  # 浅绿色
            '#F1948A',  # 浅红色
            '#D2B4DE',  # 薰衣草色
            '#A3E4D7',  # 水绿色
            '#FAD7A0',  # 杏色
            '#AED6F1',  # 粉蓝色
            '#F5B7B1',  # 粉红色
        ],
        'pastel': [
            '#FFB3BA',  # 粉红色
            '#FFDFBA',  # 桃色
            '#FFFFBA',  # 淡黄色
            '#BAFFC9',  # 淡绿色
            '#BAE1FF',  # 淡蓝色
            '#E1BAFF',  # 淡紫色
            '#FFBABA',  # 浅红色
            '#BAFFFF',  # 淡青色
            '#C9FFBA',  # 浅绿色
            '#FFBAFF',  # 粉紫色
            '#BFFFBA',  # 薄荷色
            '#FFE1BA',  # 香槟色
            '#BABFFF',  # 薰衣草蓝
            '#FFBAC9',  # 樱花粉
            '#C9BAFF',  # 淡丁香色
            '#BAFFC9',  # 春绿色
            '#FFBAE1',  # 粉玫瑰色
            '#BAE1C9',  # 海绿色
        ],
        'vibrant': [
            '#FF0066',  # 鲜红色
            '#00CCFF',  # 鲜蓝色
            '#00FF66',  # 鲜绿色
            '#FF6600',  # 橙色
            '#6600FF',  # 紫色
            '#FFCC00',  # 黄色
            '#FF0099',  # 品红色
            '#00FFCC',  # 青色
            '#66FF00',  # 酸橙绿
            '#FF3300',  # 红橙色
            '#0066FF',  # 蓝色
            '#CC00FF',  # 洋红色
            '#00FF99',  # 春绿色
            '#FF9900',  # 深橙色
            '#3300FF',  # 蓝紫色
            '#CCFF00',  # 黄绿色
            '#FF0033',  # 深红色
            '#0099FF',  # 天蓝色
        ],
        'cool': [
            '#2E86AB',  # 钢蓝色
            '#A23B72',  # 梅红色
            '#F18F01',  # 橙色
            '#C73E1D',  # 砖红色
            '#5D737E',  # 蓝灰色
            '#64A6BD',  # 天蓝色
            '#90A959',  # 橄榄绿
            '#F4A261',  # 沙橙色
            '#E76F51',  # 陶土色
            '#264653',  # 深绿色
            '#2A9D8F',  # 蓝绿色
            '#E9C46A',  # 金黄色
            '#F4A261',  # 橙色
            '#E76F51',  # 红色
            '#577590',  # 蓝色
            '#43AA8B',  # 绿色
            '#90E0EF',  # 浅蓝色
            '#F9844A',  # 橙红色
        ],
        'warm': [
            '#FFBE0B',  # 金黄色
            '#FB5607',  # 橙红色
            '#FF006E',  # 品红色
            '#8338EC',  # 紫色
            '#3A86FF',  # 蓝色
            '#F77F00',  # 橙色
            '#FCBF49',  # 黄色
            '#F71735',  # 红色
            '#FF9F1C',  # 橙黄色
            '#FFB3C6',  # 粉红色
            '#FB8500',  # 深橙色
            '#FF5E5B',  # 珊瑚色
            '#FFD23F',  # 亮黄色
            '#EE6C4D',  # 砖红色
            '#F72585',  # 玫红色
            '#B5179E',  # 紫红色
            '#7209B7',  # 深紫色
            '#480CA8',  # 靛青色
        ]
    }
    
    selected_palette = palettes.get(style, palettes['modern'])
    
    # 如果需要更多颜色，循环使用调色板
    if num_colors > len(selected_palette):
        multiplier = (num_colors // len(selected_palette)) + 1
        extended_palette = selected_palette * multiplier
        return extended_palette[:num_colors]
    
    return selected_palette[:num_colors]

def create_box_vertices(x, y, z, lx, ly, lz):
    vertices = [
        [lx, ly, lz], [lx+x, ly, lz], [lx+x, ly+y, lz], [lx, ly+y, lz],
        [lx, ly, lz+z], [lx+x, ly, lz+z], [lx+x, ly+y, lz+z], [lx, ly+y, lz+z]
    ]
    return vertices

def plot_box(ax, vertices, color='blue', alpha=0.7):
    """绘制一个3D盒子"""
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    collection = Poly3DCollection(
        faces, 
        alpha=alpha,
        linewidth=0.8,           # 设置边框宽度
        edgecolor='black',       # 设置边框颜色
        facecolor=color          # 设置面颜色
    )
    ax.add_collection3d(collection)

def create_plotly_box_mesh(x, y, z, lx, ly, lz, color='blue', opacity=1, name='Box'):
    """使用plotly创建3D盒子网格"""
    # 定义盒子的8个顶点
    vertices = np.array([
        [lx, ly, lz], [lx+x, ly, lz], [lx+x, ly+y, lz], [lx, ly+y, lz],
        [lx, ly, lz+z], [lx+x, ly, lz+z], [lx+x, ly+y, lz+z], [lx, ly+y, lz+z]
    ])
    
    # 定义12个三角面片（每个面由2个三角形组成）
    faces = np.array([
        [0, 1, 2], [0, 2, 3], 
        [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2]
    ])
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1], 
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showscale=False,
        hovertemplate=f'<b>{name}</b><br>'
                     f'Size: ({x}, {y}, {z})<br>'
                     f'Position: ({lx}, {ly}, {lz})<br>'
                     f'<extra></extra>'
    )

def create_pyvista_box(x, y, z, lx, ly, lz):
    """使用pyvista创建3D盒子"""
    # 创建一个立方体网格
    mesh = pv.Box(bounds=[lx, lx+x, ly, ly+y, lz, lz+z])
    return mesh

def visualize_boxes_enhanced(container_size=(10, 10, 10), boxes=None, method='pyvista', color_style='modern'):
    if method.lower() == 'pyvista':
        return visualize_boxes_pyvista_themed(container_size, boxes, color_style)
    elif method.lower() == 'plotly':
        return visualize_boxes_plotly_themed(container_size, boxes, color_style)
    elif method.lower() == 'matplotlib':
        return visualize_boxes_matplotlib_themed(container_size, boxes, color_style)
    else:
        raise ValueError("method must be one of: 'pyvista', 'plotly', 'matplotlib'")

def visualize_boxes_pyvista_themed(container_size=(10, 10, 10), boxes=None, color_style='modern'):
    """使用pyvista创建带主题颜色的3D可视化"""
    # 创建绘图器
    plotter = pv.Plotter(window_size=[1000, 800])
    
    # 设置背景颜色
    plotter.background_color = 'white'
    
    # 创建容器边框
    cx, cy, cz = container_size
    container_outline = pv.Box(bounds=[0, cx, 0, cy, 0, cz])
    # 只显示边框
    plotter.add_mesh(container_outline, style='wireframe', color='gray', 
                     line_width=3, opacity=1.0, label='Container')
    
    # 使用指定主题的颜色调色板
    colors = get_color_palette(color_style, len(boxes))
    
    # 添加盒子
    for i, box in enumerate(boxes):
        x, y, z, lx, ly, lz = box
        
        # 创建盒子网格
        box_mesh = create_pyvista_box(x, y, z, lx, ly, lz)
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 添加盒子到场景 - 显式指定渲染样式为surface
        plotter.add_mesh(
            box_mesh, 
            color=color, 
            opacity=1,
            show_edges=True,
            edge_color='black',
            line_width=1.5,
            style='surface',  # 显式指定为surface渲染
            label=f'Box {i+1}',
            smooth_shading=True  # 平滑着色
        )
    
    # # 添加坐标轴
    # plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    # 设置相机位置以获得更好的视角
    plotter.camera_position = [(cx*1.5, cy*1.5, cz*1.5), (cx/2, cy/2, cz/2), (0, 0, 1)]
    
    # 显示
    plotter.show()
    
    return plotter

def visualize_boxes_plotly_themed(container_size=(10, 10, 10), boxes=None, color_style='modern'):
    """使用plotly创建带主题颜色的3D可视化"""
    fig = go.Figure()
    
    # 创建容器边框线
    cx, cy, cz = container_size
    
    # 容器的12条边
    edges = [
        # 底面4条边
        ([0, cx], [0, 0], [0, 0]),
        ([cx, cx], [0, cy], [0, 0]),
        ([cx, 0], [cy, cy], [0, 0]),
        ([0, 0], [cy, 0], [0, 0]),
        # 顶面4条边
        ([0, cx], [0, 0], [cz, cz]),
        ([cx, cx], [0, cy], [cz, cz]),
        ([cx, 0], [cy, cy], [cz, cz]),
        ([0, 0], [cy, 0], [cz, cz]),
        # 垂直4条边
        ([0, 0], [0, 0], [0, cz]),
        ([cx, cx], [0, 0], [0, cz]),
        ([cx, cx], [cy, cy], [0, cz]),
        ([0, 0], [cy, cy], [0, cz])
    ]
    
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2],
            mode='lines',
            line=dict(color='gray', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 使用指定主题的颜色调色板
    colors = get_color_palette(color_style, len(boxes))
    
    # 添加盒子
    for i, box in enumerate(boxes):
        x, y, z, lx, ly, lz = box
        
        # 创建盒子网格
        box_mesh = create_plotly_box_mesh(
            x, y, z, lx, ly, lz, 
            color=colors[i % len(colors)], 
            opacity=1,
            name=f'Box {i+1}'
        )
        fig.add_trace(box_mesh)
    
    # 设置布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Length)',
            yaxis_title='Y (Width)', 
            zaxis_title='Z (Height)',
            xaxis=dict(range=[0, container_size[0]], showgrid=True),
            yaxis=dict(range=[0, container_size[1]], showgrid=True),
            zaxis=dict(range=[0, container_size[2]], showgrid=True),
            aspectmode='cube',  # 保持比例
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # 设置默认观察角度
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=80)
    )
    
    fig.show()
    return fig

def visualize_boxes_matplotlib_themed(container_size=(10, 10, 10), boxes=None, color_style='modern'):
    """使用matplotlib创建带主题颜色的3D可视化"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 添加容器边界框
    container_vertices = create_box_vertices(container_size[0], container_size[1], 
                                          container_size[2], 0, 0, 0)
    plot_box(ax, container_vertices, color='gray', alpha=0.1)

    # 使用指定主题的颜色调色板
    colors = get_color_palette(color_style, len(boxes))
    total_volume = 0
    
    # 优化点：根据Y坐标排序盒子，改善遮挡
    sorted_boxes_with_colors = sorted(zip(boxes, colors), key=lambda item: item[0][4] + item[0][1]/2, reverse=True)
    
    for i, (box, color) in enumerate(sorted_boxes_with_colors):
        x, y, z, lx, ly, lz = box
        vertices = create_box_vertices(x, y, z, lx, ly, lz)
        plot_box(ax, vertices, color=color, alpha=0.9)

    # 设置坐标轴标签
    ax.set_xlabel('X (Length)', fontsize=12)
    ax.set_ylabel('Y (Width)', fontsize=12)
    ax.set_zlabel('Z (Height)', fontsize=12)

    ax.set_xlim(0, container_size[0])
    ax.set_ylim(0, container_size[1])
    ax.set_zlim(0, container_size[2])

    # 优化点：设置更好的观察角度
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()
    
    return ax

if __name__ == '__main__':
    # 演示数据
    test_boxes = [
        (2, 2, 2, 0, 0, 0),
        (3, 3, 3, 2, 0, 0),
        (1, 4, 2, 5, 0, 0),
        (2, 2, 3, 0, 2, 0),
        (1, 1, 4, 6, 4, 0),
    ]
    
    # 演示不同的颜色主题
    color_themes = ['modern', 'pastel', 'vibrant', 'cool', 'warm']
    for theme in color_themes:
        try:
            visualize_boxes_enhanced(
                container_size=(10, 10, 10), 
                boxes=test_boxes, 
                method='pyvista',
                color_style=theme
            )
        except Exception as e:
            print(f" {theme.upper()} 失败: {e}")

    '''
    可用的颜色主题:
    1. Modern  - 现代风格，色彩柔和协调
    2. Pastel  - 马卡龙色系，淡雅温馨
    3. Vibrant - 鲜艳色彩，对比强烈
    4. Cool    - 冷色调，专业商务风
    5. Warm    - 暖色调，活力四射
    ''' 
