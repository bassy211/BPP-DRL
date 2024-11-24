from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import matplotlib.pyplot as plt

def create_box_vertices(x, y, z, lx, ly, lz):
    vertices = [
        [lx, ly, lz], [lx+x, ly, lz], [lx+x, ly+y, lz], [lx, ly+y, lz],
        [lx, ly, lz+z], [lx+x, ly, lz+z], [lx+x, ly+y, lz+z], [lx, ly+y, lz+z]
    ]
    return vertices

def plot_box(ax, vertices, color='blue', alpha=1):
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
        linewidth=0.5,           # 设置边框宽度
        edgecolor='black',     # 设置边框颜色
        facecolor=color        # 设置面颜色
    )
    ax.add_collection3d(collection)

def visualize_boxes(container_size=(10, 10, 10), boxes=None):
    if boxes is None:
        boxes = [(2, 2, 2, 0, 0, 0), (2, 2, 2, 5, 0, 0)]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 添加容器边界框
    container_vertices = create_box_vertices(container_size[0], container_size[1], 
                                          container_size[2], 0, 0, 0)
    plot_box(ax, container_vertices, color='black', alpha=0.1)

    # 绘制boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
    for box, color in zip(boxes, colors):
        x, y, z, lx, ly, lz = box
        vertices = create_box_vertices(x, y, z, lx, ly, lz)
        plot_box(ax, vertices, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, container_size[0])
    ax.set_ylim(0, container_size[1])
    ax.set_zlim(0, container_size[2])

    plt.show()

# if __name__ == '__main__':
#     visualize_boxes()