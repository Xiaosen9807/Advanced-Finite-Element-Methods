import gmsh
from matplotlib.font_manager import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

a_b=0.5
mesh_shape=1
mesh_size=10
show=True


def create_mesh(a_b=0.05, mesh_shape=1, mesh_size=8, show=False):
    # 初始化gmsh
    gmsh.initialize()
    gmsh.model.add('2D shape')
    # 创建正方形
    rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, 40, 40)

    b = 20
    a = b*a_b
    # 创建椭圆
    ellipse_tag = gmsh.model.occ.addEllipse(0, 0, 0, b, a)

    # 旋转椭圆90度
    center = [0, 0, 0]
    axis_direction = [0, 0, 1]  # Z轴方向
    angle = 90 * np.pi / 180  # 将角度转换为弧度
    gmsh.model.occ.rotate([(1, ellipse_tag)], center[0], center[1], center[2], axis_direction[0], axis_direction[1], axis_direction[2], angle)

    # 创建一个椭圆的线循环
    ellipse_loop = gmsh.model.occ.addCurveLoop([ellipse_tag])

    # 创建一个椭圆面
    ellipse_surface = gmsh.model.occ.addPlaneSurface([ellipse_loop])

    # 对椭圆进行裁剪
    cut_tag, _ = gmsh.model.occ.cut([(2, rect_tag)], [(2, ellipse_surface)])

    gmsh.option.setNumber("General.Terminal", 0)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Algorithm",8)

    # gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    gmsh.option.setNumber("Mesh.RecombineAll", mesh_shape)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)


    # 同步模型
    gmsh.model.occ.synchronize()

    # 生成2D网格
    gmsh.model.mesh.generate(2)

    # 保存模型和网格
    gmsh.write('2Dshape.msh')

    # 获取所有的节点信息（节点标签，节点坐标和参数化坐标）
    node_tags, node_coords, parametric_coords = gmsh.model.mesh.getNodes()
    # print(node_tags)

    # 获取所有的单元信息（元素类型，元素标签和节点连接性）
    element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()

    # 因为getNodes和getElements返回的数据都是flattened arrays（扁平化数组），我们需要根据节点或单元的维度来reshape数组。
    node_coords = node_coords.reshape(-1, 3)[:, :2]
    # print(node_coords)
    # print(element_types)
    # print(len(element_tags[1]))
    # print(element_nodes )

    if show:
        gmsh.fltk.run()

    # 清理
    gmsh.finalize()
    assert len(element_types)<=3, "The mesh method is wrong, the elements have {} types".format(len(element_types))

    return node_coords, element_nodes[1]-1 

def in_ellipse(xy, a, b):
    x = xy[0]
    y = xy[1]
    return (x/b)**2 + (y/a)**2 <= 1+1e-3

@dataclass
class Node:
    xy: tuple
    id: int
    type='center'
    BC = [0, 0]

def Boundary(node_coords, a_b):
    b=20
    a = b*a_b
    resiual = 1e-3
    Node_list = []
    for i in range(len(node_coords)):
        Node_list.append(Node(xy=node_coords[i],id=i))

    for i in range(len(Node_list)):
        # Check ellipse node
        if in_ellipse(Node_list[i].xy, a, b):
            Node_list[i].type = 'ellipse'
            Node_list[i].BC = [-1, -1]
        # Check left edge
        if abs(Node_list[i].xy[0]-0)<=resiual:
            if abs(Node_list[i].xy[1] - a) <= resiual:
                Node_list[i].type ='lbc'
                Node_list[i].BC = [1, -1]
            else:
                Node_list[i].type = 'le'
                Node_list[i].BC = [1, -1]
        # Check right edge
        elif abs(Node_list[i].xy[0] - 40 )<=resiual:
            Node_list[i].type = 're'
            Node_list[i].BC = [-1, -1] 
        # Check bottom edge
        if abs(Node_list[i].xy[1]-0) <= resiual:
            if abs(Node_list[i].xy[0]-b) <= resiual:
                Node_list[i].type='blc'
                Node_list[i].BC = [-1, 1]
            elif Node_list[i].type=='re':
                Node_list[i].type = 'rbc'
                Node_list[i].BC = [-1, 1]
            else:
                Node_list[i].type = 'be'
                Node_list[i].BC = [-1, 1]

        # Check top edge
        if abs(Node_list[i].xy[1]-40)<=resiual:

            if Node_list[i].type=='le':
                Node_list[i].type ='ltc'
                Node_list[i].BC = [1, -1]
            elif Node_list[i].type == 're':
                Node_list[i].type='rtc'
                Node_list[i].BC = [-1, -1]
            else:
                Node_list[i].type = 'te'
                Node_list[i].BC = [-1, -1]
                
    return Node_list

if __name__=='__main__':
    a_b = 0.5
    mesh_shape=0
    mesh_size= 8
    show = False
    node_coords, element_nodes =  create_mesh(a_b=a_b, mesh_shape=mesh_shape, mesh_size=mesh_size, show=show)
    Nodes_list = Boundary(node_coords, a_b)
        
    print(len(Nodes_list))
    print(len(element_nodes))
    print(element_nodes)
    # print(im=esh_ori.getNodes())
    plt.scatter(node_coords[:, 0], node_coords[:, 1])
    for node in Nodes_list:
        print(node.id, node.xy, node.type, node.BC)

    with open('nodes_data.txt', 'w') as file:
        # 遍历列表中的每个节点
        for node in Nodes_list:
            # 将每个节点的属性写入文件
            # 使用逗号或其他分隔符分隔每个属性，以便于阅读和未来的数据解析
            file.write(f"{node.id}, {node.xy}, {node.type}, {node.BC}\n")
    for (x, y), node in zip(node_coords, Nodes_list):
            # 在指定的坐标处显示文本
            plt.text(x, y, node.id) 

    # 显示图形
   # plt.show()
