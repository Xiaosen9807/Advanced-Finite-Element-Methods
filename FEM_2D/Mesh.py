import gmsh
import numpy as np
import matplotlib.pyplot as plt
from tools_2D import *
from shape_fns import *


a_b=0.5
mesh_shape=1
mesh_size=10
show=True


def mesh(a_b=0.05, mesh_shape=1, mesh_size=8, show=False):
        
    # 初始化gmsh
    gmsh.initialize()
    gmsh.model.add('2D shape')

    # 创建正方形
    rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, 40, 40)

    b = 20
    a = b*a_b
    # 创建椭圆
    ellipse_tag = gmsh.model.occ.addEllipse(0, 0, 0, b, a)

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


    # 同步模型
    gmsh.model.occ.synchronize()

    # 生成2D网格
    gmsh.model.mesh.generate(2)

    # 保存模型和网格
    gmsh.write('2Dshape.msh')

    # 获取所有的节点信息（节点标签，节点坐标和参数化坐标）
    node_tags, node_coords, parametric_coords = gmsh.model.mesh.getNodes()

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

    return node_coords, element_nodes

if __name__=='__main__':
   node_coords, element_nodes =  mesh(a_b=0.5, mesh_shape=1, mesh_size=4, show=False)
   Nodes_list = []
   for i in range(len(node_coords)):
       Nodes_list.append(Node(xy=node_coords[i],id=i+1))
       
