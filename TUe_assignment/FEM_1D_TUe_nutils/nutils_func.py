from nutils import function, mesh, solver
from nutils.expression_v2 import Namespace
import numpy as np
from matplotlib import pyplot as plt
from consts import *
#    plt.xticks(np.linspace(0, 1, 5))
def creat_mesh(num_elems_per_segment = 3  ):
    # 初始化网格数组
    interfaces = interfaces_global
    mesh = np.array([])

    # 遍历界面列表，为每个子区间生成网格
    for i in range(len(interfaces)-1):
        # 当前子区间的起始点和结束点
        start, end = interfaces[i], interfaces[i+1]

        # 在当前子区间内生成等间距的节点
        # np.linspace包括区间的起始和结束点，但为避免重复添加界面节点，我们从第二个节点开始添加（当i不为0时）
        sub_mesh = np.linspace(start, end, num_elems_per_segment + 1)
        if i > 0:
            sub_mesh = sub_mesh[1:]  # 移除子网格的第一个节点，因为它是上一个子网格的最后一个节点

        # 将子网格添加到总网格中
        mesh = np.concatenate((mesh, sub_mesh))
    return mesh

    

mesh_global = creat_mesh(3)
print(mesh_global)

mesh.rectilinear([mesh_global])
topo, geom = mesh.rectilinear([mesh_global])

topo.boundary['left']
basis = topo.basis('spline', degree=1)
print(basis.shape)