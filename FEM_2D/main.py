import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Elements import *
from Mesh import create_mesh, Boundary
from scipy.special import roots_legendre


if __name__=='__main__':
 
   vertices = np.array([[0, 0], [1, 0], [1, 2], [0, 2]])
   Load_x = 50 # N/mm
   Load_y = 0 # N/mm
   A = 40 # mm^2
   a_b = 0.5

   mesh_shape = 1
   mesh_size = 2
   nodes_coord,  element_nodes = create_mesh(a_b, mesh_shape, mesh_size)
   nodes_list = Boundary(nodes_coord, a_b)
   element_list = []
   if mesh_shape == 0:
      element_nodes = element_nodes.reshape(-1, 3)
   elif mesh_shape == 1:
      element_nodes = element_nodes.reshape(-1, 4)


   for ele_lst in element_nodes:
      this_nodes = [node for id in ele_lst for node in nodes_list if node.id == id]
      try:
         element_list.append(Q4(this_nodes))
      except:
         element_list.append(T3(this_nodes))
   DOFs = 2*len(nodes_list)
   glo_K = np.zeros((DOFs, DOFs))
   glo_F = np.zeros(DOFs)
   for elem in element_list:
      loc_F = elem.F
      for i, node_i in enumerate(elem.nodes):
         global_dof = 2 * node_i.id
         if abs(node_i.xy[0]-40)< 1e-3:
            # print(node_i.xy)
            glo_F[global_dof] += loc_F[0] * Load_x 
            glo_F[global_dof + 1] += loc_F[1] * Load_y

   for elem in element_list:
      loc_K = elem.K
      for i, node_i in enumerate(elem.nodes):
         for j, node_j in enumerate(elem.nodes):
               for dof_i in range(2):  # 每个节点的dof: 0和1
                  for dof_j in range(2):
                     # 计算全局dof的位置
                     global_dof_i = 2 * node_i.id + dof_i
                     global_dof_j = 2 * node_j.id + dof_j

                     # 组装到全局矩阵
                     glo_K[global_dof_i][global_dof_j] += loc_K[2*i + dof_i][2*j + dof_j]
   for elem in element_list:
      # ...[现有的组装矩阵代码]...

      for i, node_i in enumerate(elem.nodes):
         for dof_i in range(2):  # 每个节点的dof: 0和1 (x和y方向)
               global_dof_i = 2 * node_i.id + dof_i

               # 检查迪里希莱边界条件
               if node_i.BC[dof_i] == 1:
                  # 修改刚度矩阵和载荷向量
                  glo_K[global_dof_i, :] = 0
                  glo_K[:, global_dof_i] = 0
                  glo_K[global_dof_i, global_dof_i] = 1e15  # 大数约束
                  glo_F[global_dof_i] = 0

   for node in nodes_list:
      print(node.id, node.BC)

   np.set_printoptions(precision=2, suppress=True)
   glo_K[np.abs(glo_K) < 1e-5] = 0
   print('glo_K', glo_K[5:10, 5:10])
   print('glo_F', glo_F)
   U = np.linalg.solve(glo_K, glo_F)
   print(U)
   print(len(U), DOFs)
   for id in range(len(nodes_list)):
      displacement = np.array([U[id*2], U[id*2+1]])
      nodes_list[id].value = displacement

   # plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1])
   for node in nodes_list:
      print(node.id, node.xy, node.type, node.BC, node.value)
  

   # 提取x, y坐标和温度值
   x_coords = [node.xy[0] for node in nodes_list]
   y_coords = [node.xy[1] for node in nodes_list]
   temperatures = [node.value[1] for node in nodes_list]

   # 创建散点图
   plt.scatter(x_coords, y_coords, c=temperatures, cmap='inferno')
   plt.colorbar(label='Displacement')
   plt.title('Displacements Distribution')
   for (x, y), node in zip(nodes_coord, nodes_list):
         # 在指定的坐标处显示文本
      plt.text(x, y, node.id) 
   pass

   plt.show()
