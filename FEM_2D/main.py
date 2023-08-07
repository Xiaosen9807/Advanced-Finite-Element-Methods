import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Elements import *
from Mesh import create_mesh, identify_nodes
from scipy.special import roots_legendre


if __name__=='__main__':
 
   vertices = np.array([[0, 0], [1, 0], [1, 2], [0, 2]])
      
   a_b = 0.05

   mesh_shape = 1
   mesh_size = 8
   nodes_coord,  element_nodes = create_mesh(a_b, mesh_shape, mesh_size)
   nodes_list = identify_nodes(nodes_coord, a_b)
   element_list = []
   if mesh_shape == 0:
      element_nodes = element_nodes.reshape(-1, 3)
   elif mesh_shape == 1:
      element_nodes = element_nodes.reshape(-1, 4)


   for ele_lst in element_nodes:
      this_nodes = [node for id in ele_lst for node in nodes_list if node.id == id]
      element_list.append(Q4(this_nodes))
   DOFs = 2*len(nodes_list)
   glo_K = np.zeros((DOFs, DOFs))

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

   np.set_printoptions(precision=2, suppress=True)
   glo_K[np.abs(glo_K) < 1e-5] = 0
   print(glo_K[5:10, 5:10])
   num_zeros = np.count_nonzero(glo_K == 0)
   print(glo_K.size-num_zeros)
   print(element_nodes.size)
      
   print(element_list[0].K)
   print(element_list[0].node_id)
