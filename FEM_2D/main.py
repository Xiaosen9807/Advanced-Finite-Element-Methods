import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Elements import *
from Mesh import mesh

from scipy.special import roots_legendre

def identi_nodes(nodes_coord, )

if __name__=='__main__':
 
   vertices = [[1, 0.9], [2, 1.3], [0.5, 1.7]]
   Nodes_list = []
   for i in range(len(vertices)):
      Nodes_list.append(Node(vertices[i], i))
      
   triangle = T3(Nodes_list)
   print(triangle.phipxs[1](x=[0.2, 0.3]))
   # print(mul(triangle.ihiixs[0], triangle.phipxs[1])([0]))
   print('J_det', triangle.J_det)
   print('J', np.linalg.inv(triangle.J_inv))
   
   x, w = roots_legendre(3)
   print(x, w)

   num_shape_fns = len(triangle.phis)
   K = np.zeros((num_shape_fns, num_shape_fns))
   invers_J = triangle.J_inv*triangle.J_det
   F = np.zeros(num_shape_fns)
   for i in range(num_shape_fns):
       
       for j in range(num_shape_fns):
           pu_pxi = plus(mul(invers_J[0][0], triangle.phipxs[i]),
                        mul(invers_J[0][1], triangle.phipys[i]))
           pu_pxj = plus(mul(invers_J[0][0], triangle.phipxs[j]),
                        mul(invers_J[0][1], triangle.phipys[j]))
           pu_pyi = plus(mul(invers_J[1][0], triangle.phipxs[i]),
                        mul(invers_J[1][1], triangle.phipys[i]))
           pu_pyj = plus(mul(invers_J[1][0], triangle.phipxs[j]),
                        mul(invers_J[1][1], triangle.phipys[j]))
           this_funs = plus(mul(pu_pxi, pu_pxj), mul(pu_pyi, pu_pyj))
           K[i, j] = G_integrate_2D(this_funs)

   print(K)
   print(triangle.K)

   nodes_coord,  element_nodes = mesh(0.05, 0, 10)
   nodes_list = []
   element_list = []
   for i in range(len(nodes_coord)):
      nodes_list.append(Node(nodes_coord[i], i))
   element_nodes = element_nodes.reshape(-1, 3)


   for ele_lst in element_nodes:
      this_nodes = [node for id in ele_lst for node in nodes_list if node.id == id]
      element_list.append(T3(this_nodes))
   print(len(nodes_list))
   glo_K = np.zeros((len(nodes_list), len(nodes_list)))

   for elem in element_list:
      loc_K = elem.K
      # print(loc_K)
      for i in range(len(loc_K)):
         for j in range(len(loc_K)):
            node_i = elem.nodes[i]
            node_j = elem.nodes[j]
            # print(node_i.id, node_j.id)
            glo_K[node_i.id][node_j.id] += loc_K[i][j]
   np.set_printoptions(precision=2, suppress=True)
   glo_K[np.abs(glo_K) < 1e-5] = 0
   print(glo_K[5:10, 5:10])
   num_zeros = np.count_nonzero(glo_K == 0)
   print(glo_K.size-num_zeros)
   print(element_nodes.size)
      
   print(element_list[0].K)
   print(element_list[0].node_id)
