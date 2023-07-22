import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Mesh import mesh

from scipy.special import roots_legendre


if __name__=='__main__':
 
   vertices = [[1, 0.9], [2, 1.3], [0.5, 1.7]]
   Nodes_list = []
   for i in range(len(vertices)):
      Nodes_list.append(Node(vertices[i], i))
      
   triangle = T3(Nodes_list)
   print(triangle.phipxs[1](x=[0.2, 0.3]))
   # print(mul(triangle.phipxs[0], triangle.phipxs[1])([0]))
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

   nodes_coord, elements_coord = mesh(0.05, 0, 8)
   nodes_list = []
   eleemnt_list = []
   for i in range(len(nodes_coord)):
      nodes_list.append(Node(nodes_coord, i+1))
   print(len(nodes_coord))
   
