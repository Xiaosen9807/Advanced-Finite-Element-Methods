import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from scipy.special import roots_legendre


if __name__ =='__main__':
 
   vertices = [[1, 0.9], [2, 1.3], [0.5, 1.7]]
   triangle = T3(vertices)
   print(triangle.phipxs[1](x=[0.2, 0.3]))
   print(mul(triangle.phipxs[0], triangle.phipxs[1])([0]))
   print('J_det', triangle.J_det)
   print('J', np.linalg.inv(triangle.J_inv))
   
   x, w = roots_legendre(3)
   print(x, w)
   
   K = np.zeros((3, 3))
   invers_J = triangle.J_inv*triangle.J_det

   for i in range(3):
       for j in range(3):
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
