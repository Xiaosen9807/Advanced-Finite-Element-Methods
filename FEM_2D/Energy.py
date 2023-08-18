import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from shape_fns import *
from Elements import *
from tools_2D import *                
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获取脚本所在的目录
script_dir = os.path.dirname(script_path)
# 更改工作目录到脚本所在的目录
os.chdir(script_dir)

if __name__=='__main__':
    with open("Data/data.pkl", "rb") as f:
        data_ori = pickle.load(f)
    index = -6
    E = 200e3
    nu = 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
        ])
    elements_list = data_ori[index]['elements_list']
    data_keys = list(data_ori[index].keys())[:3]
    for key in data_keys:
        print(key, data_ori[index][key])

    energy = 0
    GPN = 2
    for elem in elements_list:
        U_list = np.zeros((2*elem.n_nodes, 1))
        points, Ws = Gauss_points(elem, GPN)
        for i in range(elem.n_nodes) :
            node = elem.nodes[i]
            U = node.value
            U_list[2*i] =   node.value[0]
            U_list[2*i+1] = node.value[1]
        elem_energy = 0
        for g in range(len(Ws)):
            xy = points[g]
            W = Ws[g]
            dN = elem.gradshape(xy[0], xy[0])
            J = jacobian(elem.vertices, dN)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)
            B = elem.B_matrix(J, dN)
            # print('U_list', U_list)
            # print('B', B)
            # print('D', D)
            elem_energy+=0.5 * W * U_list.T @ B.T @ D @ B @ U_list
        energy += elem_energy[0][0]
    print(energy)

    energy_2 = 0
    for elem in elements_list:
        U_list = np.zeros((2*elem.n_nodes,))
        strain_list = np.zeros_like(U_list)
        for i in range(elem.n_nodes) :
            node = elem.nodes[i]
            U = node.value
            U_list[2*i] += U[0] 
            U_list[2*i+1] += U[1]
        energy_2 += 0.5 * U_list.T @ elem.K @ U_list #* elem.area
    print(energy_2)
    energy_3 = 0
    for elem in elements_list:
        U_list = np.zeros((2*elem.n_nodes,))


