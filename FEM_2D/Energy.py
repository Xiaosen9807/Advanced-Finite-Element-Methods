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

def cal_energy(elements_list, GPN = 2):
    E = 200e3
    nu = 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
        ])
    energy = 0
    for elem in elements_list:
        U_list = np.zeros((2*elem.n_nodes, 1))
        points, Ws = Gauss_points(elem, GPN)
        scale = 4 if elem.shape=="triangle" else 1
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
            elem_energy+=0.5 * W * U_list.T @ B.T @ D @ B @ U_list * scale
        energy += elem_energy[0][0]
    return energy


def cal_energy_exact(elements_list, GPN = 2):
    E = 200e3
    nu = 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
        ])
    energy = 0
    for elem in elements_list:
        elem_energy = 0
        points, Ws = Gauss_points(elem, GPN)
        loop = 0
        scale = 4 if elem.shape=="triangle" else 1
        for g in range(len(Ws)):
            loc_xy = points[g]
            W = Ws[g]
            NP_net = elem.gradshape(loc_xy[0], loc_xy[1])
            J = jacobian(vertices, NP_net)
            xy = J @ loc_xy.T
            stress_list = np.zeros((3,))
            stress_list[0] = exact_fn(xy[0], xy[1], 'x')
            stress_list[1] = exact_fn(xy[0], xy[1], 'y')
            stress_list[2] = exact_fn(xy[0], xy[1], 'xy')
            this_energy = 0.5 * W * stress_list.T @ np.linalg.inv(D) @ stress_list * scale
            elem_energy += this_energy 
            loop+=1
        energy+=elem_energy
    return energy

def cal_energy_2(elements_list, GPN = 2):
    E = 200e3
    nu = 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
        ])
    energy = 0
    for elem in elements_list:
        elem_energy = 0
        points, Ws = Gauss_points(elem, GPN)
        loop = 0
        scale = 4 if elem.shape=="triangle" else 1
        for g in range(len(Ws)):
            xy = points[g]
            W = Ws[g]
            strain_list = np.zeros((3,))
            strain_list[0] = elem(xy[0], xy[1], 'x', 'strain')
            strain_list[1] = elem(xy[0], xy[1], 'y', 'strain')
            strain_list[2] = elem(xy[0], xy[1], 'xy', 'strain')
            this_energy = 0.5 * W * strain_list.T @ D @ strain_list * scale
            elem_energy += this_energy 
            loop+=1
        energy+=elem_energy
    return energy
def save_energy(data_ori, save=True):
    data_1 = []
    data_05 = []
    data_005 = []
    for i in range(len(data_ori)):
        elements_list = data_ori[i]['elements_list']
        this_data = {}
        data_keys = list(data_ori[i].keys())[:4]
        for key in data_keys:
            if save:
                print(key, data_ori[i][key])
            this_data[key] = data_ori[i][key]
        this_data['energy'] = cal_energy(elements_list)
        if data_ori[i]['a_b'] == 1:
            data_1.append(this_data)
        elif data_ori[i]['a_b'] == 0.5:
            data_05.append(this_data)
        elif data_ori[i]['a_b'] == 0.05:
            data_005.append(this_data)
        if save == True:
            print('energy', cal_energy(elements_list))
    data_U = {'1':data_1, '05':data_05, '005':data_005}
    if save:
        print("Energy data has been saved!")
        with open("Data/data_U.pkl", "wb") as f:
            pickle.dump(data_U, f)
    return data_U

def posterior_energy(energy_list_array, DOFs_array, slope):
    if len(energy_list_array)<3:
        raise AssertionError("The value of energy should be greater than three!")
    elif len(energy_list_array)!= len(DOFs_array):
        raise AssertionError("The number of energy values should be equal to the number of DOFs!")

    Bh = abs(slope)
    i = 0
    U_list = []
    while i+3 < len(energy_list_array):
        U0, U1, U2 = energy_list_array[i:i+3]
        h0, h1, h2 = 1/np.sqrt(DOFs_array[i:i+3])
        Q = np.log((h0/h1))/np.log((h1/h2))
        lhs = lambda U: np.log(abs((U-U0)/(U-U1)))/np.log(abs((U-U1)/(U-U2)))
        initial_guess = np.mean(energy_list_array[1:])
        result = minimize(lhs, initial_guess)
        U_list.append(result.x)
        i+=1
    return np.mean(U_list)

if __name__=='__main__':
    with open("Data/data.pkl", "rb") as f:
        data_ori = pickle.load(f)
    index = -2
    print(index)
    E = 200e3
    nu = 0.3
    D = E / (1 - nu**2)* np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
        ])
    elements_list = data_ori[index]['elements_list']
    data_U = save_energy(data_ori, False)
    for key in data_U.keys():
        U_1 = []
        DOF_1 = []
        for data in data_U[key]:
            U_1.append(data['energy'])
            DOF_1.append(data['DOF'])
        U_1_FEM = posterior_energy(U_1, DOF_1, 1)
        print("The strain energy for a/b={} is {}".format(data_U[key][0]['a_b'], U_1_FEM))