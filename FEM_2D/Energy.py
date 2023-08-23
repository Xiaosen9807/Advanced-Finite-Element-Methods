import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from shape_fns import *
from Elements import *
from tools_2D import *                
import sys
from scipy.optimize import minimize, fsolve
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
            elem_energy+=0.5 * W * U_list.T @ B.T @ D @ B @ U_list * scale #*elem.area #* J_det
        energy += elem_energy[0][0]
    return energy


def cal_energy_exact(elements_list,a_b, GPN = 2):
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
        xys = elem.mapping(points)
        loop = 0
        scale = 1 if elem.shape=="triangle" else 1
        for g in range(len(Ws)):
            xy = xys[g]
            W = Ws[g]
            dN = elem.gradshape(xy[0], xy[0])
            J = jacobian(elem.vertices, dN)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)
            
            stress_list = exact_fn(xy[0], xy[1], a_b)
            strain_list = np.linalg.inv(D) @ stress_list
            # this_energy = 0.5 * W * stress_list.T @ np.linalg.inv(D) @ stress_list * scale
            # print(strain_list, stress_list)

            this_energy = 0.5 * W * strain_list.T @ D @ strain_list * scale * elem.area 
            # print(this_energy)
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
            strain_list = elem(xy[0], xy[1], 'strain')
            this_energy = 0.5 * W * strain_list.T @ D @ strain_list * scale
            elem_energy += this_energy 
            loop+=1
        energy+=elem_energy
    return energy[0][0]

def save_energy(data_ori, save=True):
    data_1 = []
    data_05 = []
    data_005 = []
    for i in range(len(data_ori)):
        elements_list = data_ori[i]['elements_list']
        a_b = data_ori[i]['a_b']
        this_data = {}
        data_keys = list(data_ori[i].keys())[:4]
        for key in data_keys:
            if save:
                print(key, data_ori[i][key])
            this_data[key] = data_ori[i][key]
        this_data['E_FEM'] = cal_energy(elements_list)
        this_data['E_exa'] = cal_energy_exact(elements_list, a_b)
        if data_ori[i]['a_b'] == 1:
            data_1.append(this_data)
        elif data_ori[i]['a_b'] == 0.5:
            data_05.append(this_data)
        elif data_ori[i]['a_b'] == 0.05:
            data_005.append(this_data)
        if save == True:
            print('E_FEM', this_data['E_FEM'] )
            print('E_exa', this_data['E_exa'] )
    data_U = {'1':data_1, '0.5':data_05, '0.05':data_005}
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
    def equation(U, U0, U1, U2, Q):
        return (((U-U0)/(U-U1) - ((U-U1)/(U-U2))**Q ))**2
        return (np.log(np.abs((U-U0)/(U-U1))) /( Q * np.log(np.abs((U-U1)/(U-U2)))))**2

    Bh = abs(slope)
    i = 0
    U_list = []
    print("Energy", energy_list_array)
    while i+3 <= len(energy_list_array):
        U0, U1, U2 = energy_list_array[i:i+3]
        h0, h1, h2 = 1/np.sqrt(DOFs_array[i:i+3])
        # print(h0, h1, h2)
        # h0, h1, h2  = [0.008, 0.004, 0.002]
        N0, N1, N2 = DOFs_array[i:i+3]
        Q = np.log((h0/h1))/np.log((h1/h2))
        # Q = np.log((N1/N0))/np.log((N2/N1))
        initial_guess = np.mean(energy_list_array)
        # 使用 minimize
        lower_bound = min(energy_list_array[i:i+3])
        upper_bound = max(energy_list_array[i:i+3])

        bounds = [(lower_bound*0.5,  upper_bound*2.)]

        U_solution = minimize(equation, initial_guess, args=(U0, U1, U2, Q), bounds=bounds).x
        # U_solution =fsolve(equation, initial_guess, args=(U0, U1, U2, Q)) 

        U_list.append(U_solution )
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
    U_post = {}
    for key in data_U.keys():
        print('\n')
        print(key)
        U_T = []
        U_Q = []
        U_2 = []
        DOF_Q = []
        DOF_T = []
        DOF_2 = []
        mesh_size_T = []
        mesh_size_Q = []
        for data in data_U[key]:
            if data['mesh_shape'] == 'T3':
                U_T.append(data['E_FEM'])
                DOF_T.append(data['DOF'])
                mesh_size_T.append(data['mesh_size'])
            elif data['mesh_shape'] == 'Q4':
                U_Q.append(data['E_FEM'])
                DOF_Q.append(data['DOF'])
                mesh_size_Q.append(data['mesh_size'])
            else:
                raise ValueError("Unknown mesh")
            U_2.append(data['E_FEM'])
            DOF_2.append(data['DOF'])
        print('U_T', U_T)
        print('U_Q', U_Q)
        # print('U_2', U_2)
        U_T_FEM = posterior_energy(U_T, DOF_T, 1)
        U_Q_FEM = posterior_energy(U_Q, DOF_Q, 2)
        U_2_exa = posterior_energy(U_2, DOF_2, 1)
        U_post[key] = {'T3':{'U':U_T_FEM, 'U_exa':U_2_exa, 'U_list':U_T, 'DOF':DOF_T,'mesh_size':mesh_size_T}, 'Q4':{'U':U_Q_FEM,'U_exa':U_2_exa,  'U_list':U_Q, 'DOF':DOF_Q,'mesh_size':mesh_size_Q}}
        # U_2_exa = np.mean(U_2)

        print("The strain energy in FEM for a/b={} within T3 is {}".format(data_U[key][0]['a_b'], U_T_FEM))
        print("The DOF or a/b={} within T3 is {}".format(data_U[key][0]['a_b'], DOF_T))
        print("The strain energy in FEM for a/b={} within Q4 is {}".format(data_U[key][0]['a_b'], U_Q_FEM))
        print("The DOF or a/b={} within Q4 is {}".format(data_U[key][0]['a_b'], DOF_Q))
        print("\nThe energy or a/b={} within total is {}".format(data_U[key][0]['a_b'], U_2_exa))
        # print("The strain energy in exact for a/b={} is {}".format(data_U[key][0]['a_b'], U_2_exa))
        # print("The strain energy in exact for a/b={} is {}".format(data_U[key][0]['a_b'], U_2_exa))
    with open("Data/data_U_post.pkl", "wb") as f:
        pickle.dump(U_post, f)
