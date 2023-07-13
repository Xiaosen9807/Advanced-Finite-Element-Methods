from typing import Any
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
import matplotlib.pyplot as plt
plt.style.use('default')
import copy
from shape_functions import *
from tools import *

def h_FEM(shape_class = linear, num_elems = 3,p=3, domain = (0, 1),rhs_func = rhs_fn(a=50, xb=0.8), exact_func=exact_fn(0.5,0.8), BCs = (0, 0), verbose = False):
    N=6
    mesh = np.linspace(domain[0], domain[1], num_elems+1)
    ori_phi_phip = {'phis': [], 'phips': []}
    for elem in range(num_elems):
        scale = [mesh[elem], mesh[elem+1]]
        phis, phips = shape_class(scale, p)
        ori_phi_phip['phis'].append(phis)
        ori_phi_phip['phips'].append(phips)


    linear_phi_phip = {'phis': [], 'phips': []}  # Linear
    for elem in range(num_elems):
        linear_phis = []
        linear_phips = []
        for idx in range(len(ori_phi_phip['phis'][elem])):
            phi = ori_phi_phip['phis'][elem][idx]
            phip = ori_phi_phip['phips'][elem][idx]
            linear_phi_phip['phis'].append(phi)
            linear_phi_phip['phips'].append(phip)
            linear_phis.append(phi)
            linear_phips.append(phip)
        linear_K_sub = np.zeros((len(linear_phips), len(linear_phips)))
        for indx, x in np.ndenumerate(linear_K_sub):
            linear_K_sub[indx] = G_integrate(
                mul(linear_phips[indx[0]], linear_phips[indx[-1]]), N=6, scale=linear_phips[indx[0]].scale)
            if abs(linear_K_sub[indx]) < 1e-10:
                linear_K_sub[indx] = 0
        # print('K_sub', linear_K_sub)
        linear_F_sub = np.zeros(len(linear_K_sub))
        for indx in range(len(linear_F_sub)):
            linear_F_sub[indx] = G_integrate(
                mul(rhs_func, linear_phis[indx]), N=N, scale=linear_phis[indx].scale)
            # print(phis[indx](mesh[i]))
        if elem == 0:
            K = linear_K_sub
            F = linear_F_sub
        else:
            K = assemble(K, linear_K_sub)
            F = assemble(F, linear_F_sub)
            
    # Applying boundary condition
    K[0, 1:] = 0.0 
    K[-1, :-1] = 0.0
    F[0] = BCs[0]* K[0, 0] # -= or = ??
    F[-1] = BCs[-1] * K[-1, -1]

    U = -la.solve(K, F)
    phi_phip = {'phis': [], 'phips': []}
    phi_phip['phis'] = joint_funcs(linear_phi_phip['phis'])
    phi_phip['phips'] = joint_funcs(linear_phi_phip['phips']) 
    u_list = []
    for i in range(len(phi_phip['phis'])):
        u_list.append(mul(U[i], phi_phip['phis'][i]))
    uh = plus(u_list)
    if verbose == True:
        print(f"Shape class: {shape_class.__name__}, Number of elements: {num_elems}, Polynomial order:{p},  Domain: {domain}, Boundary conditions: {BCs}")
        x_data = np.linspace(domain[0], domain[1], 128)
        plt.plot(x_data, exact_func(x_data), label=' Analytical solution')
        plt.plot(x_data, uh(x_data), label='FEM solution {} elements'.format(num_elems))
        for i in range(len(phi_phip['phis'])):
            func = phi_phip['phis'][i]
            plt.plot(x_data, U[i]*func(x_data))
        plt.legend()
        plt.show()
    return U, phi_phip, uh


def FEM(shape_class = Hierarchical, p = 3, num_elems = 3, domain = (0, 1),rhs_func = rhs_fn(a=50, xb=0.8), exact_func=exact_fn(0.5,0.8), BCs = (0, 0), verbose = False):
    N=6
    mesh = np.linspace(domain[0], domain[1], num_elems+1)
    ori_phi_phip = {'phis': [], 'phips': []}
    for elem in range(num_elems):
        scale = [mesh[elem], mesh[elem+1]]
        phis, phips = shape_class(scale, p)
        ori_phi_phip['phis'].append(phis)
        ori_phi_phip['phips'].append(phips)


    linear_phi_phip = {'phis': [], 'phips': []}  # Linear
    for elem in range(num_elems):
        linear_phis = []
        linear_phips = []
        for idx in range(len(ori_phi_phip['phis'][elem])):
            if ori_phi_phip['phis'][elem][idx].p < 2:
                phi = ori_phi_phip['phis'][elem][idx]
                phip = ori_phi_phip['phips'][elem][idx]
                linear_phi_phip['phis'].append(phi)
                linear_phi_phip['phips'].append(phip)
                linear_phis.append(phi)
                linear_phips.append(phip)
        linear_K_sub = np.zeros((len(linear_phips), len(linear_phips)))
        for indx, x in np.ndenumerate(linear_K_sub):
            linear_K_sub[indx] = G_integrate(
                mul(linear_phips[indx[0]], linear_phips[indx[-1]]), N=6, scale=linear_phips[indx[0]].scale)
            if abs(linear_K_sub[indx]) < 1e-10:
                linear_K_sub[indx] = 0
        # print('K_sub', K_sub)
        linear_F_sub = np.zeros(len(linear_K_sub))
        for indx in range(len(linear_F_sub)):
            linear_F_sub[indx] = G_integrate(
                mul(rhs_func, linear_phis[indx]), N=N, scale=linear_phis[indx].scale)
            # print(phis[indx](mesh[i]))
        if elem == 0:
            K = linear_K_sub
            F = linear_F_sub
        else:
            K = assemble(K, linear_K_sub)
            F = assemble(F, linear_F_sub)
            
    # Applying boundary condition
    K[0, 1:] = 0.0 
    K[-1, :-1] = 0.0
    F[0] = BCs[0]* K[0, 0] # -= or = ??
    F[-1] = BCs[-1] * K[-1, -1]

    nonlinear_phi_phip = {'phis': [], 'phips': []}
    for order in range(2, p+1):  # Non Linear
        # print('order', order)
        for elem in range(num_elems):
            for idx in range(len(ori_phi_phip['phis'][elem])):
                if (ori_phi_phip['phis'][elem][idx].p == order) or (ori_phi_phip['phips'][elem][idx].p == order):
                    nonlinear_phi = ori_phi_phip['phis'][elem][idx]
                    nonlinear_phip = ori_phi_phip['phips'][elem][idx]
                    nonlinear_phi_phip['phis'].append(nonlinear_phi)
                    nonlinear_phi_phip['phips'].append(nonlinear_phip)
                    nonlinear_K_sub = np.zeros((2, 2))
                    # print('nonlinear_phip', nonlinear_phip.p)
                    # print(G_integrate(mul(nonlinear_phip, nonlinear_phip),N=N, scale=nonlinear_phip.scale))
                    
                    nonlinear_K_sub[-1, -1] = G_integrate(mul(nonlinear_phip, nonlinear_phip),N=N, scale=nonlinear_phip.scale)
                    nonlinear_F_sub = np.zeros(2)
                    nonlinear_F_sub[-1] = G_integrate(mul(rhs_func, nonlinear_phi), N=N, scale=nonlinear_phi.scale)

                    K = assemble(K, nonlinear_K_sub)
                    F = assemble(F, nonlinear_F_sub)
                else:
                    pass

    U = -la.solve(K, F)
    # print(F)
    phi_phip = {'phis': [], 'phips': []}
    phi_phip['phis'] = joint_funcs(linear_phi_phip['phis']) + nonlinear_phi_phip['phis']
    phi_phip['phips'] = joint_funcs(linear_phi_phip['phips']) + nonlinear_phi_phip['phips']
    u_list = []
    for i in range(len(phi_phip['phis'])):
        u_list.append(mul(U[i], phi_phip['phis'][i]))
    uh = plus(u_list)
    if verbose == True:
        print(f"Shape class: {shape_class.__name__}, Number of elements: {num_elems}, Polynomial order:{p},  Domain: {domain}, Boundary conditions: {BCs}")
        x_data = np.linspace(domain[0], domain[1], 101)
        plt.plot(x_data, exact_func(x_data), label='Analytical solution')
        plt.plot(x_data, uh(x_data), label='FEM solution {} elements'.format(num_elems))
        for i in range(len(phi_phip['phis'])):
            func = phi_phip['phis'][i]
            plt.plot(x_data, U[i]*func(x_data))
        plt.legend()
        plt.show()
    return U, phi_phip, uh



def cal_energy(U_array, phi_phip_array):
    U_energy = 0
    u_prime_list = []
    scales = []
    for i in range(len(phi_phip_array['phis'])):
        u_prime = mul(U_array[i], phi_phip_array['phips'][i])
        u_prime_list.append(u_prime)
        scales.append(u_prime.scale)
    # 首先，把嵌套的列表变为一个扁平的列表
    flat_scales = [item for sublist in scales for item in sublist]

    # 然后，把所有的值都保留到五位小数
    rounded_scales = [round(num, 5) for num in flat_scales]

    # 现在，用 set 来获取所有的唯一值
    nodes = list(set(rounded_scales))
    mesh = np.linspace(min(nodes), max(nodes), len(nodes))
    # print(mesh)
    for i in range(len(mesh)-1):
        scale = [mesh[i], mesh[i+1]]
        U_energy+=G_integrate(mul(plus(u_prime_list), plus(u_prime_list)),N=6, scale=scale)
    # scale = [min(mesh), max(mesh)]
    # print(scale)
    # U_energy+=G_integrate(mul(plus(u_prime_list), plus(u_prime_list)),N=6, scale=scale)
    return U_energy/2
