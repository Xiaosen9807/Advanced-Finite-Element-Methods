from typing import Any
import numpy as np
import sympy as sp
from scipy.special import roots_legendre
import scipy.linalg as la
import matplotlib.pyplot as plt
plt.style.use('default')
import copy
from shape_functions import *
from tools_1D import *
import time

def creat_mesh(interfaces = [0, .25, .5, .75, 1], num_elems = 3):
    N=3
    num_elems_per_segment = num_elems
    num_elems = len(interfaces) * num_elems_per_segment
    # 初始化网格数组
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

def FEM_1D(shape_class = linear, p = 3, interfaces = [ 0, .25, 5, .75, 1], num_elems = 4, domain = (0, 1),rhs_func = rhs_fn(a=50, xb=0.8), exact_func=exact_fn(0.5,0.8), BCs = (0, 0), verbose = False):
    start_time = time.time()
    N=3
    mesh = np.linspace(domain[0], domain[1], num_elems+1)
    
    ori_phi_phip = {'phis': [], 'phips': [], 'r_rs': []}
    for elem in range(num_elems):
        scale = [mesh[elem], mesh[elem+1]]
        phis, phips = shape_class(scale, p)
        
        this_r_r = [r_r(scale), r_r(scale)]
        ori_phi_phip['phis'].append(phis)
        ori_phi_phip['phips'].append(phips)
        ori_phi_phip['r_rs'].append(this_r_r)


    linear_phi_phip = {'phis': [], 'phips': [], 'r_rs': []}  # Linear

    for elem in range(num_elems):
        linear_phis = []
        linear_phips = []
        linear_r_rs = []
        for idx in range(len(ori_phi_phip['phis'][elem])):
            if ori_phi_phip['phis'][elem][idx].p < 2:
                phi = ori_phi_phip['phis'][elem][idx]
                phip = ori_phi_phip['phips'][elem][idx]
                this_r_r = ori_phi_phip['r_rs'][elem][idx]
                linear_phi_phip['phis'].append(phi)
                linear_phi_phip['phips'].append(phip)
                linear_phi_phip['r_rs'].append(this_r_r)
                linear_phis.append(phi)
                linear_phips.append(phip)
                linear_r_rs.append(this_r_r)
        linear_K_sub = np.zeros((len(linear_phips), len(linear_phips)))
        for indx, x in np.ndenumerate(linear_K_sub):
            B2 = 12
            K_value = G_integrate(
                # mul(linear_phips[indx[0]], linear_phips[indx[-1]]), N=N, scale=linear_phips[indx[0]].scale)
                mul(linear_r_rs[indx[0]], linear_phips[indx[0]], linear_phips[indx[-1]]), N=N, scale=linear_phips[indx[0]].scale)
            linear_K_sub[indx] = K_value
            # print(K_value)
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
            
    linear_num = len(F)
    # print(linear_num)
    # K[0, 1:] = 0
    K[linear_num-1, :linear_num-1] = 0
    # F[0] = BCs[0]* K[0, 0] # -= or = ??
    F[linear_num-1] = BCs[-1] * K[linear_num-1, linear_num-1]


    U = la.solve(K, F)
    if verbose:
        print("K:", K)
        print("F:", F)
        print("U:", U)
    phi_phip = {'phis': [], 'phips': []}
    phi_phip['phis'] = joint_funcs(linear_phi_phip['phis']) 
    phi_phip['phips'] = joint_funcs(linear_phi_phip['phips'])
    u_list = []
    for i in range(len(phi_phip['phis'])):
        u_list.append(mul(U[i], phi_phip['phis'][i]))
    uh = plus(u_list)
    if verbose == True:
        duration = time.time() - start_time

        print("Duration: {:.2f}s".format(duration))
        print(f"Shape class: {shape_class.__name__}, Number of elements: {num_elems}, Polynomial order:{p},  Domain: {domain}, Boundary conditions: {BCs}")
        x_data = np.linspace(domain[0], domain[1], 101)
        A_z_exact = exact_func(x_data)
        A_z_FEM = uh(x_data)
        error_FEM = cal_error(A_z_exact=A_z_exact, A_z_pred=A_z_FEM)
        print("Error: {:.2f}%".format(error_FEM))

        plt.plot(x_data, A_z_exact, label='Analytical solution')
        plt.plot(x_data, A_z_FEM, label='FEM solution {} elements'.format(num_elems))
        
        x_position = 0.85  # Start of the x-axis
        y_position = max(A_z_exact)  # Position text at the minimum of the exact solution for visibility

        # Add text with a black background
        plt.text(x_position, y_position, "Duration: {:.2f}s".format(duration), fontsize=14, color='white', 
                 bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))
        plt.title('Comparison between FEM Solution and Exact Solution', fontsize=16)

        plt.grid(True)
        plt.legend()
        plt.show()
    eigenvalues = np.linalg.eigvals(K)
    cont_K = max(eigenvalues)/min(eigenvalues)
    
    return U, phi_phip, uh, cont_K

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
        U_energy+=G_integrate(mul(plus(u_prime_list), plus(u_prime_list)),N=9, scale=scale)
    # scale = [min(mesh), max(mesh)]
    # print(scale)
    # U_energy+=G_integrate(mul(plus(u_prime_list), plus(u_prime_list)),N=6, scale=scale)
    return U_energy/2


if __name__=="__main__":
    verbose = True
    num_elems = 5
    domain = (0, 1)
    p = 1
    mesh = np.linspace(domain[0], domain[1], num_elems+1)
    a = .5*1
    xb = 0.8
    if a == 50:
        U_init = 1.585854059271320
    elif a == 0.5:
        U_init = 0.03559183822564316
    exact_func = exact_fn(a = a, xb=xb)
    rhs_func = rhs_fn(a=a, xb=xb)
    BCs = (exact_func(domain[0]), exact_func(domain[-1]))
    # BCs = (71, )
    start_time = time.time()
    U_l_test, phi_phip_l_test, uh_l_test, cont_K_l_test = FEM_1D(shape_class = linear,p=p, num_elems = num_elems, domain = domain,rhs_func = rhs_func,exact_func=exact_func, BCs = BCs, verbose = verbose)
    


    print(BCs)
    # cal_energy(U_l_test, phi_phip_l_test)