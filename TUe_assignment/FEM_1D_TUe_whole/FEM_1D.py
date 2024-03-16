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
from consts import consts, interfaces_global




def creat_mesh(interfaces = [0, .25, .5, .75, 1], num_elems_per_segment = 3):
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
def find_element_region(mesh_elem):
    interfaces = interfaces_global
    # mesh_elem 是一个包含两个元素的列表，表示当前元素的起始和结束位置
    for i in range(len(interfaces) - 1):
        # 检查当前元素是否位于第 i 个区间内
        if interfaces[i] <= mesh_elem[0] and mesh_elem[1] <= interfaces[i + 1]:
            return i  # 返回当前元素所在的区间索引
    return None  # 如果没有找到符合条件的区间，则返回 None

def define_interfaces(domain):
    
    # 确保domain的起始和结束值都包含在interfaces中
    start, end = domain
    updated_interfaces = [start] + [i for i in interfaces_global if start < i < end] + [end]
    
    return tuple(updated_interfaces)
"""
def find_region(value):
    r1 = consts["r1"]
    r2 = consts["r2"]
    r3 = consts["r3"]
    r4 = consts["r4"]

    # 判断值落在哪个区间
    if 0 <= value < r1: 
        return int(0)
    elif r1 <= value < r2:
        return int(1)
    elif r2 <= value < r3:
        return int(2)
    elif r3 <= value <= r4:
        return int(3)
    else:
        raise ValueError("Value {} is less than 0 or larger than r4".format(value))
"""
def find_interface_indices(mesh, interfaces):
    # 将interfaces转换为numpy数组，以便使用searchsorted
    interfaces_array = np.array(interfaces)
    
    # 使用searchsorted找到interfaces中每个元素在mesh中的索引
    indices = np.searchsorted(mesh, interfaces_array, side='left')
    
    # 如果界面值正好是mesh中的值，searchsorted会返回正确的索引
    # 如果界面值在mesh的两个值之间，searchsorted会返回较大值的索引
    # 在这种情况下，可能需要根据具体情况调整索引
    
    return indices
def FEM_1D(shape_class = linear, p = 3, interfaces = [ 0, .0053, .0063, .0101, 0.0111], num_elems_region = 1, domain = (0, 1),rhs_func = rhs_fn(a=50, xb=0.8), exact_func=exact_fn(), BCs = (0, 0), verbose = False):
    start_time = time.time()
    N=3
    mesh = creat_mesh(interfaces, num_elems_region)
    print(mesh)
    # mesh = np.linspace(domain[0], domain[1], num_elems+1)
    num_elems = len(mesh) - 1 
    BCs = list(BCs)
    # BCs[-1] = 0

    
    ori_phi_phip = {'phis': [], 'phips': [], 'r_rs': []}
    for elem in range(num_elems):
        scale = [mesh[elem], mesh[elem+1]]
        interface_index = find_element_region(scale)
        
        # exit()
        
        phis, phips = shape_class(scale, p)
        
        this_r_r = [r_r(scale), r_r(scale)]
        ori_phi_phip['phis'].append(phis)
        ori_phi_phip['phips'].append(phips)
        ori_phi_phip['r_rs'].append(this_r_r)


    linear_phi_phip = {'phis': [], 'phips': [], 'r_rs': []}  # Linear

    for elem in range(num_elems):
        scale = [mesh[elem], mesh[elem+1]]

        region_index = find_element_region(scale)
        # print(region_index)
        Param = consts["Params"][region_index]
        mu = Param["mu"]
        Jz = Param["Jz"]
        rhs_func.mu = mu
        rhs_func.Jz = Jz
        print("mu, Jz", mu, Jz)

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
            # print('linear_phips[indx[0]].scale', linear_phips[indx[0]].scale)
            # print('linear_phips[indx[0]](x)', linear_phips[indx[0]](0.0103))
            K_value = G_integrate(
                # mul(linear_phips[indx[0]], linear_phips[indx[-1]]), N=N, scale=linear_phips[indx[0]].scale)
                
                mul(linear_r_rs[indx[0]], linear_phips[indx[0]], linear_phips[indx[-1]]), N=N, scale=linear_phips[indx[0]].scale) / mu

            linear_K_sub[indx] = K_value
            # print(K_value)
            if abs(linear_K_sub[indx]) < 1e-10:
                linear_K_sub[indx] = 0
        # print('K_sub', linear_K_sub)
        linear_F_sub = np.zeros(len(linear_K_sub))
        for indx in range(len(linear_F_sub)):
            linear_F_sub[indx] = -G_integrate(
                mul(linear_r_rs[indx], rhs_func, linear_phis[indx]), N=N, scale=linear_phis[indx].scale)
                # mul(rhs_func, linear_phis[indx]), N=N, scale=linear_phis[indx].scale)
            # print(phis[indx](mesh[i]))
        # print(linear_F_sub)
        if elem == 0:
            K = linear_K_sub
            F = linear_F_sub
        else:
            K = assemble(K, linear_K_sub)
            F = assemble(F, linear_F_sub)
            
    # print(linear_num)
    # K[0, 1:] = 0
    # K[0, 0] = 1
    # F[0] = BCs[0]* K[0, 0] # -= or = ??

    # K[-1, :-1] = 0
    # K[-1, -1] = 1
    # F[-1] = BCs[-1] * K[-1,-1]
    # F[-1] = -198
    # F[0] = 0
    interface_index = find_interface_indices(mesh, interfaces)
    print("interface_index", interface_index)
    # print(interface_index[-2])

    bnd = -1
    K[bnd, :] = 0  # 将第bnd行的所有元素设置为0
    # K[:, bnd] = 0  # 将第bnd列的所有元素设置为0
    K[bnd, bnd] = 1  # 将对角线元素设置为1
    F[bnd] = BCs[-1] * K[bnd, bnd]
    # F[interface_index[-2]] *= 21
    # F[interface_index[-3]] *= 21
    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        # 打印矩阵
        print("K:\n", K)
        print("F:", F)

    U = la.solve(K, F)
    
    if verbose:
        print("U:", U)
        print("mesh:", mesh)
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
        # print(A_z_exact[-1], A_z_FEM[-1])
        error_FEM = cal_error(A_z_exact=A_z_exact, A_z_pred=A_z_FEM)
        print("Error: {:.2f}%".format(error_FEM))

        plt.plot(x_data, A_z_exact, label='Analytical solution')
        plt.plot(x_data, A_z_FEM, label='FEM solution {} elements'.format(num_elems_region))
        # plt.scatter(mesh, U, label='FEM solution {} elements'.format(num_elems))
        
        x_position = 0.85  # Start of the x-axis
        y_position = max(A_z_exact)  # Position text at the minimum of the exact solution for visibility

        # Add text with a black background
        plt.text(x_position, y_position, "Duration: {:.2f}s".format(duration), fontsize=14, color='white', 
                 bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.5'))
        plt.title('Comparison between FEM Solution and Exact Solution', fontsize=16)
        plt.legend()
        plt.axvline(x=interfaces_global[1], color='r', linestyle='--', label='r1')  
        plt.axvline(x=interfaces_global[2], color='g', linestyle='--', label='r2')  
        plt.axvline(x=interfaces_global[3], color='b', linestyle='--', label='r3')  
        plt.axvline(x=interfaces_global[4], color='y', linestyle='--', label='r4')  
        # plt.xlim(0.01, 0.01111)
        # plt.ylim(-1, 1)
        plt.grid(True)
        plt.show()
    eigenvalues = np.linalg.eigvals(K)
    cont_K = max(eigenvalues) / min(eigenvalues)
    
    return U, phi_phip, uh, cont_K

if __name__=="__main__":
    verbose = True
    num_elems_region = 2
    domain = (0.00, 0.0111)
    # domain = (0.0101, 0.0111)
    # domain = (0, 0.0111)
    interfaces = define_interfaces(domain)
    p = 1
    mesh = creat_mesh(interfaces, num_elems_region)
    a = .5*1
    xb = 0.8
    if a == 50:
        U_init = 1.585854059271320
    elif a == 0.5:
        U_init = 0.03559183822564316
    exact_func = exact_fn()
    rhs_func = rhs_fn(a=a, xb=xb)
    BCs = (exact_func(domain[0]), exact_func(domain[-1]))
    # BCs = (71, )
    start_time = time.time()
    U_l_test, phi_phip_l_test, uh_l_test, cont_K_l_test = FEM_1D(shape_class = linear,p=p, num_elems_region = num_elems_region, interfaces=interfaces, domain = domain,rhs_func = rhs_func,exact_func=exact_func, BCs = BCs, verbose = verbose)
    print(phi_phip_l_test['phis'])
    out_func = plus(phi_phip_l_test['phis'][0], phi_phip_l_test['phis'][1])
    # plt.plot(phi_phip_l_test['phis'][0](mesh))
    # plt.plot(phi_phip_l_test['phis'][1](mesh))
    # plt.plot(out_func(mesh))
    plt.plot(phi_phip_l_test['phips'][0](mesh))
    print(phi_phip_l_test['phips'][0].check_name)
    plt.plot(phi_phip_l_test['phips'][1](mesh))
    # plt.show()


    print(BCs)
    # cal_energy(U_l_test, phi_phip_l_test)