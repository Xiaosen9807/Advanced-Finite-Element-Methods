import numpy as np
import seaborn as sns
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
plt.style.use('default') # 使用默认样式
from tools_2D import *
from shape_fns import *
from Elements import *
from Mesh import create_mesh, Boundary
import pickle
from scipy.interpolate import griddata


def draw_exact(elements_list, dir ='x',type='stress',  show=True):
    # Initialize global_min and global_max
    global_min = float('inf')
    global_max = float('-inf')
    refine = 3
    a_b =elements_list[0].a_b
    # Loop over each element in the elements_list
    for test_element in elements_list:
        
        # Evaluate the exact_fn at each mapping point of the test_element
        values = []
        test_inputs = test_element.sample_points(refine)
        for xy in test_element.mapping(test_inputs):
            value = output(exact_fn(xy[0], xy[1], a_b, type), dir, type)
            values.append(value)
        
        # Find the minimum and maximum value for the current element
        local_min = np.min(values)
        local_max = np.max(values)
        
        # Update global_min and global_max
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max

    if type == 'stress':
        print('Direction:', dir)
        print('Maximum stress value:', global_max)
        print('SCF in model:', global_max/50)
        print('SCF in theory:', 1+2*1/a_b)

    for i in range(len(elements_list)):
        test_element = elements_list[i]
        test_inputs = test_element.sample_points(refine)
        test_mapping = test_element.mapping(test_inputs)
        test_output = [output(exact_fn(xy[0], xy[1], a_b, type), dir, type) for xy in test_mapping]
        test_x, test_y, test_z = grid_to_mat(test_mapping, test_output)
        # plt.scatter(test_mapping[:, 0], test_mapping[:, 1], s=1, c=test_output)
        plt.imshow(test_z, extent=(test_mapping[:, 0].min(),
                                            test_mapping[:, 0].max(),
                                            test_mapping[:, 1].min(),
                                            test_mapping[:, 1].max()),
                                            origin='lower', aspect='auto',
                                            interpolation='bilinear',cmap='jet',
                                            vmin=global_min, vmax=global_max)
        # 绘制元素的边界
        vertices = test_element.vertices
        vertices = np.vstack([vertices, vertices[0]])  # 将第一个顶点再次添加到数组的末尾，以便封闭形状
        vertices_x, vertices_y = zip(*vertices)  # 解压顶点坐标
        plt.plot(vertices_x, vertices_y,  color='white', linewidth=0.7)  # 使用黑色线绘制边界，并使用小圆点表示顶点s
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    # Display the color bar
    plt.colorbar()
    dir_str = "{ %s }" % dir
 
    if type == 'strain':
        type_str = '\\epsilon'
    elif type == 'stress':
        type_str = '\\sigma'

    plt.title(rf"Exact solution: ${type_str}_{dir_str}$")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    with open("Data/data.pkl", "rb") as f:
        data_ori = pickle.load(f)

    dirs = ['x']
    dir ='xy'
    type = 'stress'
    if type == 'disp':
        type_str = 'U'
    elif type == 'strain':
        type_str = '\\epsilon'
    elif type == 'stress':
        type_str = '\\sigma'
    dir_str = "{ %s }" % dir
    iii = -1
    refine = 3

    for dir in dirs:
        for type in types:
            for iii in range(indexs):
                elements_list = data_ori[iii]['elements_list']
                mesh_size = data_ori[iii]['mesh_size']
                mesh_shape = data_ori[iii]['mesh_shape']
                if mesh_size != 2:
                    print("mesh_size should be 2, not {}".format(mesh_size))
                    pass
                    # raise ValueError ("mesh_size should be 2")
                else:
                    a_b =data_ori[iii]['a_b'] 
                    data_keys = list(data_ori[iii].keys())[:3]
                    for key in data_keys:
                        print(key, data_ori[iii][key])

                    # draw(elements_list, dir, type)
                    # draw(elements_list, 'y', type)
                    draw_exact(elements_list, dir, type, show=False)
