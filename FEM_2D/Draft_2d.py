import pickle
import matplotlib.pyplot as plt
import numpy as np
from tools_2D import *
from shape_fns import *
from Elements import *

with open("data.pkl", "rb") as f:
    data_ori = pickle.load(f)
print(len(data_ori[-1]['nodes_list']))
def draw(elements_list, dir ='xy',type = 'disp'):
    refine = 3
    global_min = min([np.min([test_element(xy[0], xy[1], dir, type) for xy in test_element.sample_points(refine)]) for test_element in elements_list])
    global_max = max([np.max([test_element(xy[0], xy[1], dir, type) for xy in test_element.sample_points(refine)]) for test_element in elements_list])
    for test_element in elements_list:
        test_mapping = test_element.mapping(refine)
        test_inputs = test_element.sample_points(refine)
        test_output = [test_element(xy[0], xy[1],dir, type) for xy in test_inputs]
        test_x, test_y, test_z = grid_to_mat(test_mapping, test_output)
        plt.imshow(test_z, extent=(test_mapping[:, 0].min(),
                                    test_mapping[:, 0].max(),
                                    test_mapping[:, 1].min(),
                                    test_mapping[:, 1].max()),
                                    origin='lower', aspect='auto',
                                    interpolation='bilinear',
                                        vmin=global_min, vmax=global_max)
            # 绘制元素的边界
        vertices = test_element.vertices
        vertices = np.vstack([vertices, vertices[0]])  # 将第一个顶点再次添加到数组的末尾，以便封闭形状
        vertices_x, vertices_y = zip(*vertices)  # 解压顶点坐标
        plt.plot(vertices_x, vertices_y,  color='white')  # 使用黑色线绘制边界，并使用小圆点表示顶点

    plt.xlim(0, 40)
    plt.ylim(0, 40)
    # Display the color bar
    plt.colorbar()
    plt.legend()
    if type == 'disp':
        type_str = 'U'
    elif type == 'strain':
        type_str = '\\epsilon'
    elif type == 'stress':
        type_str = '\\sigma'
    dir_str = "{ %s }" % dir
    # if dir == 'xy':
    #    dir_str = '{xy}'
    # elif dir == 'von':
    #    dir_str = '{von}'
    # else:
    #    dir_str = dir
    plt.title(rf"${type_str}_{dir_str}$")
    plt.show()
draw(data_ori[-1]['elements_list'], dir = 'xy', type = 'disp')
