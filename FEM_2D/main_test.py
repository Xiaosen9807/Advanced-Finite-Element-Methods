import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Elements import *
from Energy import *
from Mesh import create_mesh, Boundary
from exact_solu import draw_exact
import pickle
import sys
import os

# 获取当前脚本的路径
script_path = sys.argv[0]

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(script_path))

# 更改工作目录
os.chdir(script_dir)

def FEM(a_b, mesh_size, mesh_shape, GPN=2, show=False):
    Load_x = 28.9  # N/mm
    Load_y = 0  # N/mm
    A = 40  # mm^2
    nodes_coord,  element_nodes = create_mesh(a_b, mesh_shape, mesh_size)
    # with open("Coords/saved_data.pkl", "rb") as file:
    #    nodes_coord,  element_nodes = pickle.load(file)
    nodes_list = Boundary(nodes_coord, a_b)
    element_list = []
    if mesh_shape == 0:
        element_nodes = element_nodes.reshape(-1, 3)
    elif mesh_shape == 1:
        element_nodes = element_nodes.reshape(-1, 4)

    for ele_lst in element_nodes:
        this_nodes = [
            node for id in ele_lst for node in nodes_list if node.id == id]
        # try:
        #     elem = Q4(this_nodes, GPN=GPN)
        # except Exception as e:
        #     print("Error:", e)
        #     elem = T3(this_nodes, GPN=GPN)
        try:
            elem = Q4(this_nodes, GPN=GPN)
        except:
            elem = T3(this_nodes, GPN=GPN)
        elem.a_b = a_b
        element_list.append(elem)
    DOFs = 2*len(nodes_list)
    glo_K = np.zeros((DOFs, DOFs))
    glo_F = np.zeros(DOFs)

    for elem in element_list:  # Assemble Force vector
        loc_F = elem.F
        for i, node_i in enumerate(elem.nodes):
            global_dof = 2 * node_i.id
            # print(loc_F[2*i])
            if abs(node_i.xy[0]-40) < 1e-3:
                glo_F[global_dof] += Load_x * loc_F[2*i]
                # glo_F[global_dof] += Load_x * 1 
                glo_F[global_dof + 1] += Load_y * loc_F[2*i+1]

    for elem in element_list:  # Assemble Stiffness matrix
        loc_K = elem.K
        # print(loc_K)
        for i, node_i in enumerate(elem.nodes):
            for j, node_j in enumerate(elem.nodes):
                for dof_i in range(2):  # 每个节点的dof: 0和1
                    for dof_j in range(2):
                        # 计算全局dof的位置
                        global_dof_i = 2 * node_i.id + dof_i
                        global_dof_j = 2 * node_j.id + dof_j

                        # 组装到全局矩阵
                        glo_K[global_dof_i][global_dof_j] += loc_K[2 * i + dof_i][2*j + dof_j]
    for elem in element_list:  # Boundary condition

        for i, node_i in enumerate(elem.nodes):
            for dof_i in range(2):  # 每个节点的dof: 0和1 (x和y方向)
                global_dof_i = 2 * node_i.id + dof_i

                # 检查迪里希莱边界条件
                if node_i.BC[dof_i] == 1:
                    # 修改刚度矩阵和载荷向量
                    # print(node_i.id, global_dof_i, node_i.BC)

                    glo_K[global_dof_i, :] = 0
                    # glo_K[:, global_dof_i] = 0
                    glo_K[global_dof_i, global_dof_i] = 1e15  # 大数约束
                    glo_F[global_dof_i] = 0

    # np.set_printoptions(precision=2, suppress=True)
    # glo_K[np.abs(glo_K) < 1e-9] = 0
    U = np.linalg.solve(glo_K, glo_F)
    # print(U)
    for id in range(len(nodes_list)):
        displacement = np.array([U[id*2], U[id*2+1]])
        nodes_list[id].value = displacement

    if show == True:

    #     x_coords = [node.xy[0] for node in nodes_list]
    #     y_coords = [node.xy[1] for node in nodes_list]
    #     temperatures = [np.linalg.norm(node.value) for node in nodes_list]

    #     # 创建散点图
    #     plt.scatter(x_coords, y_coords, c=temperatures, cmap='inferno')
    #     plt.colorbar(label='Displacement in megnitude')
    #     plt.title('Displacements Distribution')
    #     for (x, y), node in zip(nodes_coord, nodes_list):
    #         # 在指定的坐标处显示文本
    #         plt.text(x, y, node.id)

        #     plt.show()
        x_coords = [node.xy[0] for node in nodes_list]
        y_coords = [node.xy[1] for node in nodes_list]

        temperatures = [np.linalg.norm(node.value) for node in nodes_list]

        tri = []
        for c in element_nodes:
            tri.append([c[0], c[1], c[2]])
            try:
                tri.append([c[0], c[2], c[3]])
            except:
                pass

        # 4. 使用 tricontourf 绘制图形
        plt.tricontourf(x_coords, y_coords, temperatures, triangles=tri,  levels=15, cmap=plt.cm.jet)
        plt.colorbar(label='Displacement in magnitude')
        plt.title('Displacements Distribution')
        plt.close()
        # plt.show()
    return U, nodes_coord, copy.deepcopy(element_list)


def draw(elements_list, dir='xy', type='disp', show = True, save=False):
    global_min = min([np.min([output(test_element(xy[0], xy[1], type), dir, type)
                        for xy in test_element.sample_points(refine)]) for test_element in elements_list])

    global_max = max([np.max([output(test_element(xy[0], xy[1], type), dir, type)
                        for xy in test_element.sample_points(refine)]) for test_element in elements_list])

    for test_element in elements_list:
        test_inputs = test_element.sample_points(refine)
        test_mapping = test_element.mapping(test_inputs)
        test_output = [output(test_element(xy[0], xy[1], type), dir, type)
                        for xy in test_inputs]
        test_x, test_y, test_z = grid_to_mat(test_mapping, test_output)
        # plt.scatter(test_mapping[:, 0], test_mapping[:, 1], s=1, c=test_output)
        plt.imshow(test_z, extent=(test_mapping[:, 0].min(),
                                    test_mapping[:, 0].max(),
                                    test_mapping[:, 1].min(),
                                    test_mapping[:, 1].max()),
                    origin='lower', aspect='auto',
                    interpolation='none',cmap='jet',
                    vmin=global_min, vmax=global_max)
        # 绘制元素的边界
        vertices = test_element.vertices
        # 将第一个顶点再次添加到数组的末尾，以便封闭形状
        vertices = np.vstack([vertices, vertices[0]])
        vertices_x, vertices_y = zip(*vertices)  # 解压顶点坐标
        plt.plot(vertices_x, vertices_y,  color='white',
                linewidth=0.7)  # 使用黑色线绘制边界，并使用小圆点表示顶点

    plt.xlim(0, 40)
    plt.ylim(0, 40)
    # Display the color bar
    cbar = plt.colorbar()
    ticks = np.linspace(global_min, global_max, num=5)  # 例如5个ticks
    cbar.set_ticks(ticks)
    if type == 'disp':
        type_str = 'U'
    elif type == 'strain':
        type_str = '\\epsilon'
    elif type == 'stress':
        type_str = '\\sigma'
    dir_str = "{ %s }" % dir
    plt.title(rf"${type_str}_{dir_str}$")
    # plt.title(rf"${type_str}$")
    if save:
        pass
        # plt.savefig('images/Q2_1/Q1_{}_{}_{}.png'.format(a_b, mesh_size, test_element.shape))
    if show:
        plt.show()
    plt.close()


if __name__=='__main__':
    experi = False
    show = True
    save = True
    GPN = 4
    refine = 3
    if experi == True:
        a_b_lst = [1, 0.5, 0.05]
        mesh_size_lst = [8, 4, 2]
        mesh_shape_lst = [0, 1]

    else:
        a_b_lst = [0.5]
        mesh_size_lst = [2]
        mesh_shape_lst = [1]

    data_dict = []
    for a_b in a_b_lst:
        for mesh_size in mesh_size_lst:
            for mesh_shape in mesh_shape_lst:
                U, nodes_list, elements_list = FEM(a_b, mesh_size, mesh_shape, GPN, show)
                shape = 'Q4' if mesh_shape == 1 else 'T3'
                this_data_dict = {'a_b': a_b, 'mesh_size': mesh_size,
                                  'mesh_shape': shape,
                                  "DOF": len(nodes_list)*2,
                                  "U": U, "nodes_coord": nodes_list,
                                  "elements_list": elements_list}
                data_dict.append(this_data_dict)
    #          break
    #       break
    #    break

    # # 打开一个文件并保T3存字典
    # if save == True:
    #     with open("Data/data.pkl", "wb") as f:
    #         pickle.dump(data_dict, f)
    # Determine global min and max values
    dir = 'von'
    type = 'stress'
    data_ori = data_dict
    
    for i in range(len(data_ori)):
        elements_list = data_ori[i]['elements_list']
        a_b = data_ori[i]['a_b']
        mesh_size = data_ori[i]['mesh_size']
        mesh_shape = data_ori[i]['mesh_shape']
        draw(elements_list, dir, type, show = show, save=save)
        # draw(elements_list, 'y', type, show = show, save=save)
        # draw_exact(elements_list, dir, type, show=show)
