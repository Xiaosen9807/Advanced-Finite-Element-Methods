import numpy as np
import sympy as sp
from tools_2D import *
from shape_fns import *
from Elements import *
from Mesh import create_mesh, Boundary
import pickle

def FEM(a_b, mesh_size, mesh_shape, GPN = 2, show = False): 
   Load_x = 50 # N/mm
   Load_y = 0 # N/mm
   A = 40 # mm^2
   nodes_coord,  element_nodes = create_mesh(a_b, mesh_shape, mesh_size)
   nodes_list = Boundary(nodes_coord, a_b)
   element_list = []
   if mesh_shape == 0:
      element_nodes = element_nodes.reshape(-1, 3)
   elif mesh_shape == 1:
      element_nodes = element_nodes.reshape(-1, 4)


   for ele_lst in element_nodes:
      this_nodes = [node for id in ele_lst for node in nodes_list if node.id == id]
      try:
         element_list.append(Q4(this_nodes, GPN = GPN))
      except:
         element_list.append(T3(this_nodes, GPN = GPN))
   DOFs = 2*len(nodes_list)
   glo_K = np.zeros((DOFs, DOFs))
   glo_F = np.zeros(DOFs)

   for elem in element_list: # Assemble Force vector
      loc_F = elem.F
      for i, node_i in enumerate(elem.nodes):
         global_dof = 2 * node_i.id
         if abs(node_i.xy[0]-40)< 1e-3:
            glo_F[global_dof] +=  Load_x * loc_F[0]
            glo_F[global_dof + 1] += Load_y * loc_F[1] 



   for elem in element_list: # Assemble Stiffness matrix 
      loc_K = elem.K
      for i, node_i in enumerate(elem.nodes):
         for j, node_j in enumerate(elem.nodes):
               for dof_i in range(2):  # 每个节点的dof: 0和1
                  for dof_j in range(2):
                     # 计算全局dof的位置
                     global_dof_i = 2 * node_i.id + dof_i
                     global_dof_j = 2 * node_j.id + dof_j

                     # 组装到全局矩阵
                     glo_K[global_dof_i][global_dof_j] += loc_K[2*i + dof_i][2*j + dof_j]
   for elem in element_list: # Boundary condition

      for i, node_i in enumerate(elem.nodes):
         for dof_i in range(2):  # 每个节点的dof: 0和1 (x和y方向)
               global_dof_i = 2 * node_i.id + dof_i

               # 检查迪里希莱边界条件
               if node_i.BC[dof_i] == 1:
                  # 修改刚度矩阵和载荷向量
                  print(node_i.id, global_dof_i, node_i.BC)
                  glo_K[global_dof_i, :] = 0
                  glo_K[:, global_dof_i] = 0
                  glo_K[global_dof_i, global_dof_i] = 1e5  # 大数约束
                  glo_F[global_dof_i] = 0

   np.set_printoptions(precision=2, suppress=True)
   glo_K[np.abs(glo_K) < 1e-5] = 0
   U = np.linalg.solve(glo_K, glo_F)
   # print(U)
   for id in range(len(nodes_list)):
      displacement = np.array([U[id*2], U[id*2+1]])
      nodes_list[id].value = displacement

   # plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1])
   # for node in nodes_list:
   #    break
   #    print(node.id, node.xy, node.type, node.BC, node.value)
   # for elem in element_list:
   #    for node in elem.nodes:
   #       # break
   #       print(node.id, node.xy, node.type, node.BC, node.value)
  

   # 提取x, y坐标和温度值
   if show == True:

      x_coords = [node.xy[0] for node in nodes_list]
      y_coords = [node.xy[1] for node in nodes_list]
      temperatures = [node.value[1] for node in nodes_list]

      # 创建散点图
      plt.scatter(x_coords, y_coords, c=temperatures, cmap='inferno')
      plt.colorbar(label='Displacement in Y direction')
      plt.title('Displacements Distribution')
      for (x, y), node in zip(nodes_coord, nodes_list):
            # 在指定的坐标处显示文本
         plt.text(x, y, node.id) 

      plt.show()
   return U, copy.deepcopy(nodes_list), copy.deepcopy(element_list)

if __name__=='__main__':
      a_b_lst = [1, 0.5, 0.05]
      mesh_size_lst = [8, 4, 2]
      mesh_shape_lst = [0, 1]
      show = False
      GPN = 2
      refine = 3
      data_dict = []
      for a_b in a_b_lst:
         for mesh_size in mesh_size_lst:
            for mesh_shape in mesh_shape_lst:
               U, nodes_list, elements_list = FEM(a_b, mesh_size,
                                                mesh_shape, GPN, show)
               shape = 'Q4' if mesh_shape==1 else 'T3'
               this_data_dict = { 'a_b': a_b, 'mesh_size':mesh_size,
                           'mesh_shape': shape, "U": U, "nodes_list":
                           nodes_list, "elements_list": elements_list }
               data_dict.append(this_data_dict)
      #          break
      #       break
      #    break

      # # 打开一个文件并保存字典
      with open("data.pkl", "wb") as f:
         pickle.dump(data_dict, f)

      # Determine global min and max values
      dir = 'y'
      type = 'stress'
   
      rand_data = 0.2
      if dir == 'y':
         point = [rand_data, -1]
      elif dir =='x':
         point = [-1, rand_data]
      else:
         point = [-1, -1]

      for element in elements_list:
         print(element(point[0], point[1], dir, type))
         if abs(element(point[0], point[1], dir, type) - 0) < 1e-4:
            print(element)

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
      # for node in nodes_list:
      #    print(node.id, node.BC, node.value)

      # plt.imshow(output, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
      # plt.colorbar()
      # plt.title('Shape Function')
      # plt.xlabel('x')
      # plt.ylabel('y')
      # plt.show()


