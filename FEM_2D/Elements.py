import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample
import sympy as sp
from shape_fns import *
from tools_2D import *                
import sys

   
def id_to_index(a, b, c, d):
    mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    i = mapping[str(a)+str(b)]
    j = mapping[str(c)+str(d)]
    return i, j

def constitutive(i, j, k, l, E, nu):
    delta = lambda x, y: 1 if x == y else 0
    term1 = (E / (2 * (1 + nu))) * (delta(i, l) * delta(j, k) + delta(i, k) * delta(j, l))
    term2 = (E * nu) / (1 - nu**2) * delta(i, j) * delta(k, l)

    return term1 + term2

def force(element, GPN=2):
    points, Ws = Gauss_points(element, GPN)
    n_nodes = element.n_nodes
    vertices = element.vertices
    E = element.E
    nu = element.nu
    F = np.zeros(n_nodes*2)
    for i in range(n_nodes):
        for g in range(len(Ws)):
            point = points[g]
            W = Ws[g]
            NP = []
            for node_id in range(n_nodes):
                NP.append([element.phipxs[node_id](point[0], point[1]),
                            element.phipys[node_id](point[0], point[1])])
            NP_net = np.array(NP)
            # print('NP', NP.T)
            J = jacobian(vertices, NP_net)
            for dof in range(2):
                F[2*i+dof] += W * element.phis[i](point[0], point[1]) * np.linalg.det(J) 
            
    return F
    
def stiffness(element, GPN=2):
    points, Ws = Gauss_points(element, GPN)
    n_nodes = element.n_nodes
    vertices = element.vertices
    E = element.E
    nu = element.nu
    K = np.zeros((n_nodes*2, n_nodes*2))
    for i in range(n_nodes):
        for j in range(n_nodes):
            Kij = np.zeros((2, 2))
            Fij = np.zeros(2)
            for g in range(len(points)):
                point = points[g]
                W = Ws[g]
                NP = []
                for node_id in range(n_nodes):
                    NP.append([element.phipxs[node_id](point[0], point[1]),
                               element.phipys[node_id](point[0], point[1])])
                NP_net = np.array(NP)
                # print('NP', NP.T)
                J = jacobian(vertices, NP_net)
                NP = np.linalg.inv(J).T @ NP_net.T
                # print(NP)
                for a in range((len(Kij))):
                    for c in range((len(Kij))):
                        for b in range(2):
                            for d in range(2):
                                Kij[a, c]+= NP[b, i] * constitutive(a, b, c, d, E, nu) * NP[d, j] * np.linalg.det(J) * W

                # print(left_v)
            K[2*i:2*(i+1), 2*j:2*(j+1)] = Kij

    K[np.abs(K) < 1e-10] = 0
    return K


class Node:
    def __init__(self, xy, id=0):
        self.xy = xy
        self.id=id
        self.value = np.zeros(2)
        self.type='center'
        self.BC = [0, 0] # -1: Neumann, 1: Dirichlet
        assert self.type in ['center','ellipse' 'le','re', 'be', 'te', 'ltc', 'rtc',
                             'lbc', 'blc', 'rbc'], "No.{} Node has a wrong type{}".format(self.id, self.type)



class Element:
    def __init__(self, nodes, E=200e3, nu=0.3, A=40, id=0, GPN=3):
        # E = 2000Mpa, nu = 0.3, A=4omm2
        self.E = E
        self.nu = nu
        self.A = A
        self.id = id
        self.nodes = nodes
        self.n_nodes = len(self.nodes)
        vertices = []
        self.node_id = []
        for Node in self.nodes:
            vertices.append(Node.xy)
            self.node_id.append(Node.id)
        self.vertices = np.array(vertices)
        self.phis = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [T3_phipx([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipys = [T3_phipy([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.K = np.zeros((self.n_nodes, self.n_nodes))
        self.shape = 'init'
        self.scale_xi = [0, 1]
        self.scale_eta = [0, 1]
        self.xi_eta = np.array([[1, 0], [0, 1], [0, 0]])
        self.GPN = GPN
    def mapping(self, refine):
        xi_eta = self.sample_points(refine)
        vertices = self.vertices
        vts = []
        for loc in xi_eta:
            xi, eta = loc
            vet = np.zeros(2)
            for i in range(len(vertices)):
                vet += self.phis[i](xi, eta) * vertices[i]
            vts.append(vet)
        return np.array(vts)
    # def mapping(self, refine):
    #     NP = np.zeros((self.n_nodes, 2))
    #     for i in range(len(self.nodes)):
    #         xi, eta = 0, 0
    #         NP[i][0] = self.phipxs[i](xi, eta)
    #         NP[i][1] = self.phipys[i](xi, eta)
    #         # self.NP[i][0] = self.phipxs[i](self.xi_eta[i][0], self.xi_eta[i][1])
    #         # self.NP[i][1] = self.phipys[i](self.xi_eta[i][0], self.xi_eta[i][0])
    #     J_init = jacobian(self.vertices, NP)
    #     print('self.NP', NP)
    #     J_inv = np.linalg.inv(J_init)
    #     local_points = self.sample_points(refine)
    #     global_points = [J_inv @ loc_point.T for loc_point in local_points]
    #         # Remove the homogeneous coordinate
    #     return np.array(global_points)

    def sample_points(self, refine):
        xi_eta = self.xi_eta
        # Determine if it's a triangle or a quadrilateral based on the number of vertices
        if len(xi_eta) == 3:  # Triangle
            xi_range = np.linspace(0, 1, refine)
            eta_range = np.linspace(0, 1, refine)
            points = []
            for xi in xi_range:
                for eta in eta_range:
                    if xi + eta <= 1:
                        points.append([xi, eta])
        elif len(xi_eta) == 4:  # Quadrilateral
            xi_range = np.linspace(-1, 1, refine)
            eta_range = np.linspace(-1, 1, refine)
            points = [[xi, eta] for xi in xi_range for eta in eta_range]
        else:
            raise ValueError("Only triangles and quadrilaterals are supported.")
    
        return np.array(points)

    def __str__(self):
        return str(self.vertices)
    
    def __call__(self, x=0, y=0):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            # Deal with single numbers
            value = 0
            for i in range(len(self.nodes)):
                value += self.nodes[i].value * self.phis[i](x, y)
            return value
        else:
            # Deal with arrays
            x = np.asarray(x)
            y = np.asarray(y)

            value = np.zeros((len(x), len(y)))
            for i in range(len(self.nodes)):
                for ix, x_val in enumerate(x):
                    for iy, y_val in enumerate(y):
                        value[iy, ix] += self.nodes[i].value[0] * self.phis[i](x_val, y_val)
            return value
    def in_element(self, x, y):
        point = x, y
        polygon = np.array(self.vertices)
        n = len(polygon)
        count = 0  # 初始化count

        for i in range(n):
            p1, p2 = polygon[i], polygon[(i + 1) % n]

            if np.all(p1 == point) or np.all(p2 == point):
                return True

            if p1[1] == p2[1]:  # 当边界是水平线时
                if point[1] == p1[1] and min(p1[0], p2[0]) < point[0] < max(p1[0], p2[0]):  # 仅当点位于两个端点之间时才返回True
                    return True
            elif min(p1[1], p2[1]) < point[1] <= max(p1[1], p2[1]):  # 确保点位于线段的y范围内
                x = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if x == point[0]:
                    return True
                elif x > point[0]:
                    count += 1
        return count % 2 == 1
    
    # def __str__(self):
    #     return str(self.vertices)
    
    # def __call__(self, x=0, y=0):
        
    #     if isinstance(x, (int, float)) and isinstance(y, (int, float)):
    #         # Deal with single numbers
    #         value = 0
    #         for i in range(len(self.nodes)):
    #             if self.in_element(x, y):
    #                 value += self.nodes[i].value * self.phis[i](x, y)
    #         return value
    #     else:
    #         # Deal with arrays
    #         x = np.asarray(x)
    #         y = np.asarray(y)

    #         value = np.zeros((len(x), len(y)))
    #         for i in range(len(self.nodes)):
    #             for ix, x_val in enumerate(x):
    #                 for iy, y_val in enumerate(y):
    #                     if self.in_element(x_val, y_val):
    #                         value[iy, ix] += self.nodes[i].value[0] * self.phis[i](x_val, y_val)
    #         return value
        
class T3(Element):
    def __init__(self, nodes, E=200e3, nu=0.3, A=40, id=0, GPN=3):
        super().__init__(nodes,E, nu, A, id, GPN)
        assert len(self.nodes)==3, "The number of nodes must be 3 in T3 element, rather than {}".format(len(self.nodes))
        
        self.phis = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [T3_phipx([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipys = [T3_phipy([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.shape='triangle'
        self.scale_xi = [0, 1]
        self.scale_eta = [0, 1]
        self.xi_eta = np.array([[1, 0], [0, 1], [0, 0]])


        self.K = stiffness(self, GPN)        
        self.F = force(self,GPN)
        
class Q4(Element):
    def __init__(self, nodes, E=200e3, nu=0.3, A=40, id=0, GPN=2):
        super().__init__(nodes,E, nu, A, id, GPN)
        assert len(self.nodes)==4, "The number of nodes must be 4 in Q4 element, rather than {}".format(len(self.nodes))
        
        self.phis = [Q4_phi([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [Q4_phipx([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipys = [Q4_phipy([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.shape = 'quad'
        self.scale_xi = [-1, 1]
        self.scale_eta = [-1, 1]
        self.xi_eta = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        self.K = stiffness(self, GPN) 
        self.F = force(self,GPN)
        
# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the global coordinate of the phip

    # compute Jacobian matrix
    J = X.T @ dN
    return J 

def assemable_elements(element_list):
    def model(x, y, type='stress', dir='x'):
        init_matrix = element_list[0](x, y)
        for i in range(1, len(element_list)):
            init_matrix = add_matrices(init_matrix, element_list[i](x, y))
        return init_matrix
    return model
if __name__=="__main__":
    E = 8/3
    nu = 1/3
    GPN = 4
    vertices_T3 = np.array([[16.45327476, 25.20273424], [23.90255057, 19.62681142], [24.7911839 , 27.45556408]])
    # vertices_T3 = [[1, 0], [0, 1], [0, 0]]
    # vertices_T3 = [[0, 0], [2, 0], [1, 1]]

    vertices_Q4 = np.array([[0.5, 0.5], [1, 0], [1, 2], [0, 2]])
    vertices_Q4_2 = [[1, 0], [2, 0], [2, 2], [1, 2]]
    # vertices_Q4 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    # vertices_Q4 = [[1, 1], [2., 1], [2.5, 2.5], [1., 2]]

    Node_list_T3 = []
    for i in range(len(vertices_T3)):
        Node_list_T3.append(Node(vertices_T3[i], i))
        # Node_list_T3[-1].value = [np.random.rand(), np.random.rand()]
        Node_list_T3[-1].value = [1, 1]

    Node_list_Q4 = []
    Node_list_Q4_2= []
    for i in range(len(vertices_Q4)):
        Node_list_Q4.append(Node(vertices_Q4[i], i))
        Node_list_Q4[-1].value = [np.random.rand(),np.random.rand() ]
        Node_list_Q4[-1].value = [1, 1]
        Node_list_Q4_2.append(Node(vertices_Q4_2[i], i))
        Node_list_Q4_2[-1].value = [np.random.rand(),np.random.rand() ]
        Node_list_Q4_2[-1].value = [1, 1]

    T3_node = Node_list_T3[0]
    Q4_node = Node_list_Q4[0]
    Q4_node_2 = Node_list_Q4_2[0]
    T3_element = T3(Node_list_T3, E=E, nu=nu, GPN=GPN)
    
    print(T3_element.K)
    Q4_element = Q4(Node_list_Q4, E=E, nu=nu, GPN=GPN)
    Q4_element_2 = Q4(Node_list_Q4_2, E=E, nu=nu, GPN=GPN)
    print(Q4_element.K)
    t3_phi = T3_phi(0)
    F_T3 = force(T3_element)
    print('F_T3', F_T3)

    F_Q4 = force(Q4_element)
    print('F_Q4', F_Q4)

    # ????????????
    x0, x1 = [0, 1]
    y0, y1 = [0, 1]
    xi = np.linspace(x0, x1, 100)
    eta = np.linspace(y0, y1, 100)
    #    xi = 0.1
    #    eta = 0.2
    
    # ??expression???????
    T3_element.nodes[0].value = [1, 1] 
    output = t3_phi(xi, eta)
    refine = 100
    Q4_mapping = Q4_element.mapping(refine)
    T3_mapping = T3_element.mapping(refine)
    plt.scatter(T3_mapping[:, 0],T3_mapping[:, 1], s=100,  label='test')
    plt.scatter(vertices_T3[:, 0], vertices_T3[:, 1], label='real')
    plt.legend()
    plt.show()

    # output = Q4_element(xi, eta)
    element_list = [Q4_element, Q4_element_2]
    model = assemable_elements(element_list)
    # output = model(xi, eta)
    test_point = [0, 1.1]
    print(T3_element.in_element(test_point[0], test_point[1]))
    print(output)
    plt.imshow(output, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
    plt.colorbar()
    plt.title('Shape Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
