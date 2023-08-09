import numpy as np
import matplotlib.pyplot as plt
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
            F[2*i] += W * element.phis[i](point[0], point[1]) * np.linalg.det(J) 
            F[2*i+1] += W * element.phis[i](point[0], point[1]) * np.linalg.det(J)
            
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
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0, GPN=3):
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
        self.GPN = GPN
    def in_element(self, x, y):
        point = x, y
        polygon = np.array(self.vertices)
        n = len(polygon)
        for i in range(n):
            p1, p2 = polygon[i], polygon[(i + 1) % n]

            if np.all(p1 == point) or np.all(p2 == point):
                return True

            if p1[1] == p2[1]:
                if point[1] == p1[1] and min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]):
                    return True
            elif min(p1[1], p2[1]) <= point[1] < max(p1[1], p2[1]):
                x = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]  
                if x == point[0]:
                    return True
                elif x > point[0]:
                    count += 1
        return count % 2 == 1
    
    def __str__(self):
        return str(self.vertices)
    
    def __call__(self, x=0, y=0):
        
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            # Deal with single numbers
            value = 0
            for i in range(len(self.nodes)):
                if self.in_element(x, y):
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
                        if self.in_element(x_val, y_val):
                            value[iy, ix] += self.nodes[i].value * self.phis[i](x_val, y_val)
            return value
        
class T3(Element):
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0, GPN=3):
        super().__init__(nodes,E, nu, A, id, GPN)
        assert len(self.nodes)==3, "The number of nodes must be 3 in T3 element, rather than {}".format(len(self.nodes))
        
        self.phis = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [T3_phipx([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipys = [T3_phipy([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.shape='triangle'
        self.scale_xi = [0, 1]
        self.scale_eta = [0, 1]


        self.K = stiffness(self, GPN)        
        self.F = force(self,GPN)
        
class Q4(Element):
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0, GPN=2):
        super().__init__(nodes,E, nu, A, id, GPN)
        assert len(self.nodes)==4, "The number of nodes must be 4 in Q4 element, rather than {}".format(len(self.nodes))
        
        self.phis = [Q4_phi([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [Q4_phipx([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipys = [Q4_phipy([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.shape = 'quad'
        self.scale_xi = [-1, 1]
        self.scale_eta = [-1, 1]
        self.K = stiffness(self, GPN) 
        self.F = force(self,GPN)

        
# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the global coordinate of the phip

    # compute Jacobian matrix
    J = X.T @ dN
    return J 
if __name__=="__main__":
    E = 8/3
    nu = 1/3
    GPN = 4
    vertices_T3 = [[16.45327476, 25.20273424], [23.90255057, 19.62681142], [24.7911839 , 27.45556408]]
    vertices_T3 = [[1, 0], [0, 1], [0, 0]]
    vertices_T3 = [[0, 0], [2, 0], [1, 1]]

    vertices_Q4 = [[0, 0], [1, 0], [1, 2], [0, 2]]
    # vertices_Q4 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    # vertices_Q4 = [[1, 1], [2., 1], [2.5, 2.5], [1., 2]]

    Node_list_T3 = []
    for i in range(len(vertices_T3)):
        Node_list_T3.append(Node(vertices_T3[i], i))
    Node_list_Q4 = []
    for i in range(len(vertices_Q4)):
        Node_list_Q4.append(Node(vertices_Q4[i], i))
    T3_node = Node_list_T3[0]
    Q4_node = Node_list_Q4[0]
    T3_element = T3(Node_list_T3, E=E, nu=nu, GPN=GPN)
    print(T3_element.K)
    Q4_element = Q4(Node_list_Q4, E=E, nu=nu, GPN=GPN)
    print(Q4_element.K)
    t3_phi = T3_phi(0)
    F_T3 = force(T3_element)
    print('F_T3', F_T3)

    F_Q4 = force(Q4_element)
    print('F_Q4', F_Q4)

    # ????????????
    x0, x1 = [0, 2]
    y0, y1 = [0, 2]
    xi = np.linspace(x0, x1, 100)
    eta = np.linspace(y0, y1, 100)
    #    xi = 0.1
    #    eta = 0.2
    
    # ??expression???????
    output = t3_phi(xi, eta)
    print('output', output)
    plt.imshow(output, origin='lower', extent=[x0, x1, y0, y1], cmap='jet')
    plt.colorbar()
    plt.title('Shape Function')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
