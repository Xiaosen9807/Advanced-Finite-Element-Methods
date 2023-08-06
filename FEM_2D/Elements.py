import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from shape_fns import *
from tools_2D import *                

   
def id_to_index(a, b, c, d):
    mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    i = mapping[str(a)+str(b)]
    j = mapping[str(c)+str(d)]
    return i, j

def delta(i, j):
    return int(i == j)

def cal_K(element):
    GPN = 3
    points, Ws = Gauss_points(GPN, element.scale_xi, element.scale_eta)
    n_nodes = element.n_nodes
    vertices = element.vertices
    E = element.E
    nu = element.nu
    E_matrix = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    i, j = id_to_index(a, b, c, d)
                    E_matrix[i, j] = E/(2+2*nu)*(delta(a,d)*delta(b,c)+delta(a,c)*delta(b,d)) + E*nu/(1-nu**2)*delta(a, b)*delta(c, d)
    K = np.zeros((n_nodes*2, n_nodes*2))
    for i in range(n_nodes):
        for j in range(n_nodes):
            Kij = np.zeros((2, 2))
            for a in range((len(Kij))):
                for c in range((len(Kij))):
                    E_abcd = np.zeros((2, 2))
                    for b in range(2):
                        for d in range(2):
                            id1, id2 = id_to_index(a, b, c, d)
                            E_abcd[b, d] = E_matrix[id1, id2]
                    for g in range(len(points)):
                        point = points[g]
                        NPi = [element.phipxs[i](point[0],point[1]),
                                element.phipys[i](point[0],point[1])]
                        NPj = [element.phipxs[j](point[0],point[1]),
                                element.phipys[j](point[0],point[1])]
                        left_v = np.dot(element.J_inv,NPi)
                        right_v = np.dot(element.J_inv, NPj)
                        Kij[a, c]+=Ws[g]*np.dot(np.dot(np.transpose(left_v),
                                        E_abcd), right_v)*element.J_det
                # print(left_v)
            K[2*i:2*(i+1), 2*j:2*(j+1)] = Kij

    K[np.abs(K) < 1e-10] = 0
    return K


class Node:
    def __init__(self, xy, id=0):
        self.xy = xy
        self.id=id
        self.value = 0
        self.type='center'
        assert self.type in ['center','ellipse' 'le','re', 'be', 'te', 'ltc', 'rtc',
                             'lbc', 'blc', 'rbc'], "No.{} Node has a wrong type{}".format(self.id, self.type)



class Element:
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0):  # E = 2000Mpa, nu = 0.3, A=4omm2
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
        # self.Jacobi()
    def initialize(self):
        self.phips = []
        for i in range(self.n_nodes):
            self.phips.append([self.phipxs[i], self.phipys[i]])
        self.NP = np.zeros((self.n_nodes, 2))
        for i in range(len(self.nodes)):
            self.NP[i][0] = self.phips[i][0](self.nodes[i].xy[0], self.nodes[i].xy[1])
            self.NP[i][1] = self.phips[i][1](self.nodes[i].xy[0], self.nodes[i].xy[1])
        self.J = jacobian(self.vertices, self.NP)
        self.J_inv = np.linalg.inv(self.J)
        self.J_det = np.linalg.det(self.J)
        self.scale_xi = [0, 1]
        self.scale_eta = [0, 1]
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
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0):
        super().__init__(nodes,E, nu, A, id)
        assert len(self.nodes)==3, "The number of nodes must be 3 in T3 element, rather than {}".format(len(self.nodes))
        
        self.phis = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [T3_phipx([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipys = [T3_phipy([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.initialize()
        self.scale_xi = [0, 1]
        self.scale_eta = [0, 1]


        self.K = cal_K(self)        
        
class Q4(Element):
    def __init__(self, nodes, E=2e3, nu=0.3, A=40, id=0):
        super().__init__(nodes,E, nu, A, id)
        assert len(self.nodes)==4, "The number of nodes must be 4 in Q4 element, rather than {}".format(len(self.nodes))
        
        self.phis = [Q4_phi([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [Q4_phipx([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipys = [Q4_phipy([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.initialize()
        print(np.dot(np.transpose(self.vertices), self.NP))
        self.scale_xi = [-1, 1]
        self.scale_eta = [-1, 1]
        self.K = cal_K(self) 

        
# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the global coordinate of the phip

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)

    return J 
if __name__=="__main__":
    E = 10
    nu = 0.5
    vertices_T3 = [[0, 0], [2, 1], [0.5, 2]]
    vertices_T3 = [[16.45327476, 25.20273424], [23.90255057, 19.62681142], [24.7911839 , 27.45556408]]
    # vertices_T3 = [[1, 0.2], [2, 1.3], [0.5, 1.7]]
    vertices_T3 = [[1, 0], [0, 1], [0, 0]]
    
    vertices_Q4 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    # vertices_Q4 = [[1, 1], [2, 1], [3, 2], [2, 2]]
    # vertices_Q4 = [[1, 1], [2., 1], [2.5, 2.5], [1., 2]]

    Node_list_T3 = []
    for i in range(len(vertices_T3)):
        Node_list_T3.append(Node(vertices_T3[i], i))
    Node_list_Q4 = []
    for i in range(len(vertices_Q4)):
        Node_list_Q4.append(Node(vertices_Q4[i], i))
    triangle = T3(Node_list_T3, E=E, nu=nu)
    print(triangle.K)
    Q4_element = Q4(Node_list_Q4, E=E, nu=nu)
    print(Q4_element.K)
    t3_phi = T3_phi(0)
    

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
    Node_list = Node_list_Q4
    
    # min_x = min([:, 0])
    for i in range(len(Node_list)):
        # Check left edge
        if Node_list[i].xy[0]==0:
            if Node_list[i].xy[1] == a:
                Node_list[i].type ='lbc'
            else:
                Node_list[i].type = 'le'
        # Check right edge
        elif Node_list[i].xy[0] == 40:
            Node_list[i].type = 're'
        # Check bottom edge
        if Node_list[i].xy[1]==0:
            if Node_list[i].xy[0]==b:
                Node_list[i].type='blc'
            if Node_list[i].type=='re':
                Node_list[i].type = 'rbc'
            else:
                Node_list[i].type = 'be'

        if Node_list[i].xy[1]==40:
            if Node_list[i].type=='le':
                Node_list[i].type ='ltc'
            else:
                Node_list[i].type='tc'
                      
