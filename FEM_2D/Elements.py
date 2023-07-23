import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from shape_fns import *
from tools_2D import *                

def cal_K(element):
    num_shape_fns = len(element.phis)
    K = np.zeros((num_shape_fns, num_shape_fns))
    invers_J = element.J_inv*element.J_det
    F = np.zeros(num_shape_fns)
    for i in range(num_shape_fns):
        for j in range(num_shape_fns):
            pu_pxi = plus(mul(invers_J[0][0], element.phipxs[i]),
                         mul(invers_J[0][1], element.phipys[i]))
            pu_pxj = plus(mul(invers_J[0][0], element.phipxs[j]),
                         mul(invers_J[0][1], element.phipys[j]))
            pu_pyi = plus(mul(invers_J[1][0], element.phipxs[i]),
                         mul(invers_J[1][1], element.phipys[i]))
            pu_pyj = plus(mul(invers_J[1][0], element.phipxs[j]),
                         mul(invers_J[1][1], element.phipys[j]))
            this_funs = plus(mul(pu_pxi, pu_pxj), mul(pu_pyi, pu_pyj))
            scale_x = this_funs.scale_x
            scale_y = this_funs.scale_y
            K_value= G_integrate_2D(this_funs, N=3, scale_x=scale_x, scale_y=scale_y)
            if abs(K_value)>1e-10:
                K[i, j] = K_value
    
    return K

class Node:
    def __init__(self, xy, id=0):
        self.xy = xy
        self.id=id
        self.value = 0


class Element:
    def __init__(self, nodes, id=0):
       
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
        self.phixs = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phiys = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.eta = 1
        self.xi = 1
        self.K = np.zeros((self.n_nodes, self.n_nodes))
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
    def __init__(self, nodes, id=0):
        super().__init__(nodes, id)
        assert len(self.nodes)==3, "The number of nodes must be 3 in T3 element, rather than {}".format(len(self.nodes))
        
        self.phis = [T3_phi([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [T3_phipx([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.phipys = [T3_phipy([0, 1], [0, 1], p) for p in range(len(self.nodes))]
        self.eta = 1
        self.xi = 1
        self.J = np.array([[self.vertices[1][0]-self.vertices[0][0], self.vertices[1][1]-self.vertices[0][1]], 
                           [self.vertices[2][0]-self.vertices[0][0], self.vertices[2][1]-self.vertices[0][1]]])
        self.J_inv = np.linalg.inv(self.J)
        self.J_det = np.linalg.det(self.J)
        self.K = cal_K(self)        
        
class Q4(Element):
    def __init__(self, nodes, id=0):
        super().__init__(nodes, id)
        assert len(self.nodes)==4, "The number of nodes must be 4 in Q4 element, rather than {}".format(len(self.nodes))
        
        self.phis = [Q4_phi([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipxs = [Q4_phipx([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.phipys = [Q4_phipy([-1, 1], [-1, 1], p) for p in range(len(self.nodes))]
        self.eta = 1
        self.xi = 1
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        self.J = np.array([[((x[1]-x[0])*(1-self.eta)+(x[2]-x[3])*(1+self.eta))/4, ((y[1]-y[0])*(1-self.eta)+(y[2]-y[3])*(1+self.eta))/4 ], [((x[2]-x[1])*(1+self.xi)+(x[3]-x[0])*(1-self.xi))/4, ((y[2]-y[1])*(1+self.xi)+(y[3]-y[1])*(1-self.xi))/4]]) # http://www.ce.memphis.edu/7111/notes/class_notes/chapter_03c_slides.pdf
        self.J_inv = np.linalg.inv(self.J)
        self.J_det = np.linalg.det(self.J)
        self.K = cal_K(self) 

        
# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the local coordinate of the phip

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    print(X, np.transpose(X))
    print(dN)

    return np.linalg.inv(J), np.linalg.det(J)
if __name__=="__main__":
    vertices_T3 = [[0, 0], [2, 1], [0.5, 2]]
    vertices_T3 = [[16.45327476, 25.20273424], [23.90255057, 19.62681142], [24.7911839 , 27.45556408]]
    # vertices_T3 = [[1, 0.2], [2, 1.3], [0.5, 1.7]]
    vertices_T3 = [[0, 0], [1, 0], [0, 1]]
    
    vertices_Q4 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    # vertices_Q4 = [[1, 1], [2, 1], [3, 2], [2, 2]]
    # vertices_Q4 = [[1, 1], [2., 1], [2.5, 2.5], [1., 2]]

    Node_list_T3 = []
    for i in range(len(vertices_T3)):
        Node_list_T3.append(Node(vertices_T3[i], i+1))
    Node_list_Q4 = []
    for i in range(len(vertices_Q4)):
        Node_list_Q4.append(Node(vertices_Q4[i], i+1))
    triangle = T3(Node_list_T3)
    print(triangle.K)
    Q4_element = Q4(Node_list_Q4)
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
    

 
