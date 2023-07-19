import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jacobi
import sympy as sp

class shape_fns:
    def __init__(self, scale_x = [0, 1], scale_y = [0, 1], p=0):
        self.scale_x = scale_x
        self.scale_y = scale_y
        # self.x_l = scale_x[0]
        # self.x_r = scale_x[1]
        # self.y_l = scale_y[0]
        # self.y_r = scale_y[1]
        self.p = p
        
    def expression(self, ksi, neta): 
        return 1-ksi-neta
    
    def __call__(self, x=0, y=0):
        x = np.asarray(x)  # convert x to a numpy array if it's not already
        y = np.asarray(y)  # convert y to a numpy array if it's not already
        expression_vectorized = np.vectorize(self.expression, otypes=['d'])
        return np.where((self.scale_x[0] <= x) & (x <= self.scale_x[1]) & (self.scale_y[0] <= y) & (y <= self.scale_y[1]), expression_vectorized(x, y), 0)


class T3_phi(shape_fns):
    def expression(self, ksi, neta): 
        if self.p == -1:
             return 1-ksi-neta
        elif self.p == 0:
            return ksi
        elif self.p == 1:
            return neta 
        
class T3_phipx(shape_fns):
    def expression(self, ksi=0, neta=0):
        if self.p == -1:
             return -1+np.zeros_like(ksi)
        elif self.p == 0:
            return 1+np.zeros_like(ksi)
        elif self.p == 1:
            return 0+np.zeros_like(ksi)

class T3_phipy(shape_fns):
    def expression(self, ksi=0, neta=0):
        if self.p == -1:
             return  -1 +np.zeros_like(neta)
        elif self.p == 0:
            return 0+np.zeros_like(neta)
        elif self.p == 1:
            return 1+np.zeros_like(neta)
        
class Element:
    def __init__(self, vertices ):
        self.vertices  = vertices 
        self.nodes = [Node(co) for co in vertices ]
        self.n_nodes = len(self.nodes)
        self.neta = 1
        self.ksi = 1
        
class Node:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]

class T3(Element):
    def __init__(self, vertices ):
        super().__init__(vertices)
        assert len(vertices) == 3, "The number of vertices must be 3 in T3 element"
        self.vertices_l = [[0, 0], [1, 0], [0, 1]]
        self.funcs = [T3_phi([0, 1], [0, 1], p) for p in range(-1, 2)]
        self.phipxs = [T3_phipx([0, 1],[0, 1], p) for p in range(-1, 2)]
        self.phipys = [T3_phipy([0, 1],[0, 1], p) for p in range(-1, 2)]

        self.J = np.array([[vertices[1][0]-vertices[0][0], vertices[1][1]-vertices[0][1]], 
                           [vertices[2][0]-vertices[0][0], vertices[2][1]-vertices[0][1]]])
        self.J_inv = np.linalg.inv(self.J)
        self.J_det = np.linalg.det(self.J)
        
# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the local coordinate of the phip

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    print(X, np.transpose(X))
    print(dN)

    return np.linalg.inv(J), np.linalg.det(J)



        
if __name__=='__main__':
 
   vertices = [[0, 0], [2, 1], [0.5, 2]]
   vertices = [[1, 0.2], [2, 1.3], [0.5, 1.7]]
   triangle = T3(vertices)
   print(triangle.phipxs[1](x=[0.2, 0.3]))
   print(triangle.J_inv)
   print(triangle.J)
   K = np.zeros((3, 3))
