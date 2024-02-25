import numpy as np
from numpy.linalg import inv, det

# functions that return shape functions, their derivatives, and the global coordinate

def fns_2(xi, X):
    
    N = 0.5*np.array([1 - xi, 1 + xi])
    dN = np.array([[-1/2],[1/2]])
    return N, dN


def fns_3(xi, X):

    N = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
    dN = np.array([[-1, -1], [1, 0], [0, 1]])
    return N, dN
    

def fns_4(xi, X):

    N = 0.25*np.array([(1 + xi[0])*(1 + xi[1]),
                       (1 - xi[0])*(1 + xi[1]),
                       (1 - xi[0])*(1 - xi[1]),
                       (1 + xi[0])*(1 - xi[1])])
    dN = 0.25*np.array([[ (1 + xi[1]), (1 + xi[0])],
                        [-(1 + xi[1]), (1 - xi[0])],
                        [-(1 - xi[1]),-(1 - xi[0])],
                        [ (1 - xi[1]),-(1 + xi[0])]])
    return N, dN


# compute Jacobian matrix
def jacobian(X, dN):
    # X is the global coordinate for each node.
    # dN is the local coordinate of the phip

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    return inv(J), det(J)

if __name__ == '__main__':
    xi = [3, 3]
    X = np.array([[0, 0],
              [3, 0],
              [0, 3]])
    N, dN = fns_3(xi, X)
    print('N', N, 'dN', dN)
    J_inv, detJ = jacobian(X, dN)
    print(J_inv, detJ)
