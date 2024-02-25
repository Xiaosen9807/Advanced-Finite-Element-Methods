import numpy as np
import math
import sys
def Gauss_points(NPE, order):
    """
    Return Gauss integration points and weights for the given shape and order using leggauss.
    
    Parameters:
    - shape: 'quad' for quadrilateral, 'triangle' for triangle
    - order: desired accuracy of integration (1, 2, 3, ...)

    Returns:
    - points: list of Gauss points
    - weights: list of Gauss weights
    """
    
    if NPE == 4:
        xi, wi = np.polynomial.legendre.leggauss(order)
        points = [(x, y) for x in xi for y in xi]
        weights = [wx * wy for wx in wi for wy in wi]
        
    elif NPE == 3:
        NGP_data = {
            1: {
                'points': [(1/3, 1/3)],
                'weights': [1/2]
            },
            3: {
                'points': [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)],
                'weights': [1/6, 1/6, 1/6]
            },
            4: {
                'points': [(1/3, 1/3), (0.6, 0.2), (0.2, 0.6), (0.2, 0.2)],
                'weights': [-27/96, 25/96, 25/96, 25/96]
            }
        }
        
        if order == 2:
            order = 3
        points, weights = NGP_data[order]['points'], NGP_data[order]['weights']
    else:
        raise ValueError("Shape not supported")

    return points, weights

def constitutive(i, j, k, l):
    E = 8/3
    nu = 1/3
    delta = lambda x, y: 1 if x == y else 0
    term1 = (E / (2 * (1 + nu))) * (delta(i, l) * delta(j, k) + delta(i, k) * delta(j, l))
    term2 = (E * nu) / (1 - nu**2) * delta(i, j) * delta(k, l)

    return term1 + term2

def grad_N_nat(NPE, xi, eta):
    PD = 2
    result = np.zeros([PD, NPE])

    if NPE == 3:
        result[0, 0] = 1
        result[0, 1] = 0
        result[0, 2] = -1
        result[1, 0] = 0
        result[1, 1] = 1
        result[1, 2] = -1

    if NPE == 4:
        result[0, 0] = -1/4 * (1 - eta)
        result[0, 1] = 1/4 * (1 - eta)
        result[0, 2] = 1/4 * (1 + eta)
        result[0, 3] = -1/4 * (1 + eta)
        result[1, 0] = -1/4 * (1 - xi)
        result[1, 1] = -1/4 * (1 + xi)
        result[1, 2] = 1/4 * (1 + xi)
        result[1, 3] = 1/4 * (1 - xi)

    return result

def stiffness(X, GPE):
    NPE = np.size(X, 0)
    PD = np.size(X, 1)
    K = np.zeros([NPE*PD, NPE*PD])
    coor = X.T
    print(coor)

    for i in range(1, NPE + 1):
        for j in range(1, NPE + 1):
            k = np.zeros([PD, PD])
            points, alphas = Gauss_points(NPE, GPE)
            for gp in range(1, GPE + 1):
                (xi, eta) = points[gp-1]
                alpha = alphas[gp-1]
                J = np.zeros([PD, PD])
                grad = np.zeros([PD, NPE])
                grad_nat = grad_N_nat(NPE, xi, eta)
                J = coor @ grad_nat.T
                
                # print('coor', coor)
                grad = np.linalg.inv(J).T @ grad_nat
                # print(K.sdgu)
                for a in range(1, PD + 1):
                    for c in range(1, PD + 1):
                        for b in range(1, PD + 1):
                            for d in range(1, PD + 1):
                                k[a-1, c-1] = k[a-1, c-1] + grad[b-1, i-1] * constitutive(a, b, c, d) * grad[d-1, j-1] * np.linalg.det(J) * alpha
            K[(i-1)*PD:i*PD, (j-1)*PD:j*PD] = k

    return K
if __name__=="__main__":
    x = np.array([[0, 0],
                  [1, 0],
                  [1, 2],
                  [0, 2]])
    x = np.array([[0, 0], [2, 0], [1, 1]])
    GPN = 1
    K = stiffness(x, GPN)
    print(K)

    
    
