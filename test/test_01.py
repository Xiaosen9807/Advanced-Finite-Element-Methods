import numpy as np

NL = np.array([[0, 0],
              [0, 1],
               [0.5, 1]])

EL = np.array([[1, 2],
               [2, 3],
               [3, 1]])
DorN = np.array([[-1, -1],
                 [1, -1],
                 [1, 1]])
Fu = np.array([[0, 0],
               [0, 0],
               [0, -20]])
U_u = np.array([[0, 0],
                [0, 0],
                [0, 0]])
E = 1e6
A = 0.01

PD = np.size(NL, 1) # Problem Dimension
NoN = np.size(NL, 0) # Number of nodes
ENL = np.zeros([NoN, 6*PD])
ENL[:, 0:PD] = NL[:, :]
ENL[:, PD:2*PD] = DorN[:, :]

(ENL, DOFs, DOCs) = assign_BCs(NL, ENL)
