import numpy as np

def assign_BCs(NL, ENL):
    PD = np.size(NL, 1) # Problem Dimension
    NoN = np.size(NL, 0) # Number of nodes
    DOFs = 0
    DOCs = 0

    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i, PD+j] == -1:
                DOCs -=1
                ENL[i, 2*PD+j] = DOCs
            else:
                DOFs+=1
                ENL[i, 2*PD+j] = DOFs
    for i in range(0, NoN):
        for j in range(0, PD):
            if ENL[i,2*PD+j]<0:
                ENL[i, 3*PD+j] = abs(ENL[i, 2*PD+j]) + DOFs
        
                
    
