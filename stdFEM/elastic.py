import numpy as np

# obtain Lamé consants
def lame(E, Nu):
    return 0.5*E/(1+Nu), E*Nu/((1+Nu)*(1-2*Nu))


def constitutive(material, dim):

    # 1D analysis
    if dim == 1:
        
        E, A = material
        return E*A

    # 2D analysis
    elif dim == 2:
    
        E, Nu, case = material
        Mu, Lambda = lame(E, Nu)
        
        if case == "plane_strain":
            
            # compute elasticity tensor
            return np.array([[Lambda+2*Mu, 0, 0, Lambda],
                             [0, Mu, Mu, 0],
                             [0, Mu, Mu, 0],
                             [Lambda, 0, 0, Lambda+2*Mu]])
        
        elif case == "plane_stress":
        
            matDT = np.zeros((4,4))
            
            # compute elasticity tensor
            matDT[0,0] = matDT[3,3] = E / (1 - pow(Nu,2.))
            matDT[0,3] = matDT[3,0] = E * Nu / (1 - pow(Nu,2.))
            matDT[1,1] = matDT[1,2] = matDT[2,1] = matDT[2,2] = 0.5 * E / (1 + Nu)
            
            return matDT
        
        else:
        
            frame = inspect.currentframe()
            print("{}:{}: error: 2D analysis requires either plane stress or plane strain conditions\n".format(__file__,frame.f_lineno))
            quit()

    else:
    
        frame = inspect.currentframe()
        print("{}:{}: error: {}-dimensional analysis not supported for elastic material\n".format(__file__,frame.f_lineno, dim))
        quit()
