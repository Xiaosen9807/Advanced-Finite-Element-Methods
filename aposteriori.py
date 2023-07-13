from math import log
from scipy.optimize import brentq

# Note: N0 < N1 < N2
N0, N1, N2 = 17, 23, 30
# Note: U0 < U1 < U2
U0, U1, U2 = 0.0702794952, 0.0702856015, 0.0702874107

# compute exponent
Q = log(N1/N0) / log(N2/N1)

def fn(U):
    return (U - U0)/(U - U1) - ( (U - U1)/(U - U2) )**Q

root = brentq(fn, 0.0702, 0.0705, full_output=True)

print("Approximated value of energy:", root[0])
print(root)
