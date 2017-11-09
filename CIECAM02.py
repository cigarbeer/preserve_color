import deviceCharacteristicModel as dch 

import numpy as np 

M_CAT02 = np.array([
    [ 0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975,  0.0061],
    [ 0.0030, 0.0136,  0.9834]
])

c = np.array([
    0.690,
    0.590,
    0.525
])

N_c = np.array([
    1.0, 
    0.9,
    0.8
])

F = np.array([
    1.0,
    0.9,
    0.8
])

M_H = np.array([
    [ 0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340,  0.04641],
    [ 0.00000, 0.00000,  1.00000]
])

L_A = 60

Y_b = 25

