import numpy as np 

def D(F, L_A):
    result = F * (1 - 1/3.6 * np.exp(-(L_A+42)/92))
    return result

def F_L(L_A):
    k = 1 / (5 * L_A + 1)
    result = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4)**2 * (5 * L_A)**(1 / 3)
    return result

def n(Y_b, Y_w):
    result = Y_b / Y_w
    return result 

def N_cb(n):
    result = 0.725 * (1 / n)**0.2
    return result 

def N_bb(n):
    result = N_cb(n=n)
    return result 

def z(n):
    result = 1.48 + n**0.5
    return result 

def h(a, b):
    rad = np.arctan2(b, a)
    degree = np.degrees(rad)
    positiveDegree = np.mod(degree, 360)
    return positiveDegree

def J(A, A_white, c, z):
    result = 100 * (A / A_white)**(c * z)
    return result

def e_t(h):
    result = 1/4 * (np.cos(np.pi/180 * h + 2) + 3.8)
    return result 

def t(N_c, N_cb, e_t, a, b, HPE):
    result = (50000/13 * N_c * N_cb * e_t * (a**2 + b**2)**0.5) / (HPE[:, :, 0] + HPE[:, :, 1] + 21/20*HPE[:, :, 2])
    return result

def C(t, J, n):
    result = t**0.9 * (0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73
    return result 

def t_inv(C, J, n):
    result = (C / ((0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73))**(1/0.9)
    return result 

def A_inv(A_white, J, c, z):
    result = (0.01 * J)**(1 / (c * z)) * A_white
    return result 

def ab_inv(A, h, t, e_t, N_c, N_cb, N_bb):
    h = np.deg2rad(h)

    p1 = (50000/13 * N_c * N_cb * e_t) / t 
    p2 = (A / N_bb) + 0.305
    p3 = 21 / 20
    
    sin_h = np.sin(h)
    cos_h = np.cos(h)
    tan_h = np.tan(h)
    cot_h = np.nan_to_num(1 / tan_h)

    singtcos = np.abs(sin_h) >= np.abs(cos_h)
    sinltcos = ~singtcos

    # sin > cos
    p4 = np.nan_to_num(p1 / sin_h)
    cb_gt = (p2 * (2 + p3) * (460/1403)) / (p4 + (2 + p3) * (220/1403) * cot_h - (27/1403) + p3 * (6300/1403))
    ca_gt = cb_gt * cot_h
    # sin < cos
    p5 = np.nan_to_num(p1 / cos_h)
    ca_lt = (p2 * (2 + p3) * (460/1403)) / (p5 + (2 + p3) * (220/1403) - ((27/1403) + p3 * (6300/1403)) * tan_h)
    cb_lt = ca_lt * tan_h

    a = np.zeros(h.shape)
    b = np.zeros(h.shape)

    a[singtcos] = ca_gt[singtcos]
    a[sinltcos] = ca_lt[sinltcos]

    b[singtcos] = cb_gt[singtcos]
    b[sinltcos] = cb_lt[sinltcos]

    return (a, b)