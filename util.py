import numpy as np 


def dot(M, img):
    shape = img.shape
    img_2D = img.reshape((-1, shape[-1]))
    img_2D_t = np.moveaxis(a=img_2D, source=-1, destination=0)
    result_2D_t = np.dot(a=M, b=img_2D_t)
    result_2D = np.moveaxis(a=result_2D_t, source=0, destination=-1)
    result = result_2D.reshape(shape)
    return result

def mul(v, img):
    result = np.multiply(v, img)
    return result

def pow(exponent, img):
    result = np.power(img, exponent)
    return result

def angle(u, v):
    # inner = np.sum(np.multiply(u, v))
    # outer = np.linalg.norm(np.cross(u, v), axis=-1)
    ang = np.arctan2(u, v)
    return ang

def angle_degree(u, v):
    ang = angle(u, v)
    angDeg = np.mod(np.degrees(ang), 360)
    return angDeg


# def XYZtoLMS(XYZ, M):
#     # XYZ (3, n, m)
#     # M (3, 3)
#     LMS = dot(M=M, img=XYZ)
#     return LMS 
# def normalizeRGB(RGB):
#     piexlSum = np.sum(a=RGB, axis=0)
#     RGB_n = np.divide(RGB, piexlSum)
#     return RGB_n 
# def gammaCorrection(RGB_n, gamma):
#     RGB_l = pow(exponent=gamma, img=RGB_n)
#     return RGB_l
# def RGB_ltoXYZ(RGB_l, M):
#     XYZ = dot(M=M, img=RGB_l)
#     return XYZ
# def luminanceLevelAdaptationFactor(L_A):
#     k = 1 / (5 * L_A + 1)
#     F_L = 0.2 * k**4 * (5*L_A) + 0.1 * (1 - k**4)**2 * (5*L_A)**(1/3)
#     return F_L
# def degreeOfAdaptation(F, L_A):
#     D = F * (1 - 1/3.6 * np.exp(x=-(L_A+42)/92))
#     return D
# def gainControl(LMS, D, LMS_w):
#     # LMS (3, n ,m)
#     # D (1)
#     # LMS_w (3)

#     # ((100/LMS_w - 1) * D + 1) * LMS_t

#     LMS_w_t = (100/LMS_w - 1) * D + 1
#     LMS_c = mul(v=LMS_w_t, img=LMS)
#     return LMS_c 
# def compression(LMS_prime, F_L):
#     t = (F_L * LMS_prime / 100)**0.42
#     LMS_a_prime = (400 * t) / (27.13 + t) + 0.1
#     return LMS_a_prime
# def LMS_ctoLMS_prime(LMS_c, M_CAT02_inv, M_H):
#     XYZ_c = dot(M=M_CAT02_inv, img=LMS_c)
#     LMS_prime = dot(M=M_H, img=XYZ_c)
#     return LMS_prime