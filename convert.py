import numpy as np 
import util as ut 

M_full_backlight = np.array([
    [95.57, 64.670, 33.010],
    [49.49, 137.29, 14.760],
    [0.440, 27.210, 169.83]
]).astype(np.float16)

M_full_backlight_inv = np.array([
    [ 0.01356474, -0.0059699 , -0.00211774],
    [-0.00497165,  0.00959956,  0.00013204],
    [ 0.00076141, -0.00152257,  0.00587257]
]).astype(np.float16)

M_low_backlight = np.array([
    [4.61, 3.35, 1.78],
    [2.48, 7.16, 0.79],
    [0.28, 1.93, 8.93]
]).astype(np.float16)

M_low_backlight_inv = np.array([
    [ 0.28469582, -0.12078639, -0.04606241],
    [-0.10000966,  0.18550721,  0.00352368],
    [ 0.012688  , -0.03630557,  0.11266481]
]).astype(np.float16)

M_CAT02 = np.array([
    [ 0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975,  0.0061],
    [ 0.0030, 0.0136,  0.9834]
]).astype(np.float16)

M_CAT02_inv = np.array([
    [ 1.096124, -0.278869, 0.182745],
    [ 0.454369,  0.473533, 0.072098],
    [-0.009628, -0.005698, 1.015326]
]).astype(np.float16)

M_H = np.array([
    [ 0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340,  0.04641],
    [ 0.00000, 0.00000,  1.00000]
]).astype(np.float16)

M_H_inv = np.array([
    [1.910197, -1.112124,  0.201908],
    [0.370950,  0.629054, -0.000008],
    [0.000000,  0.000000,  1.000000]
]).astype(np.float16)

M_HPE_to_A_xab = np.array([
    [  2,      1, 1/20],
    [  1, -12/11, 1/11],
    [1/9,    1/9, -2/9]
]).astype(np.float16)

M_HPE_to_A_xab_inv = np.array([
    [ 0.32786885,  0.32145403,  0.20527441],
    [ 0.32786885, -0.63506771, -0.18602994],
    [ 0.32786885, -0.15680684, -4.49037776]
]).astype(np.float16)

gamma_full_backlight = np.array([
    2.4767,
    2.4286,
    2.3792
]).astype(np.float16)

gamma_full_backlight_inv = np.array([ 
    0.40376307,  
    0.41175986,  
    0.42030935
]).astype(np.float16)

gamma_low_backlight = np.array([
    2.2212,
    2.1044,
    2.1835
]).astype(np.float16)

gamma_low_backlight_inv = np.array([ 
    0.4502071 ,  
    0.47519483,  
    0.45798031
]).astype(np.float16)


def normalizeRGB(RGB):
    RGB_normalized = np.divide(RGB, 255)
    return RGB_normalized 

def denormalizeRGB(RGB_normalized):
    RGB = np.multiply(RGB_normalized, 255)
    return RGB 

def gammaCorrection_full_backlight(RGB_normalized):
    RGB_linear_full_backlight = np.power(RGB_normalized, gamma_full_backlight)
    return RGB_linear_full_backlight

def gammaCorrection_low_backlight(RGB_normalized):
    RGB_linear_low_backlight = np.power(RGB_normalized, gamma_low_backlight)
    return RGB_linear_low_backlight

def inverse_gammaCorrection_full_backlight(RGB_linear_full_backlight):
    abs_RGB_linear_full_backlight = np.abs(RGB_linear_full_backlight)
    RGB_normalized = np.power(abs_RGB_linear_full_backlight, gamma_full_backlight_inv)
    return RGB_normalized

def inverse_gammaCorrection_low_backlight(RGB_linear_low_backlight):
    abs_RGB_linear_low_backlight = np.abs(RGB_linear_low_backlight)
    RGB_normalized = np.power(abs_RGB_linear_low_backlight, gamma_low_backlight_inv)
    return RGB_normalized 

def RGB_linear_to_XYZ_full_backlight(RGB_linear):
    XYZ = ut.dot(M=M_full_backlight, img=RGB_linear)
    return XYZ 

def XYZ_to_RGB_linear_full_backlight(XYZ):
    RGB_linear_full_backlight = ut.dot(M=M_full_backlight_inv, img=XYZ)
    return RGB_linear_full_backlight

def XYZ_to_RGB_linear_low_backlight(XYZ):
    RGB_linear_low_backlight = ut.dot(M=M_low_backlight_inv, img=XYZ)
    return RGB_linear_low_backlight 

def RGB_linear_to_XYZ_low_backlight(RGB_linear):
    XYZ = ut.dot(M=M_low_backlight, img=RGB_linear)
    return XYZ 

def XYZ_to_HPE(XYZ):
    HPE = ut.dot(M=M_H, img=XYZ)
    return HPE 

def HPE_to_XYZ(HPE):
    XYZ = ut.dot(M=M_H_inv, img=HPE)
    return XYZ 

def XYZ_to_LMS(XYZ):
    LMS = ut.dot(M=M_CAT02, img=XYZ)
    return LMS 

def LMS_to_XYZ(LMS):
    XYZ = ut.dot(M=M_CAT02_inv, img=LMS)
    return XYZ 

def HPE_to_Aab(HPE, N_bb):
    Aab = ut.dot(M=M_HPE_to_A_xab, img=HPE)
    Aab[:, :, 0] -= 0.305
    Aab[:, :, 0] *= N_bb 
    return Aab 

def Aab_to_HPE(Aab, N_bb):
    Aab[:, :, 0] /= N_bb 
    Aab[:, :, 0] += 0.305 
    HPE = ut.dot(M=M_HPE_to_A_xab_inv, img=Aab)
    return HPE 

def LMS_to_HPE(LMS):
    XYZ = LMS_to_XYZ(LMS=LMS)
    HPE = XYZ_to_HPE(XYZ=XYZ)
    return HPE 

def HPE_to_LMS(HPE):
    XYZ = HPE_to_XYZ(HPE=HPE)
    LMS = XYZ_to_LMS(XYZ=XYZ)
    return LMS 

def dimBacklight(RGB):
    RGB_linear_low_backlight = gammaCorrection_low_backlight(RGB_normalized=RGB)
    XYZ_low_backlight = RGB_linear_to_XYZ_low_backlight(RGB_linear=RGB_linear_low_backlight)
    RGB_linear_dim = XYZ_to_RGB_linear_full_backlight(XYZ=XYZ_low_backlight)
    RGB_dim = inverse_gammaCorrection_full_backlight(RGB_linear_full_backlight=RGB_linear_dim)
    return RGB_dim