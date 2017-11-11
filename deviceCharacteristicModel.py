import numpy as np
import util as ut
import deviceCharacteristic as dch 

def normalizeRGB(RGB):
    piexlSum = np.sum(a=RGB, axis=2)
    RGB_t = np.moveaxis(a=RGB, source=-1, destination=0)
    RGB_n_t = np.divide(RGB_t, piexlSum)
    RGB_n = np.moveaxis(a=RGB_n_t, source=0, destination=-1)
    return RGB_n 

def gammaCorrection(RGB_n, gamma):
    RGB_l = ut.pow(exponent=gamma, img=RGB_n)
    return RGB_l

def spaceConversion(RGB_l, M):
    XYZ = ut.dot(M=M, img=RGB_l)
    return XYZ

def deviceCharacteristicModel(RGB):
    RGB_n = normalizeRGB(RGB)
    RGB_l_f = gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_f)
    RGB_l_l = gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_l)
    XYZ_f = spaceConversion(RGB_l=RGB_l_f, M=dch.M_f)
    XYZ_l = spaceConversion(RGB_l=RGB_l_l, M=dch.M_l)
    RGB_w_l = np.array([1, 1, 1])
    XYZ_w_f = spaceConversion(RGB_l=RGB_w_l, M=dch.M_f)
    XYZ_w_l = spaceConversion(RGB_l=RGB_w_l, M=dch.M_l)

    return (XYZ_f, XYZ_l, XYZ_w_f, XYZ_w_l)