import numpy as np
import util as ut
import deviceCharacteristic as dch 


def deviceCharacteristicModel(RGB):
    def normalizeRGB(RGB):
        piexlSum = np.sum(a=RGB, axis=0)
        RGB_n = np.divide(RGB, piexlSum)
        return RGB_n 

    def gammaCorrection(RGB_n, gamma):
        RGB_l = ut.pow(exponent=gamma, img=RGB_n)
        return RGB_l

    def spaceConversion(RGB_l, M):
        XYZ = ut.dot(M=M, img=RGB_l)
        return XYZ

    RGB_n = normalizeRGB(RGB)
    RGB_l_f = gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_f)
    RGB_l_l = gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_l)
    XYZ_f = spaceConversion(RGB_l=RGB_l_f, M=dch.M_f)
    XYZ_l = spaceConversion(RGB_l=RGB_l_l, M=dch.M_l)
    RGB_l_w = np.array([1, 1, 1])
    XYZ_f_w = spaceConversion(RGB_l=RGB_l_w, M=dch.M_f)
    XYZ_l_w = spaceConversion(RGB_l=RGB_l_w, M=dch.M_l)

    return (XYZ_f, XYZ_l, XYZ_f_w, XYZ_l_w)