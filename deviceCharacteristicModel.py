import numpy as np
import util as ut
import deviceCharacteristic as dch 


def deviceCharacteristicModel(RGB):
    RGB_n = ut.normalizeRGB(RGB)
    RGB_l_f = ut.gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_f)
    RGB_l_l = ut.gammaCorrection(RGB_n=RGB_n, gamma=dch.gamma_l)
    XYZ_f = ut.RGB_ltoXYZ(RGB_l_f, M=dch.M_f)
    XYZ_l = ut.RGB_ltoXYZ(RGB_l_l, M=dch.M_l)
    RGB_l_w = np.array([1, 1, 1])
    XYZ_f_w = ut.RGB_ltoXYZ(RGB_l=RGB_l_w, M=dch.M_f)
    XYZ_l_w = ut.RGB_ltoXYZ(RGB_l=RGB_l_w, M=dch.M_l)
    return (XYZ_f, XYZ_l, XYZ_f_w, XYZ_l_w)