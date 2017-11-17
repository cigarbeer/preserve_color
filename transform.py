import numpy as np 

def VKT_gainControl(LMS, LMS_white, Y_w, D):
    c = (Y_w/LMS_white - 1) * D + 1
    LMS_c = np.multiply(c, LMS)
    return LMS_c 

def inverse_VKT_gainControl(LMS, LMS_white, Y_w, D):
    c = 1 / ((Y_w/LMS_white - 1) * D + 1)
    result = np.multiply(c, LMS)
    return result 


def nonlinearCompression(HPE, F_L):
    t = (F_L * HPE / 100)**0.42
    HPE_a = (400 * t) / (27.13 + t) + 0.1
    return HPE_a

def inverse_nonlinearCompression(HPE, F_L):
    result = (100 / F_L) * np.abs((27.13*(HPE-0.1)) / (400 - (HPE-0.1)))**(1 / 0.42)
    return result 