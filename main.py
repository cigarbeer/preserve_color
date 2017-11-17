import image as im
import numpy as np 
from deviceCharacteristicModel import DeviceCharacteristicModel
from CIECAM02 import CIECAM02
import postGamutMapping as pgm 
import util as ut

import sys 


if __name__ == '__main__':
    # read image
    fileName = './images.jpeg'
    img = im.readImg(fileName)

    # device characteristic modeling
    dchm = DeviceCharacteristicModel()
    dchm.input(img=img)

    # ciecam02 modeling
    L_A_forward = 60
    Y_b_forward = 25 
    s_R_forward = 1
    ciecam = CIECAM02()
    h, J, C = ciecam.forward(
        XYZ=dchm.XYZ_full_backlight, 
        XYZ_white=dchm.XYZ_white_full_backlight, 
        L_A=L_A_forward, 
        Y_b=Y_b_forward, 
        s_R=s_R_forward
    )


    # ciecam02 inverse modeling
    L_A_inverse = 60
    Y_b_inverse = 25
    s_R_inverse = 1
    XYZ_enhanced = ciecam.inverse(
        h=h,
        J=J,
        C=C,
        XYZ_white=dchm.XYZ_white_low_backlight,
        L_A=L_A_inverse,
        Y_b=Y_b_inverse,
        s_R=s_R_inverse
    )

    RGB_linear_enhanced = dchm.spaceConversion_XYZ_to_RGB_linear(XYZ=XYZ_enhanced, M_inv=dchm.M_low_backlight_inv)
    RGB_enhanced = dchm.inverse_gammaCorrection(RGB_linear=RGB_linear_enhanced, gamma_inv=dchm.gamma_low_backlight_inv)

    RGB_clipped = pgm.clipping(RGB=RGB_enhanced)

    RGB_linear_origin = dchm.spaceConversion_XYZ_to_RGB_linear(XYZ=dchm.XYZ_full_backlight, M_inv=dchm.M_low_backlight_inv)
    RGB_origin = dchm.inverse_gammaCorrection(RGB_linear=RGB_linear_origin, gamma_inv=dchm.gamma_low_backlight_inv)

    J = J.reshape(J.shape+(1,))
    C = C.reshape(C.shape+(1,))
    # print('JC.shape', (J*C).shape)
    # print('RGB_clipped.shape', RGB_clipped.shape)
    # print('RGB_origin.shape', RGB_origin.shape)
    RGB_blend = pgm.blend(RGB_clipped=RGB_clipped, RGB_origin=RGB_origin, J=J, C=C)
    


