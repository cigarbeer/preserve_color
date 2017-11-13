import image as im
import numpy as np 
from deviceCharacteristicModel import DeviceCharacteristicModel
from CIECAM02 import CIECAM02
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
    s_R_inverse = 0
    XYZ_enhanced = ciecam.inverse(
        h=h,
        J=J,
        C=C,
        XYZ_white=dchm.XYZ_white_low_backlight,
        L_A=L_A_inverse,
        Y_b=Y_b_inverse,
        s_R=s_R_inverse
    )

    







