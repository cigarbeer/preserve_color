import image as im
import numpy as np 
import deviceCharacteristicModel as dchm
from CIECAM02 import CIECAM02
import util as ut

import sys 


# if __name__ == '__main__':


fileName = './images.jpeg'
RGB = im.readImg(fileName)
XYZ_f, XYZ_l, XYZ_w_f, XYZ_w_l = dchm.deviceCharacteristicModel(RGB)
cie = CIECAM02(XYZ_w=XYZ_w_f, L_A=60, Y_b=25, s_R=1)




c, N_c, F = cie.determineParameters()
LMS_c, LMS_w_c = cie.chromaticTransfrom(XYZ=XYZ_f, XYZ_w=XYZ_w_f, F=F)
LMS_a_prime, LMS_w_a_prime = cie.compression(LMS_c=LMS_c, LMS_w_c=LMS_w_c)
A, a, b, A_w = cie.opponentColorConversion(LMS_a_prime=LMS_a_prime, LMS_w_a_prime=LMS_w_a_prime)
h, J, C = cie.computePerceptualAttributes(A=A, a=a, b=b, A_w=A_w, c=c, N_c=N_c, LMS_a_prime=LMS_a_prime)