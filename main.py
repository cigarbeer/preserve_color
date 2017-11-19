import image as im 
import util as ut 
import convert as cvt 
import transform as tf 
import equation as eq 
import postGamutMapping as pgm 

import numpy as np 
import sys 

L_A = np.float16(60)
Y_b = np.float16(25)
F_L = eq.F_L(L_A=L_A)
RGB_linear_white = np.array([1, 1, 1]).reshape((1, 1, -1)).astype(np.float16)
XYZ_white_full_backlight = cvt.RGB_linear_to_XYZ_full_backlight(RGB_linear=RGB_linear_white)
XYZ_white_low_backlight = cvt.RGB_linear_to_XYZ_low_backlight(RGB_linear=RGB_linear_white)


def deviceCharacteriscModel(RGB_normalized):
    RGB_linear_full_backlight = cvt.gammaCorrection_full_backlight(RGB_normalized=RGB_normalized)
    XYZ_full_backlight = cvt.RGB_linear_to_XYZ_full_backlight(RGB_linear=RGB_linear_full_backlight)
    return XYZ_full_backlight

def determineParameters(flag):
    if flag == 1:
        return np.array([0.69, 1, 1]).astype(np.float16)
    if flag == 2:
        return np.array([0.59, 0.9, 0.9]).astype(np.float16)
    if flag == 3:
        return np.array([0.525, 0.8, 0.8]).astype(np.float16)


def forward(XYZ, XYZ_white, L_A, Y_b, c, N_c, F):
    LMS = cvt.XYZ_to_LMS(XYZ=XYZ)
    LMS_white = cvt.XYZ_to_LMS(XYZ=XYZ_white)
    # print('LMS_white', LMS_white)
    
    LMS_c = tf.VKT_gainControl(LMS=LMS, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    LMS_white_c = tf.VKT_gainControl(LMS=LMS_white, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    # print('LMS_white_c', LMS_white_c)
    
    n = eq.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
    # print('n', n)
    
    N_bb = eq.N_bb(n=n)
    # print('N_bb', N_bb)
    
    N_cb = eq.N_cb(n=n)
    # print('N_cb', N_cb)
    
    z = eq.z(n=n)
    # print('z', z)
    
    HPE = cvt.LMS_to_HPE(LMS=LMS_c)
    HPE_white = cvt.LMS_to_HPE(LMS=LMS_white_c)
    # print('HPE_white', HPE_white)
    
    HPE_a = tf.nonlinearCompression(HPE=HPE, F_L=F_L)
    HPE_white_a = tf.nonlinearCompression(HPE=HPE_white, F_L=F_L)
    # print('HPE_white_a', HPE_white_a)
    
    Aab = cvt.HPE_to_Aab(HPE=HPE_a, N_bb=N_bb)
    Aab_white = cvt.HPE_to_Aab(HPE=HPE_white_a, N_bb=N_bb)
    # print('Aab_white', Aab_white)
    
    h = eq.h(a=Aab[:, :, 1], b=Aab[:, :, 2])
    
    J = eq.J(A=Aab[:, :, 0], A_white=Aab_white[0, 0, 0], c=c, z=z)
    
    e_t = eq.e_t(h=h)
    t = eq.t(N_c=N_c, N_cb=N_cb, e_t=e_t, a=Aab[:, :, 1], b=Aab[:, :, 2], HPE=HPE_a)
    C = eq.C(J=J, t=t, n=n)
    return (h, J, C)


# inverse
def inverse(h, J, C, XYZ_white, L_A, Y_b, c, N_c, F):
    LMS_white = cvt.XYZ_to_LMS(XYZ=XYZ_white)
    # print('LMS_white', LMS_white)
    
    LMS_white_c = tf.VKT_gainControl(LMS=LMS_white, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    # print('LMS_white_c', LMS_white_c)
    
    n = eq.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
    # print('n', n)
    
    N_bb = eq.N_bb(n=n)
    # print('N_bb', N_bb)
    
    N_cb = eq.N_cb(n=n)
    # print('N_cb', N_cb)
    
    z = eq.z(n=n)
    # print('z', z)
    
    HPE_white = cvt.LMS_to_HPE(LMS=LMS_white_c)
    # print('HPE_white', HPE_white)
    
    HPE_white_a  =tf.nonlinearCompression(HPE=HPE_white, F_L=F_L)
    # print('HPE_white_a', HPE_white_a)
    
    Aab_white = cvt.HPE_to_Aab(HPE=HPE_white_a, N_bb=N_bb)
    # print('Aab_white', Aab_white)
    
    t = eq.t_inv(J=J, C=C, n=n)
    e_t = eq.e_t(h=h)
    A = eq.A_inv(J=J, A_white=Aab_white[0, 0, 0], c=c, z=z)
    a, b = eq.ab_inv(A=A, h=h, t=t, e_t=e_t, N_c=N_c, N_cb=N_cb, N_bb=N_bb)
    Aab = np.stack([A, a, b], axis=-1)
    HPE_a = cvt.Aab_to_HPE(Aab=Aab, N_bb=N_bb)
    HPE = tf.inverse_nonlinearCompression(HPE=HPE_a, F_L=F_L)
    LMS_c = cvt.HPE_to_LMS(HPE=HPE)
    LMS = tf.inverse_VKT_gainControl(LMS=LMS_c, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    XYZ = cvt.LMS_to_XYZ(LMS=LMS)
    return XYZ






if __name__ == '__main__':
    fileName = sys.argv[1]

    img = im.readImg(fileName=fileName)
    RGB_normalized = cvt.normalizeRGB(RGB=img)
    
    for i in range(1, 2):
        c, N_c, F = determineParameters(flag=i)
        D = eq.D(L_A=L_A, F=F)

        XYZ_full_backlight = deviceCharacteriscModel(RGB_normalized=RGB_normalized)
        h, J, C = forward(XYZ=XYZ_full_backlight, XYZ_white=XYZ_white_full_backlight, L_A=L_A, Y_b=Y_b, c=c, F=F, N_c=N_c)

        XYZ_enhanced = inverse(h=h, J=J, C=C, XYZ_white=XYZ_white_low_backlight, L_A=L_A, Y_b=Y_b, c=c, N_c=N_c, F=F)

        RGB_linear_enhanced = cvt.XYZ_to_RGB_linear_low_backlight(XYZ=XYZ_enhanced)
        RGB_normalized_enchanced = cvt.inverse_gammaCorrection_low_backlight(RGB_linear_low_backlight=RGB_linear_enhanced)
        RGB_normalized_clipped = pgm.clipping(RGB=RGB_normalized_enchanced)
        RGB_normalized_result = pgm.blend(RGB_clipped=RGB_normalized_clipped, RGB_origin=RGB_normalized, J=J, C=C)

        RGB_result = cvt.denormalizeRGB(RGB_normalized=RGB_normalized_result)

        fName = './result/'+fileName.split('/')[-1].split('.')[0]+'_flag'+str(i)
        im.saveImg(img=RGB_result, fileName=fName+'.png')
        im.saveImg(img=cvt.denormalizeRGB(cvt.dimBacklight(RGB=RGB_normalized_result)), fileName=fName+'_dim.png')

    im.saveImg(img=cvt.denormalizeRGB(cvt.dimBacklight(RGB=RGB_normalized)), fileName=fileName.split('.')[0]+'_dim.png')