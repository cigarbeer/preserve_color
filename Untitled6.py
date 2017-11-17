
# coding: utf-8

# In[1]:


import numpy as np 
import transform as tf
import equation as eq 
import convert as cvt 
import image as im
import postGamutMapping as pgm
np.seterr(all='print')


# In[4]:


get_ipython().run_cell_magic('timeit', '-n3 -r3', "img = im.readImg('./image.jpg')\nimg = img.astype(np.float32)\nRGB_normalized = cvt.normalizeRGB(RGB=img)\n\nRGB_linear_full_backlight = cvt.gammaCorrection_full_backlight(RGB_normalized=RGB_normalized)\n#RGB_linear_low_backlight = cvt.gammaCorrection_low_backlight(RGB_normalized=RGB_normalized)\n\n#XYZ_full_backlight = cvt.RGB_linear_to_XYZ_full_backlight(RGB_linear=RGB_linear_full_backlight)\n#XYZ_low_backlight = cvt.RGB_linear_to_XYZ_low_backlight(RGB_linear=RGB_linear_low_backlight)\n\nprint(RGB_linear_full_backlight.shape, RGB_linear_full_backlight.dtype)")


# In[4]:


print(img.shape, RGB_linear_full_backlight.dtype)


# In[4]:


L_A = 60
Y_b = 25
c = 0.69
N_c = 1
F = 1
RGB_linear_white = np.array([1, 1, 1]).reshape((1, 1, -1))


# In[5]:


D = eq.D(L_A=L_A, F=F)
F_L = eq.F_L(L_A=L_A)


# In[6]:


XYZ_white_full_backlight = cvt.RGB_linear_to_XYZ_full_backlight(RGB_linear=RGB_linear_white)
XYZ_white_full_backlight


# In[7]:


XYZ_white_low_backlight = cvt.RGB_linear_to_XYZ_low_backlight(RGB_linear=RGB_linear_white)
XYZ_white_low_backlight


# In[8]:


# forward
def forward(XYZ, XYZ_white, L_A, Y_b, c, N_c, F):
    LMS = cvt.XYZ_to_LMS(XYZ=XYZ)
    LMS_white = cvt.XYZ_to_LMS(XYZ=XYZ_white)
    print('LMS_white', LMS_white)
    
    LMS_c = tf.VKT_gainControl(LMS=LMS, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    LMS_white_c = tf.VKT_gainControl(LMS=LMS_white, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    print('LMS_white_c', LMS_white_c)
    
    n = eq.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
    print('n', n)
    
    N_bb = eq.N_bb(n=n)
    print('N_bb', N_bb)
    
    N_cb = eq.N_cb(n=n)
    print('N_cb', N_cb)
    
    z = eq.z(n=n)
    print('z', z)
    
    HPE = cvt.LMS_to_HPE(LMS=LMS_c)
    HPE_white = cvt.LMS_to_HPE(LMS=LMS_white_c)
    print('HPE_white', HPE_white)
    
    HPE_a = tf.nonlinearCompression(HPE=HPE, F_L=F_L)
    HPE_white_a = tf.nonlinearCompression(HPE=HPE_white, F_L=F_L)
    print('HPE_white_a', HPE_white_a)
    
    Aab = cvt.HPE_to_Aab(HPE=HPE_a, N_bb=N_bb)
    Aab_white = cvt.HPE_to_Aab(HPE=HPE_white_a, N_bb=N_bb)
    print('Aab_white', Aab_white)
    
    h = eq.h(a=Aab[:, :, 1], b=Aab[:, :, 2])
    
    J = eq.J(A=Aab[:, :, 0], A_white=Aab_white[0, 0, 0], c=c, z=z)
    
    e_t = eq.e_t(h=h)
    t = eq.t(N_c=N_c, N_cb=N_cb, e_t=e_t, a=Aab[:, :, 1], b=Aab[:, :, 2], HPE=HPE_a)
    C = eq.C(J=J, t=t, n=n)
    return (h, J, C)


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'h, J, C = forward(XYZ=XYZ_full_backlight, XYZ_white=XYZ_white_full_backlight, L_A=L_A, Y_b=Y_b, c=c, F=F, N_c=N_c)')


# In[10]:


# inverse
def inverse(h, J, C, XYZ_white, L_A, Y_b, c, N_c, F):
    LMS_white = cvt.XYZ_to_LMS(XYZ=XYZ_white)
    print('LMS_white', LMS_white)
    
    LMS_white_c = tf.VKT_gainControl(LMS=LMS_white, LMS_white=LMS_white, Y_w=XYZ_white[0, 0, 1], D=D)
    print('LMS_white_c', LMS_white_c)
    
    n = eq.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
    print('n', n)
    
    N_bb = eq.N_bb(n=n)
    print('N_bb', N_bb)
    
    N_cb = eq.N_cb(n=n)
    print('N_cb', N_cb)
    
    z = eq.z(n=n)
    print('z', z)
    
    HPE_white = cvt.LMS_to_HPE(LMS=LMS_white_c)
    print('HPE_white', HPE_white)
    
    HPE_white_a  =tf.nonlinearCompression(HPE=HPE_white, F_L=F_L)
    print('HPE_white_a', HPE_white_a)
    
    Aab_white = cvt.HPE_to_Aab(HPE=HPE_white_a, N_bb=N_bb)
    print('Aab_white', Aab_white)
    
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


# In[11]:


XYZ_enhanced = inverse(h=h, J=J, C=C, XYZ_white=XYZ_white_low_backlight, L_A=L_A, Y_b=Y_b, c=c, N_c=N_c, F=F)


# In[ ]:


RGB_linear_enhanced = cvt.XYZ_to_RGB_linear_low_backlight(XYZ=XYZ_enhanced)
RGB_normalized_enchanced = cvt.inverse_gammaCorrection_low_backlight(RGB_linear_low_backlight=RGB_linear_enhanced)


# In[ ]:


RGB_clipped = pgm.clipping(RGB=RGB_normalized_enchanced)
RGB_result = pgm.blend(RGB_clipped=RGB_clipped, RGB_origin=RGB_normalized, J=J, C=C)


# In[ ]:


im.showImg(cvt.dimBacklight(RGB=RGB_normalized)*255)


# In[ ]:


im.showImg(cvt.dimBacklight(RGB=RGB_result)*255)


# In[ ]:


p = np.arange(5*4*3).reshape((5, 4, 3))
q = np.arange(5*4).reshape((5, 4))
r = q.reshape(q.shape+(1,))
r*p

