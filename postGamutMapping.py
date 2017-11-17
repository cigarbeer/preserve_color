import numpy as np 
import util as ut 

@ut.debug
def clipping(RGB):
    RGBgt1 = RGB > 1
    RGBlt0 = RGB < 0
    RGB[RGBgt1] = 1
    RGB[RGBlt0] = 0
    return RGB 

@ut.debug
def blend(RGB_clipped, RGB_origin, J, C):
    JC = J * C / 10000
    JC = JC.reshape(JC.shape+(1,))
    RGB_blended = (1 - JC) * RGB_clipped + JC * RGB_origin
    return RGB_blended
