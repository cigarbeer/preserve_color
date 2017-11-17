import image as im
import numpy as np 
from deviceCharacteristicModel import DeviceCharacteristicModel
from CIECAM02 import CIECAM02
import postGamutMapping as pgm 
import util as ut
import convert as cvt 

import sys 


if __name__ == '__main__':
    # read image
    fileName = './images.jpeg'
    img = im.readImg(fileName)

    c, N_c, F = (0.69, 1, 1)
    
