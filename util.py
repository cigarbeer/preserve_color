import numpy as np 
import time

def debug(f, flag=False):
    def d_f(*args, **kargs):
        print(f.__name__)
        st = time.time()
        res = f(*args, **kargs)
        ed = time.time()
        print('use:', (ed-st)*1000)
        return res

    def nd_f(*args, **kargs):
        print(f.__name__)
        return f(*args, **kargs)
    
    if flag:
        return d_f 

    return nd_f 



def dot(M, img):
    shape = img.shape
    img_2D = img.reshape((-1, shape[-1]))
    img_2D_t = np.moveaxis(a=img_2D, source=-1, destination=0)
    result_2D_t = np.dot(a=M, b=img_2D_t)
    result_2D = np.moveaxis(a=result_2D_t, source=0, destination=-1)
    result = result_2D.reshape(shape)
    return result
