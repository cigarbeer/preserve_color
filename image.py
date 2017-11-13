import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def readImg(fileName):
    img = cv2.imread(filename=fileName, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    # img = np.moveaxis(a=img, source=-1, destination=0)
    return img

def saveImg(img, fileName):
    # img = np.moveaxis(a=img, source=0, destination=-1)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=fileName, img=img)
    return

def showImg(img):
    # img = np.moveaxis(a=img, source=0, destination=-1)
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    plt.imshow(X=img)
    plt.show()
    return


