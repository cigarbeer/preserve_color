import numpy as np
import util as ut
import deviceCharacteristic as dch 

class DeviceCharacteristicModel:
    def __init__(self):
        self.RGB_normalized = None
        self.RGB_linear_full_backlight = None
        self.RGB_linear_low_backlight = None
        self.XYZ_full_backlight = None
        self.XYZ_low_backlight = None
        self.RGB_white_linear = np.array([1, 1, 1]).reshape((1, -1))
        self.XYZ_white_full_backlight = None
        self.XYZ_white_low_backlight = None

    def input(self, img):
        self.RGB_normalized = self.normalizeRGB(RGB=img)
        self.RGB_linear_full_backlight = self.gammaCorrection(RGB_normalized=self.RGB_normalized, gamma=dch.gamma_full_backlight)
        self.RGB_linear_low_backlight = self.gammaCorrection(RGB_normalized=self.RGB_normalized, gamma=dch.gamma_low_backlight)
        self.XYZ_full_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_linear_full_backlight, M=dch.M_full_backlight)
        self.XYZ_low_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_linear_low_backlight, M=dch.M_low_backlight)
        self.XYZ_white_full_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_white_linear, M=dch.M_full_backlight)
        self.XYZ_white_low_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_white_linear, M=dch.M_low_backlight)
        return self

    def normalizeRGB(self, RGB):
        RGB_n = np.divide(RGB, 255) 
        return RGB_n 

    def gammaCorrection(self, RGB_normalized, gamma):
        RGB_l = ut.pow(exponent=gamma, img=RGB_normalized)
        return RGB_l

    def spaceConversion_RGB_linear_to_XYZ(self, RGB_linear, M):
        XYZ = ut.dot(M=M, img=RGB_linear)
        return XYZ