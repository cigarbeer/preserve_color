import numpy as np
import util as ut

class DeviceCharacteristicModel:
    M_full_backlight = np.array([
        [95.57, 64.670, 33.010],
        [49.49, 137.29, 14.760],
        [0.440, 27.210, 169.83]
    ])

    M_full_backlight_inv = np.array([
        [ 0.01356474, -0.0059699 , -0.00211774],
        [-0.00497165,  0.00959956,  0.00013204],
        [ 0.00076141, -0.00152257,  0.00587257]
    ])

    M_low_backlight = np.array([
        [4.61, 3.35, 1.78],
        [2.48, 7.16, 0.79],
        [0.28, 1.93, 8.93]
    ])

    M_low_backlight_inv = np.array([
        [ 0.28469582, -0.12078639, -0.04606241],
        [-0.10000966,  0.18550721,  0.00352368],
        [ 0.012688  , -0.03630557,  0.11266481]
    ])

    gamma_full_backlight = np.array([
        2.4767,
        2.4286,
        2.3792
    ])

    gamma_full_backlight_inv = np.array([ 
        0.40376307,  
        0.41175986,  
        0.42030935
    ])

    gamma_low_backlight = np.array([
        2.2212,
        2.1044,
        2.1835
    ])

    gamma_low_backlight_inv = np.array([ 
        0.4502071 ,  
        0.47519483,  
        0.45798031
    ])


    @ut.debug
    def __init__(self):
        self.RGB_normalized = None
        self.RGB_linear_full_backlight = None
        self.RGB_linear_low_backlight = None
        self.XYZ_full_backlight = None
        self.XYZ_low_backlight = None
        self.RGB_white_linear = np.array([1, 1, 1]).reshape((1, 1, -1))
        self.XYZ_white_full_backlight = None
        self.XYZ_white_low_backlight = None

    @ut.debug
    def input(self, img):
        self.RGB_normalized = self.normalizeRGB(RGB=img)
        self.RGB_linear_full_backlight = self.gammaCorrection(RGB_normalized=self.RGB_normalized, gamma=self.gamma_full_backlight)
        self.RGB_linear_low_backlight = self.gammaCorrection(RGB_normalized=self.RGB_normalized, gamma=self.gamma_low_backlight)
        self.XYZ_full_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_linear_full_backlight, M=self.M_full_backlight)
        self.XYZ_low_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_linear_low_backlight, M=self.M_low_backlight)
        self.XYZ_white_full_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_white_linear, M=self.M_full_backlight)
        self.XYZ_white_low_backlight = self.spaceConversion_RGB_linear_to_XYZ(RGB_linear=self.RGB_white_linear, M=self.M_low_backlight)
        return 

    @ut.debug
    def normalizeRGB(self, RGB):
        RGB_normalized = np.divide(RGB, 255) 
        return RGB_normalized 

    @ut.debug
    def gammaCorrection(self, RGB_normalized, gamma):
        RGB_linear = ut.pow(exponent=gamma, img=RGB_normalized)
        return RGB_linear

    @ut.debug
    def inverse_gammaCorrection(self, RGB_linear, gamma_inv):
        absRGB_linear = np.abs(RGB_linear)
        RGB_normalized = np.power(absRGB_linear, gamma_inv)
        return RGB_normalized 

    @ut.debug
    def denormalizeRGB(self, RGB_normalized):
        RGB = np.multiply(RGB_normalized, 255)
        return RGB

    @ut.debug
    def spaceConversion_RGB_linear_to_XYZ(self, RGB_linear, M):
        XYZ = ut.dot(M=M, img=RGB_linear)
        return XYZ

    @ut.debug
    def spaceConversion_XYZ_to_RGB_linear(self, XYZ, M_inv):
        RGB_linear = ut.dot(M=M_inv, img=XYZ)
        return RGB_linear 