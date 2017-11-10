import numpy as np 
import util as ut 

class CIECAM02:
    M_CAT02 = np.array([
        [ 0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975,  0.0061],
        [ 0.0030, 0.0136,  0.9834]
    ])

    M_CAT02_inv = np.array([
        [ 1.096124  -0.278869, 0.182745],
        [ 0.454369,  0.473533, 0.072098],
        [-0.009628, -0.005698, 1.015326]
    ])

    M_H = np.array([
        [ 0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340,  0.04641],
        [ 0.00000, 0.00000,  1.00000]
    ])

    c = np.array([
        0.690, # avg
        0.590, # dim
        0.525  # dark
    ])

    N_c = np.array([
        1.00, # avg
        0.95, # dim
        0.80  # dark
    ])

    F = np.array([
        1.0, # avg
        0.9, # dim
        0.8  # dark
    ])

    RGB_w = np.array([1, 1, 1])


    def __init__(self, XYZ_w, L_A=60, Y_b=25, s_R=0):
        self.XYZ_w = XYZ_w
        self.L_A = L_A
        self.Y_b = Y_b
        self.s_R = s_R
        self.n = None
        self.N_cb = None
        self.N_cc = None
        self.z = None
        self.F_L = None

        init(XYZ_w, L_A, Y_b, s_R)

        def init(XYZ_w, L_A, Y_b, s_R):
            self.n = Y_b / XYZ_w[1]
            self.N_cb = 0.725 * (1 / self.n)**0.2
            self.N_bb = self.N_cb
            self.z = 1.48 + self.n**0.5
            return

    def process(self, XYZ):
            return


    def determineParameters(self):
        if self.s_R > 0.2:
            return (self.c[0], self.N_c[0], self.F[0])
        if self.s_R > 0:
            return (self.c[1], self.N_c[1], self.F[1])
        if self.s_R == 0:
            return (self.c[2], self.N_c[2], self.F[2])
        return

    def chromaticTransfrom(self, XYZ, XYZ_w, F):
        def spaceConversion(XYZ, XYZ_w, M):
            LMS = ut.dot(M=M, img=XYZ)
            LMS_w = ut.dot(M=M, img=XYZ_w)
            return (LMS, LMS_w)

        def degreeOfAdaptation(F, L_A):
            D = F * (1 - 1/3.6 * np.exp(x=-(L_A+42)/92))
            return D

        def gainControl(LMS, LMS_w, D):
            LMS_w_t = (100/LMS_w - 1) * D + 1
            LMS_c = ut.mul(v=LMS_w_t, img=LMS)
            return LMS_c

        LMS, LMS_w = spaceConversion(XYZ=XYZ, XYZ_w=XYZ_w, M=self.M_CAT02)
        D = degreeOfAdaptation(F=F, L_A=self.L_A)
        LMS_c = gainControl(LMS=LMS, LMS_w=LMS_w, D=D)

        return LMS_c


    def compression(self, LMS_c, F_L):
        def luminanceLevelAdaptationFactor(L_A):
            k = 1 / (5 * L_A + 1)
            F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4)**2 * (5 * L_A)**(1 / 3)
            return F_L

        def spaceConversion(LMS_c, M_CAT02_inv, M_H):
            XYZ_c = ut.dot(M=M_CAT02_inv, img=LMS_c)
            LMS_prime = ut.dot(M=M_H, img=XYZ_c)
            return LMS_prime

        def nonLinearCompression(LMS_prime, F_L):
            t = (F_L * LMS_prime / 100)**0.42
            LMS_a_prime = (400 * t) / (27.13 + t) + 0.1
            return LMS_a_prime

        F_L = luminanceLevelAdaptationFactor(L_A=self.L_A)
        LMS_prime = spaceConversion(LMS_c=LMS_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H)
        LMS_a_prime = nonLinearCompression(LMS_prime=LMS_prime, F_L=F_L)

        return LMS_a_prime

    def opponentColorConversion(self, LMS_a_prime):
        M_C = np.array([
            [ 1, -1,  0],
            [ 0,  1, -1],
            [-1,  0,  1]
        ])
        M_Aab = np.array([
            
        ])

        C = ut.dot(M=M_C, img=LMS_a_prime)
