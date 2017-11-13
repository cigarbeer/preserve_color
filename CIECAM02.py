import numpy as np 
import util as ut 

class CIECAM02:
    M_CAT02 = np.array([
        [ 0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975,  0.0061],
        [ 0.0030, 0.0136,  0.9834]
    ])

    M_CAT02_inv = np.array([
        [ 1.096124, -0.278869, 0.182745],
        [ 0.454369,  0.473533, 0.072098],
        [-0.009628, -0.005698, 1.015326]
    ])

    M_H = np.array([
        [ 0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340,  0.04641],
        [ 0.00000, 0.00000,  1.00000]
    ])

    M_H_inv = np.array([
        [1.910197, -1.112124,  0.201908],
        [0.370950,  0.629054, -0.000008],
        [0.000000,  0.000000,  1.000000]
    ])

    M_Aab_to_LMS_a_prime = np.array([
        [0.32787,  0.32145,  0.20527], 
        [0.32787, -0.63507, -0.18603],
        [0.32787, -0.15681, -4.49038]
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


    def __init__(self):
        return

    def forward(self, XYZ, XYZ_white, L_A, Y_b, s_R):
        c, N_c, F = self.determineParameters(s_R)

        n = self.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
        N_bb = self.N_bb(n=n)
        N_cb = self.N_cb(n=n)
        z = self.z(n=n)

        LMS_c = self.chromaticTransfrom(XYZ=XYZ, XYZ_white=XYZ_white, M_CAT02=self.M_CAT02, F=F, L_A=L_A)
        LMS_a_prime = self.compression(LMS=LMS_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H, L_A=L_A)
        A, a, b = self.opponentColorConversion(LMS_a_prime=LMS_a_prime, N_bb=N_bb)

        LMS_white_c = self.chromaticTransfrom(XYZ=XYZ_white, XYZ_white=XYZ_white, M_CAT02=self.M_CAT02, F=F, L_A=L_A)
        LMS_white_a_prime = self.compression(LMS=LMS_white_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H, L_A=L_A)
        A_white, a_white, b_white = self.opponentColorConversion(LMS_a_prime=LMS_white_a_prime, N_bb=N_bb)

        h, J, C = self.computePerceptualAttributes(A=A, a=a, b=b, A_white=A_white, c=c, N_c=N_c, LMS_a_prime=LMS_a_prime, n=n, z=z, N_cb=N_cb, Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
        
        return (h, J, C)

    def determineParameters(self, s_R):
        if s_R > 0.2:
            return (self.c[0], self.N_c[0], self.F[0])
        if s_R > 0:
            return (self.c[1], self.N_c[1], self.F[1])
        if s_R == 0:
            return (self.c[2], self.N_c[2], self.F[2])
        return

    def chromaticTransfrom(self, XYZ, XYZ_white, M_CAT02, F, L_A):
        LMS = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ, M_CAT02=M_CAT02)
        LMS_white = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ_white, M_CAT02=M_CAT02)
        D = self.degreeOfAdaptation(F=F, L_A=L_A)
        LMS_c = self.gainControl(LMS=LMS, LMS_white=LMS_white, D=D)
        return LMS_c

    def compression(self, LMS, M_CAT02_inv, M_H, L_A):
        F_L = self.luminanceLevelAdaptationFactor(L_A=L_A)
        XYZ = self.spaceConversion_LMS_to_XYZ(LMS=LMS, M_CAT02_inv=M_CAT02_inv)
        LMS_prime = self.spaceConversion_XYZ_to_HPE(XYZ=XYZ, M_H=M_H)
        LMS_a_prime = self.nonLinearCompression(LMS_prime=LMS_prime, F_L=F_L)
        return LMS_a_prime

    def opponentColorConversion(self, LMS_a_prime, N_bb):
        A = self.achromaticResponse(LMS_a_prime=LMS_a_prime, N_bb=N_bb)
        a = LMS_a_prime[:, :, 0] - 12/11*LMS_a_prime[:, :, 1] + 1/11*LMS_a_prime[:, :, 2]
        b = 1/9*LMS_a_prime[:, :, 0] + 1/9*LMS_a_prime[:, :, 1] - 2/9*LMS_a_prime[:, :, 2]
        return (A, a, b)

    def computePerceptualAttributes(self, A, a, b, A_white, c, N_c, LMS_a_prime, n, z, N_cb, Y_b, Y_w):
        h = self.hue(a=a, b=b)
        J = self.lightness(A=A, A_white=A_white, c=c, z=z)

        n = self.n(Y_b, Y_w)
        e_t = self.e_t(n)
        t = self.t(N_c=N_c, N_cb=N_cb, e_t=e_t, a=a, b=b, LMS_a_prime=LMS_a_prime)

        C = self.chroma(t=t, J=J, n=n)
        return (h, J, C)


    def spaceConversion_XYZ_to_LMS(self, XYZ, M_CAT02):
        LMS = ut.dot(M=M_CAT02, img=XYZ)
        return LMS

    def degreeOfAdaptation(self, F, L_A):
        D = F * (1 - 1/3.6 * np.exp(-(L_A+42)/92))
        return D

    def gainControl(self, LMS, LMS_white, D):
        LMS_white_t = (100/LMS_white - 1) * D + 1
        LMS_c = ut.mul(v=LMS_white_t, img=LMS)
        return LMS_c

    def luminanceLevelAdaptationFactor(self, L_A):
        k = 1 / (5 * L_A + 1)
        F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4)**2 * (5 * L_A)**(1 / 3)
        return F_L

    def spaceConversion_LMS_to_XYZ(self, LMS, M_CAT02_inv):
        XYZ = ut.dot(M=M_CAT02_inv, img=LMS)
        return XYZ

    def nonLinearCompression(self, LMS_prime, F_L):
        t = (F_L * LMS_prime / 100)**0.42
        LMS_a_prime = (400 * t) / (27.13 + t) + 0.1
        return LMS_a_prime

    def spaceConversion_XYZ_to_HPE(self, XYZ, M_H):
        LMS_prime = ut.dot(M=M_H, img=XYZ)
        return LMS_prime

    def hue(self, a, b):
        h = ut.angle_degree(a, b)
        return h

    def lightness(self, A, A_white, c, z):
        J = 100 * (A / A_white)**(c * z)
        return J

    def chroma(self, t, J, n):
        C = t**0.9 * (0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73
        return C

    def achromaticResponse(self, LMS_a_prime, N_bb):
        A = (2*LMS_a_prime[:, :, 0] + LMS_a_prime[:, :, 1] + 1/20*LMS_a_prime[:, :, 2] - 0.305) * N_bb
        return A        

    def inverse(self, h, J, C, XYZ_white, L_A, Y_b, s_R):
        c, N_c, F = self.determineParameters(s_R)
        n = self.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
        N_bb = self.N_bb(n=n)
        N_cb = self.N_cb(n=n)
        z = self.z(n=n)
        F_L = self.luminanceLevelAdaptationFactor(L_A=L_A)
        D = self.degreeOfAdaptation(F=F, L_A=L_A)
        LMS_white = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ_white, M_CAT02=self.M_CAT02)
        LMS_white_c = self.chromaticTransfrom(XYZ=XYZ_white, XYZ_white=XYZ_white, M_CAT02=self.M_CAT02, F=F, L_A=L_A)
        LMS_white_a_prime = self.compression(LMS=LMS_white_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H, L_A=L_A)
        A_white = self.achromaticResponse(LMS_a_prime=LMS_white_a_prime, N_bb=N_bb)
        t = self.t_inv(C, J, n)
        e_t = self.e_t(h)
        A = self.A_inv(A_white=A_white, J=J, c=c, z=z)
        a, b = self.ab_inv(t=t, e_t=e_t, h=h, A=A, N_c=N_c, N_cb=N_cb, N_bb=N_bb)
        LMS_a_prime = self.LMS_a_prime_inv(A=A, a=a, b=b, N_bb=N_bb, M_Aab_to_LMS_a_prime=self.M_Aab_to_LMS_a_prime)
        LMS_prime = self.LMS_prime_inv(LMS_a_prime=LMS_a_prime, F_L=F_L)
        XYZ_c = self.spaceConversion_HPE_to_XYZ(LMS_prime=LMS_prime, M_H_inv=self.M_H_inv)
        LMS_c = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ_c, M_CAT02=self.M_CAT02)
        LMS = self.gainControl_inv(LMS=LMS_c, LMS_white=LMS_white, D=D)
        XYZ_enhanced = self.spaceConversion_LMS_to_XYZ(LMS=LMS, M_CAT02_inv=self.M_CAT02_inv)
        return XYZ_enhanced

    def t(self, N_c, N_cb, e_t, a, b, LMS_a_prime):
        result = (50000/13 * N_c * N_cb * e_t * (a**2 + b**2)**0.5) / (LMS_a_prime[:, :, 0] + LMS_a_prime[:, :, 1] + 21/20*LMS_a_prime[:, :, 2])
        return result
    
    def t_inv(self, C, J, n):
        result = (C / ((0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73))**(1/0.9)
        return result

    def e_t(self, h):
        result = 1/4 * (np.cos(np.pi/180 * h + 2) + 3.8)
        return result

    def A_inv(self, A_white, J, c, z):
        result = (0.01 * J)**(1 / (c *z)) * A_white
        return result

    def ab_inv(self, t, e_t, h, A, N_c, N_cb, N_bb):
        p1 = (50000/13 * N_c * N_cb * e_t) / t 
        p2 = (A / N_bb) + 0.305
        p3 = 21 / 20
        
        sin_h = np.sin(h)
        cos_h = np.cos(h)
        tan_h = np.tan(h)
        cot_h = 1 / tan_h

        singtcos = np.abs(sin_h) > np.abs(cos_h)
        sinltcos = ~singtcos

        # sin > cos
        p4 = p1 / sin_h
        cb_gt = (p2 * (2 + p3) * (460/1403)) / (p4 + (2 + p3) * (220/1403) * cot_h - (27/1403) + p3 * (6300/1403))
        ca_gt = cb_gt * cot_h
        # sin < cos
        p5 = p1 / cos_h
        ca_lt = (p2 * (2 + p3) * (460/1403)) / (p5 + (2 + p3) * (220/1403) - ((27/1403) - p3 * (6300/1403)) * tan_h)
        cb_lt = ca_lt * tan_h

        a = np.zeros(h.shape)
        b = np.zeros(h.shape)

        a[singtcos] = ca_gt[singtcos]
        a[sinltcos] = ca_lt[sinltcos]

        b[singtcos] = cb_gt[singtcos]
        b[sinltcos] = cb_lt[sinltcos]

        return (a, b)

    def LMS_a_prime_inv(self, A, a, b, N_bb, M_Aab_to_LMS_a_prime):
        x = A / N_bb + 0.305
        xab = np.stack([x, a, b], axis=-1)
        result = ut.dot(M=M_Aab_to_LMS_a_prime, img=xab)
        return result

    def LMS_prime_inv(self, LMS_a_prime, F_L):
        result = (100 / F_L) * ((27.13*(LMS_a_prime-0.1)) / (400 - (LMS_a_prime-0.1)))**(1 / 0.42)
        return result

    def spaceConversion_HPE_to_XYZ(self, LMS_prime, M_H_inv):
        XYZ = ut.dot(M=M_H_inv, img=LMS_prime)
        return XYZ


    def gainControl_inv(self, LMS, LMS_white, D):
        c = 1 / ((100/LMS_white - 1) * D + 1)
        result = c * LMS
        return result
  
    def n(self, Y_b, Y_w):
        return Y_b / Y_w

    def N_cb(self, n):
        return 0.725 * (1 / n)**0.2

    def N_bb(self, n):
        return self.N_cb(n)

    def z(self, n):
        return 1.48 + n**0.5
