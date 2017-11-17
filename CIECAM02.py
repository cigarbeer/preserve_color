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

    M_LMS_prime_to_A_xab = np.array([
        [  2,      1, 1/20],
        [  1, -12/11, 1/11],
        [1/9,    1/9, -2/9]
    ])

    M_LMS_prime_to_A_xab_inv = np.array([
        [ 0.32786885,  0.32145403,  0.20527441],
        [ 0.32786885, -0.63506771, -0.18602994],
        [ 0.32786885, -0.15680684, -4.49037776]
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

    @ut.debug
    def __init__(self):
        return

    @ut.debug
    def forward(self, XYZ, XYZ_white, L_A, Y_b, s_R):
        c, N_c, F = self.determineParameters(s_R)

        n = self.n(Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
        N_bb = self.N_bb(n=n)
        N_cb = self.N_cb(n=n)
        z = self.z(n=n)

        LMS_c = self.chromaticTransfrom(XYZ=XYZ, XYZ_white=XYZ_white, M_CAT02=self.M_CAT02, F=F, L_A=L_A)
        LMS_a_prime = self.compression(LMS=LMS_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H, L_A=L_A)
        Aab = self.opponentColorConversion(LMS_prime=LMS_a_prime, M_LMS_prime_to_A_xab=self.M_LMS_prime_to_A_xab, N_bb=N_bb)

        LMS_white_c = self.chromaticTransfrom(XYZ=XYZ_white, XYZ_white=XYZ_white, M_CAT02=self.M_CAT02, F=F, L_A=L_A)
        LMS_white_a_prime = self.compression(LMS=LMS_white_c, M_CAT02_inv=self.M_CAT02_inv, M_H=self.M_H, L_A=L_A)
        Aab_white = self.opponentColorConversion(LMS_prime=LMS_white_a_prime, M_LMS_prime_to_A_xab=self.M_LMS_prime_to_A_xab, N_bb=N_bb)

        h, J, C = self.computePerceptualAttributes(A=A, a=a, b=b, A_white=A_white, c=c, N_c=N_c, LMS_a_prime=LMS_a_prime, n=n, z=z, N_cb=N_cb, Y_b=Y_b, Y_w=XYZ_white[0, 0, 1])
        
        return (h, J, C)

    
    @ut.debug
    def determineParameters(self, s_R):
        if s_R > 0.2:
            return (self.c[0], self.N_c[0], self.F[0])
        if s_R > 0:
            return (self.c[1], self.N_c[1], self.F[1])
        if s_R == 0:
            return (self.c[2], self.N_c[2], self.F[2])
        return

    @ut.debug
    def chromaticTransfrom(self, XYZ, XYZ_white, M_CAT02, F, L_A):
        LMS = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ, M_CAT02=M_CAT02)
        LMS_white = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ_white, M_CAT02=M_CAT02)
        D = self.degreeOfAdaptation(F=F, L_A=L_A)
        LMS_c = self.gainControl(LMS=LMS, LMS_white=LMS_white, D=D)
        return LMS_c

    @ut.debug
    def compression(self, LMS, M_CAT02_inv, M_H, L_A):
        F_L = self.luminanceLevelAdaptationFactor(L_A=L_A)
        XYZ = self.spaceConversion_LMS_to_XYZ(LMS=LMS, M_CAT02_inv=M_CAT02_inv)
        LMS_prime = self.spaceConversion_XYZ_to_HPE(XYZ=XYZ, M_H=M_H)
        LMS_a_prime = self.nonLinearCompression(LMS_prime=LMS_prime, F_L=F_L)
        return LMS_a_prime

    @ut.debug
    def opponentColorConversion(self, LMS_prime, M_LMS_prime_to_A_xab, N_bb):
        Aab = self.spaceConversion_HPE_to_Opponent(LMS_prime=LMS_prime, M_LMS_prime_to_A_xab=M_LMS_prime_to_A_xab, N_bb=N_bb)
        return Aab 

    @ut.debug
    def spaceConversion_HPE_to_Opponent(self, LMS_prime, M_LMS_prime_to_A_xab, N_bb):
        A_xab = ut.dot(M=M_LMS_prime_to_A_xab, img=LMS_prime)
        Aab = A_xab
        Aab[:, :, 0] -= 0.305
        Aab[:, :, 0] *= N_bb
        return Aab 

    @ut.debug
    def spaceConversion_Opponent_to_HPE(self, Aab, M_LMS_prime_to_A_xab_inv, N_bb):
        Aab[:, :, 0] /= N_bb
        Aab[:, :, 0] += 0.305
        LMS_prime = ut.dot(M=M_LMS_prime_to_A_xab_inv, img=Aab)
        return LMS_prime 


    @ut.debug
    def computePerceptualAttributes_deprecate(self, A, a, b, A_white, c, N_c, LMS_a_prime, n, z, N_cb, Y_b, Y_w):
        h = self.hue(a=a, b=b)
        J = self.lightness(A=A, A_white=A_white, c=c, z=z)

        n = self.n(Y_b, Y_w)
        e_t = self.e_t(h=h)
        t = self.t(N_c=N_c, N_cb=N_cb, e_t=e_t, a=a, b=b, LMS_prime=LMS_a_prime)

        C = self.chroma(t=t, J=J, n=n)
        return (h, J, C)

    def computePerceptualAttributes(self, Aab, A_white, LMS_prime, c, z, n, N_c, N_cb):
        h = self.hue(a=Aab[:, :, 1], b=Aab[:, :, 2])
        J = self.lightness(A=Aab[:, :, 0] , A_white=A_white, c=c, z=z)
        e_t = self.e_t(h=h)
        t = self.t(N_c=N_c, N_cb=N_cb, e_t=e_t, a=Aab[:, :, 1], b=Aab[:, :, 2], LMS_prime=LMS_prime)
        C = self.chroma(t=t, J=J, n=n)
        return (h, J, C)


    @ut.debug
    def spaceConversion_XYZ_to_LMS(self, XYZ, M_CAT02):
        LMS = ut.dot(M=M_CAT02, img=XYZ)
        return LMS

    @ut.debug
    def degreeOfAdaptation(self, F, L_A):
        D = F * (1 - 1/3.6 * np.exp(-(L_A+42)/92))
        return D

    @ut.debug
    def gainControl(self, LMS, LMS_white, D):
        LMS_white_t = (100/LMS_white - 1) * D + 1
        LMS_c = ut.mul(v=LMS_white_t, img=LMS)
        return LMS_c

    @ut.debug
    def luminanceLevelAdaptationFactor(self, L_A):
        k = 1 / (5 * L_A + 1)
        F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4)**2 * (5 * L_A)**(1 / 3)
        return F_L

    @ut.debug
    def spaceConversion_LMS_to_XYZ(self, LMS, M_CAT02_inv):
        XYZ = ut.dot(M=M_CAT02_inv, img=LMS)
        return XYZ

    @ut.debug
    def nonLinearCompression(self, LMS_prime, F_L):
        t = (F_L * LMS_prime / 100)**0.42
        LMS_a_prime = (400 * t) / (27.13 + t) + 0.1
        return LMS_a_prime

    @ut.debug
    def spaceConversion_XYZ_to_HPE(self, XYZ, M_H):
        LMS_prime = ut.dot(M=M_H, img=XYZ)
        return LMS_prime

    @ut.debug
    def hue(self, a, b):
        h = ut.angle_degree(a, b)
        return h

    @ut.debug
    def lightness(self, A, A_white, c, z):
        J = 100 * (A / A_white)**(c * z)
        return J

    @ut.debug
    def chroma(self, t, J, n):
        C = t**0.9 * (0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73
        return C

    # @ut.debug
    # def achromaticResponse(self, LMS_a_prime, N_bb):
    #     A = (2*LMS_a_prime[:, :, 0] + LMS_a_prime[:, :, 1] + 1/20*LMS_a_prime[:, :, 2] - 0.305) * N_bb
    #     return A        

    @ut.debug
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
        LMS_a_prime = self.LMS_a_prime_inv(A=A, a=a, b=b, N_bb=N_bb, M_LMS_prime_to_A_xab_inv=self.M_LMS_prime_to_A_xab_inv)
        LMS_prime = self.LMS_prime_inv(LMS_a_prime=LMS_a_prime, F_L=F_L)
        XYZ_c = self.spaceConversion_HPE_to_XYZ(LMS_prime=LMS_prime, M_H_inv=self.M_H_inv)
        LMS_c = self.spaceConversion_XYZ_to_LMS(XYZ=XYZ_c, M_CAT02=self.M_CAT02)
        LMS = self.gainControl_inv(LMS=LMS_c, LMS_white=LMS_white, D=D)
        XYZ_enhanced = self.spaceConversion_LMS_to_XYZ(LMS=LMS, M_CAT02_inv=self.M_CAT02_inv)
        return XYZ_enhanced

    @ut.debug
    def t(self, N_c, N_cb, e_t, a, b, LMS_prime):
        result = (50000/13 * N_c * N_cb * e_t * (a**2 + b**2)**0.5) / (LMS_prime[:, :, 0] + LMS_prime[:, :, 1] + 21/20*LMS_prime[:, :, 2])
        return result
    
    @ut.debug
    def t_inv(self, C, J, n):
        result = (C / ((0.01 * J)**0.5 * (1.64 - 0.29**n)**0.73))**(1/0.9)
        return result

    @ut.debug
    def e_t(self, h):
        result = 1/4 * (np.cos(np.pi/180 * h + 2) + 3.8)
        return result

    @ut.debug
    def A_inv(self, A_white, J, c, z):
        result = (0.01 * J)**(1 / (c *z)) * A_white
        return result

    @ut.debug
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

    @ut.debug
    def LMS_a_prime_inv(self, A, a, b, N_bb, M_LMS_prime_to_A_xab_inv):
        x = A / N_bb + 0.305
        xab = np.stack([x, a, b], axis=-1)
        result = ut.dot(M=M_LMS_prime_to_A_xab_inv, img=xab)
        return result

    @ut.debug
    def LMS_prime_inv(self, LMS_a_prime, F_L):
        result = (100 / F_L) * np.abs((27.13*(LMS_a_prime-0.1)) / (400 - (LMS_a_prime-0.1)))**(1 / 0.42)
        return result

    @ut.debug
    def spaceConversion_HPE_to_XYZ(self, LMS_prime, M_H_inv):
        XYZ = ut.dot(M=M_H_inv, img=LMS_prime)
        return XYZ


    @ut.debug
    def gainControl_inv(self, LMS, LMS_white, D):
        c = 1 / ((100/LMS_white - 1) * D + 1)
        result = c * LMS
        return result
  
    @ut.debug
    def n(self, Y_b, Y_w):
        return Y_b / Y_w

    @ut.debug
    def N_cb(self, n):
        return 0.725 * (1 / n)**0.2

    @ut.debug
    def N_bb(self, n):
        return self.N_cb(n)

    @ut.debug
    def z(self, n):
        return 1.48 + n**0.5
