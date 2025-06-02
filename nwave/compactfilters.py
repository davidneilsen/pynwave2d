import numpy as np
from scipy.linalg import solve_banded
from . import utils
from . filters import Filter1D
from . types import *

class CompactFilter(Filter1D):
    def __init__(self, x, apply_filter : FilterApply, ftype : FilterType, kim_eps=0.0, kim_kc=0.0):
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.kc = kim_kc
        self.eps = kim_eps
        self.n = n
        if ftype == FilterType.KFP4:
            self.ab, self.B = init_kim_filter(kim_kc, kim_eps, n)
        else:
            raise ValueError(f"Unsupported filter type: {ftype}")

        super().__init__(self.dx, apply_filter, ftype)

    def apply(self, x):
        return np.dot(self.P, x), np.dot(self.Q, x)

    def __repr__(self):
        return f"CompactFilter(kc={self.kc}, eps={self.eps}, n={self.n})"
    
    def filter(self, du, u):
        """
        Apply the compact filter to the input array u.
        """
        if self.apply_filter == FilterApply.NONE:
            return u
        elif self.apply_filter == FilterApply.RHS:
            return self.apply(u)
        elif self.apply_filter == FilterApply.APPLY_VARS:
            return self.apply(u)
        elif self.apply_filter == FilterApply.APPLY_DERIVS:
            return self.apply(u)
        else:
            raise ValueError(f"Unsupported filter application: {self.apply_filter}")



def kim_filter_cal_coeff(kc):
    AF = 30.0 - 5.0*np.cos(kc) + 10.0*np.cos(2.0*kc) - 3.0*np.cos(3.0*kc)
    alphaF = -(30.0*np.cos(kc) + 2.0*np.cos(3.0*kc)) / AF
    betaF = (18.0 + 9.0*np.cos(kc) + 6.0*np.cos(2.0*kc) - np.cos(3.0*kc)) / (2.0*AF)
    return AF, alphaF, betaF

def init_kim_filter(kc, eps, N):
    """
    # Compact filters for wavelet transforms, based on the Kim filter design
    # from the paper "Compact Filters for Wavelet Transforms" by Kim et al.
    # https://doi.org/10.1109/TSP.2005.847059
    # This implementation provides a class for creating and applying compact filters
    # with specified cutoff frequency (kc), epsilon (eps), and filter order (n).

    Create the P and Q matrices for the compact filter based on Kim's design.
    Parameters:
    kc : float
        Cutoff frequency.
    eps : float
        Epsilon value for the filter.
    n : int
        Order of the filter.
    Returns:
    P : np.ndarray
        The P matrix for the filter.
    Q : np.ndarray
        The Q matrix for the filter.
    """

    c0  = kim_filter_cal_coeff(kc)
    cd  = kim_filter_cal_coeff(kc*(1.0 - eps*np.sin(np.pi/6)**2))
    cdd = kim_filter_cal_coeff(kc*(1.0 - eps*np.sin(np.pi/3)**2))
    cddd= kim_filter_cal_coeff(kc*(1.0 - eps*np.sin(np.pi/2)**2))

    t1  = np.cos(0.5*kc)
    aF1 = 30.0*t1**4 / c0[0]
    aF2 = -2.0*aF1 / 5.0
    aF3 = aF1 / 15.0
    aF0 = -2.0 * (aF1 + aF2 + aF3)

    alphaF, betaF = c0[1], c0[2]
    alphaFd, betaFd = cd[1], cd[2]
    alphaFdd, betaFdd = cdd[1], cdd[2]
    alphaFddd, betaFddd = cddd[1], cddd[2]

    aF1d = 30.0*np.cos(0.5*kc*(1.0 - eps*np.sin(np.pi/6)**2))**4 / cd[0]
    aF2d = -2.0*aF1d/5.0
    aF3d = aF1d/15.0

    BF = ((1.0 - betaFdd)*(1.0 + 6.0*betaFdd + 60.0*betaFdd**2)
        + (5.0 + 35*betaFdd -29.0*betaFdd**2)*alphaFdd
        + (9.0 - 5.0*betaFdd)*alphaFdd**2)
    CF = (1.0 + betaFddd*(5.0 + 4.0*betaFddd + 60.0*betaFddd**2)
        + 5.0*(1.0 + 3.0*betaFddd + 10.0*betaFddd**2)*alphaFddd
        + 2.0*(4.0 + 11.0*betaFddd)*alphaFddd**2
        + 5.0*alphaFddd**3)

    yF10 = (10.0*betaFdd**2*(8.0*betaFdd - 1.0) 
            + (1.0 + 4.0*betaFdd + 81.0*betaFdd**2)*alphaFdd
            + 5.0*(1.0 + 8.0*betaFdd)*alphaFdd**2
            + 9.0*alphaFdd**3)/BF
    yF01 = ((alphaFddd*(1.0 + alphaFddd)*(1.0 + 4.0*alphaFddd)
            + 2.0*alphaFddd*(7.0 + 3.0*alphaFddd)*betaFddd
            + 24.0*(1.0 - alphaFddd)*betaFddd**2
            - 80.0*betaFddd**3) / CF)
    yF20, yF21, yF23, yF24 = betaFd, alphaFd, alphaFd, betaFd
    yF02 = ((alphaFddd**3 
            + (1.0 + 3.0*alphaFddd + 14.0*alphaFddd**2)*betaFddd
            + 46.0*alphaFddd*betaFddd**2
            + 60.0*betaFddd**3) / CF)
    yF12 = ((alphaFdd*(1.0 + 5.0*alphaFdd + 9.0*alphaFdd**2)
            + alphaFdd*(5.0 + 36.0*alphaFdd)*betaFdd
            + (55.0*alphaFdd - 1.0)*betaFdd**2
            + 10.0*betaFdd**3) / BF)
    yF13 = (betaFdd*(1.0 + 5.0*alphaFdd + 9.0*alphaFdd**2 
            + 5.0*(1.0 + 7.0*alphaFdd)*betaFdd 
            + 50.0*betaFdd**2) / BF)

    bF20 = aF2d + 5.0*aF3d
    bF21 = aF1d - 10.0*aF3d
    bF23 = aF1d - 5.0*aF3d
    bF24 = aF2d + aF3d
    bF25 = aF3d
    bF22 = -(bF20 + bF21 + bF23 + bF24 + bF25)

    P = np.zeros((5, N))
    P[0,2:]  = betaF
    P[1,1:]  = alphaF
    P[2,:]   = 1.0
    P[3,:-2] = alphaF
    P[4,:-3] = betaF

    coeffs = [aF3, aF2, aF1, aF0, aF1, aF2, aF3]
    Q = np.zeros((N, N))
    for i in range(3,N-3):
        Q[i,i-3 : i+4] = coeffs


    # FIXME -- Check the banded matrix storage
    P[2,0], P[1,1], P[0,2] = 1.0, yF01, yF02
    P[3,0], P[2,1], P[1,2], P[0,3] = yF10, 1.0, yF12, yF13
    P[4,0], P[3,1], P[2,2], P[1,3], P[0,4] = yF20, yF21, 1.0, yF23, yF24
    P[4,N-5], P[3,N-4], P[2,N-3], P[1,N-2], P[0,N-1] = yF24, yF23, 1.0, yF21, yF20
    P[4,N-4], P[3,N-3], P[2,N-2], P[1,N-1] = yF13, yF12, 1.0, yF10
    P[4,N-3], P[3,N-2], P[2,N-1] = yF02, yF01, 1.0

    Q[2,0:6] = [bF20, bF21, bF22, bF23, bF24, bF25]
    Q[N-3,N-6:N] = [bF25, bF24, bF23, bF22, bF21, bF20]

    return P, Q

def init_jt_filter(alpha, beta, N):
    """
    Initialize the J-T filter coefficients.
    Parameters:
    alpha : float
        Alpha coefficient for the filter.
    beta : float
        Beta coefficient for the filter.
    N : int
        Grid size for the filter.
    Returns:
    P : np.ndarray
        The P matrix for the filter.
    Q : np.ndarray
        The Q matrix for the filter.
    """
    P = np.zeros((5, N))
    Q = np.zeros((N, N))

    # J-T filter coefficients
    P[0, 2:] = beta
    P[1, 1:] = alpha
    P[2, :] = 1.0
    P[3, :-2] = alpha
    P[4, :-3] = beta

    coeffs = [1.0, -4.0 * alpha + 6.0 * beta, 6.0 * alpha - 4.0 * beta, -beta]
    for i in range(3, N - 3):
        Q[i, i - 3:i + 4] = coeffs

    return P, Q 

def _filter_JT(alpha, beta):
    if ftype == FilterType.JFT6:    
        coeffs = [(11 + 10*alpha)/16, (15 + 34*alpha)/32, (-3 + 6*alpha)/16, (1-2*alpha)/32]
    elif ftype == FilterType.JFT8:
        coeffs = [(93 + 70*alpha)/128, (7 + 18*alpha)/16, (-7 + 14*alpha)/32, (1-2*alpha)/16, (-1 + 2*alpha)/128]
    elif ftype == FilterType.JFP6:    
        coeffs = [(11 + 10*alpha - 10*beta)/16, (15 + 34*alpha + 30*beta)/32, (-3 + 6*alpha + 26*beta)/16, (1-2*alpha + 2*beta)/32]
    elif ftype == FilterType.JFP8:
        coeffs = [(93 + 70*alpha - 70*beta)/128, (7 + 18*alpha + 14*beta)/16, (-7 + 14*alpha + 50*beta)/32, (1-2*alpha + 2*beta)/16, (-1 + 2*alpha - 2*beta)/128]
    else:
        raise NotImplementedError("Unknown filter")

    Q = np.zeros(N,N)
    ib = len(coeffs) - 1
    ie = N - ib
    for i in len(ib, ie):
        Q[i,i-ib:i+ib+1] = coeffs

    if fmethod == FilterType.JFP6 or fmethod == FilterType.JFP8:
        P = np.zeros(5,N)
        P[0,2:] = beta
        P[1,1:] = alpha
        P[2,:] = 1.0
        P[3,:,-1] = alpha
        P[4,:,-2] = beta
    else:
        P = np.zeros(3,N)
        P[0,1:] = alpha
        P[1,:] = 1.0
        P[2,:,-1] = alpha

