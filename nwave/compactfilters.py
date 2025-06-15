import numpy as np
from scipy.linalg import solve_banded
from . import utils
from .filters import Filter1D
from .types import *


class CompactFilter(Filter1D):
    def __init__(
        self,
        x,
        apply_filter: FilterApply,
        ftype: FilterType,
        frequency,
        alpha=0.4,
        beta=0.4,
        kim_eps=0.0,
        kim_kc=0.0,
        filter_boundary=False,
    ):
        self.N = len(x)
        self.dx = x[1] - x[0]

        if frequency <= 0:
            raise ValueError("Frequency must be positive")

        self.frequency = frequency
        self.kc = kim_kc
        self.eps = kim_eps
        self.alpha = alpha
        self.beta = beta
        self.filter_boundary = filter_boundary
        self.bands = (0, 0)  # Default bands for banded matrix storage

        if ftype == FilterType.KP4:
            self.Ab, self.B = init_kim_filter(kim_kc, kim_eps, self.N)
            self.bands = (2, 2)
        elif (
            ftype == FilterType.JTT6
            or ftype == FilterType.JTP6
            or ftype == FilterType.JTT8
            or ftype == FilterType.JTP8
        ):
            self.Ab, self.B, self.bands = init_JT_filter(
                ftype, alpha, beta, filter_boundary, self.N
            )
        else:
            raise ValueError(f"Unsupported filter type: {ftype}")

        super().__init__(self.dx, apply_filter, ftype, frequency)

    @classmethod
    def from_params(cls, params, x):
        """
        Create a CompactFilter instance from parameters.
        """
        fstr = params.get("Filter", "None")
        if fstr == "None":
            ftype = FilterType.NONE
        elif fstr == "KP4":
            ftype = FilterType.KP4
        elif fstr == "JTT6":
            ftype = FilterType.JTT6
        elif fstr == "JTP6":
            ftype = FilterType.JTP6
        elif fstr == "JTT8":
            ftype = FilterType.JTT8
        elif fstr == "JTP8":
            ftype = FilterType.JTP8
        else:
            raise ValueError(f"Unknown filter type: {fstr}")

        afstr = params.get("FilterApply", "None")
        if afstr == "RHS":
            apply_filter = FilterApply.RHS
        elif afstr == "Vars":
            apply_filter = FilterApply.APPLY_VARS
        elif afstr == "Derivs":
            apply_filter = FilterApply.APPLY_DERIVS
        elif afstr == "None":
            apply_filter = FilterApply.NONE
        else:
            raise ValueError(f"Unknown filter application: {afstr}")

        n = len(x)
        if n < 10:
            raise ValueError("Filter order N must be at least 10")

        freq = params.get("FilterFrequency", 1)
        if freq <= 0:
            raise ValueError("Filter frequency must be positive")

        alpha = params.get("FilterAlpha", 0.4)
        beta = params.get("FilterBeta", 0.4)
        kim_eps = params.get("FilterKimEpsilon", 0.25)
        kim_kc = params.get("FilterKimCutoff", 0.88)
        fbounds = params.get("FilterBoundary", False)

        return cls(
            x,
            apply_filter,
            ftype,
            freq,
            alpha,
            beta,
            kim_eps,
            kim_kc,
            filter_boundary=fbounds,
        )

    def apply(self, x):
        return np.dot(self.P, x), np.dot(self.Q, x)

    def __repr__(self):
        return f"CompactFilter(kc={self.kc}, eps={self.eps}, n={self.n})"

    def filter(self, u):
        """
        Apply the compact filter to the input array u.
        """
        # use scipy solve_banded
        rhs = np.matmul(self.B, u)
        if self.filter_type == FilterType.KP4:
            return u + solve_banded(self.bands, self.Ab, rhs)
        else:
            return solve_banded(self.bands, self.Ab, rhs)
    
    def get_A(self):
        """
        Get the banded matrix Ab for the filter.
        """
        return utils.banded_to_full_slow(self.Ab, self.bands[0], self.bands[1], self.N, self.N)

    def get_B(self):
        """
        Get the banded matrix B for the filter.
        """
        return self.B
    
    def get_Abanded(self):
        """
        Get the banded matrix Ab for the filter.
        """
        return self.Ab


def kim_filter_cal_coeff(kc):
    AF = 30.0 - 5.0 * np.cos(kc) + 10.0 * np.cos(2.0 * kc) - 3.0 * np.cos(3.0 * kc)
    alphaF = -(30.0 * np.cos(kc) + 2.0 * np.cos(3.0 * kc)) / AF
    betaF = (18.0 + 9.0 * np.cos(kc) + 6.0 * np.cos(2.0 * kc) - np.cos(3.0 * kc)) / (
        2.0 * AF
    )
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

    c0 = kim_filter_cal_coeff(kc)
    cd = kim_filter_cal_coeff(kc * (1.0 - eps * np.sin(np.pi / 6) ** 2))
    cdd = kim_filter_cal_coeff(kc * (1.0 - eps * np.sin(np.pi / 3) ** 2))
    cddd = kim_filter_cal_coeff(kc * (1.0 - eps * np.sin(np.pi / 2) ** 2))

    t1 = np.cos(0.5 * kc)
    aF1 = 30.0 * t1**4 / c0[0]
    aF2 = -2.0 * aF1 / 5.0
    aF3 = aF1 / 15.0
    aF0 = -2.0 * (aF1 + aF2 + aF3)

    alphaF, betaF = c0[1], c0[2]
    alphaFd, betaFd = cd[1], cd[2]
    alphaFdd, betaFdd = cdd[1], cdd[2]
    alphaFddd, betaFddd = cddd[1], cddd[2]

    aF1d = 30.0 * np.cos(0.5 * kc * (1.0 - eps * np.sin(np.pi / 6) ** 2)) ** 4 / cd[0]
    aF2d = -2.0 * aF1d / 5.0
    aF3d = aF1d / 15.0

    BF = (
        (1.0 - betaFdd) * (1.0 + 6.0 * betaFdd + 60.0 * betaFdd**2)
        + (5.0 + 35 * betaFdd - 29.0 * betaFdd**2) * alphaFdd
        + (9.0 - 5.0 * betaFdd) * alphaFdd**2
    )
    CF = (
        1.0
        + betaFddd * (5.0 + 4.0 * betaFddd + 60.0 * betaFddd**2)
        + 5.0 * (1.0 + 3.0 * betaFddd + 10.0 * betaFddd**2) * alphaFddd
        + 2.0 * (4.0 + 11.0 * betaFddd) * alphaFddd**2
        + 5.0 * alphaFddd**3
    )

    yF10 = (
        10.0 * betaFdd**2 * (8.0 * betaFdd - 1.0)
        + (1.0 + 4.0 * betaFdd + 81.0 * betaFdd**2) * alphaFdd
        + 5.0 * (1.0 + 8.0 * betaFdd) * alphaFdd**2
        + 9.0 * alphaFdd**3
    ) / BF
    yF01 = (
        alphaFddd * (1.0 + alphaFddd) * (1.0 + 4.0 * alphaFddd)
        + 2.0 * alphaFddd * (7.0 + 3.0 * alphaFddd) * betaFddd
        + 24.0 * (1.0 - alphaFddd) * betaFddd**2
        - 80.0 * betaFddd**3
    ) / CF
    yF20, yF21, yF23, yF24 = betaFd, alphaFd, alphaFd, betaFd
    yF02 = (
        alphaFddd**3
        + (1.0 + 3.0 * alphaFddd + 14.0 * alphaFddd**2) * betaFddd
        + 46.0 * alphaFddd * betaFddd**2
        + 60.0 * betaFddd**3
    ) / CF
    yF12 = (
        alphaFdd * (1.0 + 5.0 * alphaFdd + 9.0 * alphaFdd**2)
        + alphaFdd * (5.0 + 36.0 * alphaFdd) * betaFdd
        + (55.0 * alphaFdd - 1.0) * betaFdd**2
        + 10.0 * betaFdd**3
    ) / BF
    yF13 = (
        betaFdd
        * (
            1.0
            + 5.0 * alphaFdd
            + 9.0 * alphaFdd**2
            + 5.0 * (1.0 + 7.0 * alphaFdd) * betaFdd
            + 50.0 * betaFdd**2
        )
        / BF
    )

    bF20 = aF2d + 5.0 * aF3d
    bF21 = aF1d - 10.0 * aF3d
    bF23 = aF1d - 5.0 * aF3d
    bF24 = aF2d + aF3d
    bF25 = aF3d
    bF22 = -(bF20 + bF21 + bF23 + bF24 + bF25)

    P = np.zeros((5, N))
    P[0, 2:] = betaF
    P[1, 1:] = alphaF
    P[2, :] = 1.0
    P[3, :-2] = alphaF
    P[4, :-3] = betaF

    coeffs = [aF3, aF2, aF1, aF0, aF1, aF2, aF3]
    Q = np.zeros((N, N))
    for i in range(3, N - 3):
        Q[i, i - 3 : i + 4] = coeffs

    # FIXME -- Check the banded matrix storage
    P[2, 0], P[1, 1], P[0, 2] = 1.0, yF01, yF02
    P[3, 0], P[2, 1], P[1, 2], P[0, 3] = yF10, 1.0, yF12, yF13
    P[4, 0], P[3, 1], P[2, 2], P[1, 3], P[0, 4] = yF20, yF21, 1.0, yF23, yF24
    P[4, N - 5], P[3, N - 4], P[2, N - 3], P[1, N - 2], P[0, N - 1] = (
        yF24,
        yF23,
        1.0,
        yF21,
        yF20,
    )
    P[4, N - 4], P[3, N - 3], P[2, N - 2], P[1, N - 1] = yF13, yF12, 1.0, yF10
    P[4, N - 3], P[3, N - 2], P[2, N - 1] = yF02, yF01, 1.0

    Q[2, 0:6] = [bF20, bF21, bF22, bF23, bF24, bF25]
    Q[N - 3, N - 6 : N] = [bF25, bF24, bF23, bF22, bF21, bF20]

    return P, Q


def init_JT_filter(ftype, alpha, beta, filter_boundary, N):

    kl, ku = 0, 0
    # Set beta = 0 for tridiagonal filters
    if ftype == FilterType.JTT6 or ftype == FilterType.JTT8:
        beta = 0.0
        kl, ku = 1, 1

    if ftype == FilterType.JTP6 or ftype == FilterType.JTP8:
        kl, ku = 2, 2

    if ftype == FilterType.JTT6 or ftype == FilterType.JTP6:
        coeffs = 0.5 * np.array(
            [
                (1 - 2 * alpha + 2 * beta) / 32,
                (-3 + 6 * alpha + 26 * beta) / 16,
                (15 + 34 * alpha + 30 * beta) / 32,
                2 * (11 + 10 * alpha - 10 * beta) / 16,
                (15 + 34 * alpha + 30 * beta) / 32,
                (-3 + 6 * alpha + 26 * beta) / 16,
                (1 - 2 * alpha + 2 * beta) / 32,
            ]
        )
    elif ftype == FilterType.JTT8 or ftype == FilterType.JTP8:
        coeffs = 0.5 * np.array(
            [
                (-1 + 2 * alpha - 2 * beta) / 128,
                (1 - 2 * alpha + 2 * beta) / 16,
                (-7 + 14 * alpha + 50 * beta) / 32,
                (7 + 18 * alpha + 14 * beta) / 16,
                2 * (93 + 70 * alpha - 70 * beta) / 128,
                (7 + 18 * alpha + 14 * beta) / 16,
                (-7 + 14 * alpha + 50 * beta) / 32,
                (1 - 2 * alpha + 2 * beta) / 16,
                (-1 + 2 * alpha - 2 * beta) / 128,
            ]
        )
    else:
        raise NotImplementedError("Unknown filter")

    bands = (kl, ku)
    Q = np.zeros((N, N))

    # Fill the Q matrix with the coefficients
    ib = int((len(coeffs) - 1) / 2)
    ie = N - ib
    for i in range(ib, ie):
        Q[i, i - ib : i + ib + 1] = coeffs

    if kl == 2:
        P = np.zeros((5, N))
        P[0, 2:] = beta
        P[1, 1:] = alpha
        P[2, :] = 1.0
        P[3, :-1] = alpha
        P[4, :-2] = beta
    elif kl == 1:
        P = np.zeros((3, N))
        P[0, 1:] = alpha
        P[1, :] = 1.0
        P[2, :-1] = alpha

    for i in range(ib, ie):
        Q[i, i - ib : i + ib + 1] = coeffs

    if filter_boundary:
        # Apply dissipation boundary conditions
        if ftype == FilterType.JTT6 or ftype == FilterType.JTP6:
            bcoeffs = _jt_bounds_p6(alpha, beta)
        elif ftype == FilterType.JTT8 or ftype == FilterType.JTP8:
            bcoeffs = _jt_bounds_p8(alpha, beta)
        else:
            raise NotImplementedError("Unknown filter type for dissipation boundaries")

        for i in range(len(bcoeffs)):
            Q[i, 0 : len(bcoeffs[i])] = bcoeffs[:]
            Q[N - 1 - i, N - len(bcoeffs[i]) : N] = bcoeffs[i][::-1]
    else:
        # Q: Identity terms
        for i in range(ib):
            Q[i, i] = 1.0
            Q[N - 1 - i, N - 1 - i] = 1.0

        # P: Fill LHS boundary conditions
        for i in range(ib):
            jb = max(0, i - kl)
            je = min(N, i + ku + 1)
            for j in range(jb, je):
                ii = ku + i - j
                if j == i:
                    P[ii, j] = 1.0
                else:
                    P[ii, j] = 0.0

        # P: Fill RHS boundary conditions
        for i in range(N - ib, N):
            jb = max(0, i - kl)
            je = min(N, i + ku + 1)
            for j in range(jb, je):
                ii = ku + i - j
                if j == i:
                    P[ii, j] = 1.0
                else:
                    P[ii, j] = 0.0

    return P, Q, bands


def _jt_bounds_p6(alpha, beta):
    a00 = (63 + alpha - beta) / 64
    a01 = (3 + 29 * alpha + 3 * beta) / 32
    a02 = (-15 + 15 * alpha + 49 * beta) / 64
    a03 = 5 * (1 - alpha + beta) / 16
    a04 = 15 * (-1 + alpha - beta) / 64
    a05 = 3 * (1 - alpha + beta) / 32
    a06 = (-1 + alpha - beta) / 64
    bcoeffs0 = np.array([a00, a01, a02, a03, a04, a05, a06])

    a10 = (1 + 62 * alpha + beta) / 64
    a11 = (29 + 6 * alpha - 3 * beta) / 32
    a12 = (15 + 34 * alpha + 15 * beta) / 64
    a13 = (-5 + 10 * alpha + 11 * beta) / 16
    a14 = 15 * (1 - 2 * alpha + beta) / 64
    a15 = 3 * (-1 + 2 * alpha - beta) / 32
    a16 = (1 - 2 * alpha + beta) / 64
    bcoeffs1 = np.array([a10, a11, a12, a13, a14, a15, a16])

    a20 = (-1 + 2 * alpha + 62 * beta) / 64
    a21 = (3 + 26 * alpha + 6 * beta) / 32
    a22 = (49 + 30 * alpha - 30 * beta) / 64
    a23 = (5 + 6 * alpha + 10 * beta) / 16
    a24 = (-15 + 30 * alpha + 34 * beta) / 64
    a25 = 3 * (1 - 2 * alpha + 2 * beta) / 32
    a26 = (-1 + 2 * alpha - 2 * beta) / 64
    bcoeffs2 = np.array([a20, a21, a22, a23, a24, a25, a26])

    return [bcoeffs0, bcoeffs1, bcoeffs2]


def _jt_bounds_p8(alpha, beta):
    a00 = (255 + alpha - beta) / 256
    a01 = (1 + 31 * alpha + beta) / 32
    a02 = (-7 + 7 * alpha + 57 * beta) / 64
    a03 = 7 * (1 - alpha + beta) / 32
    a04 = 35 * (-1 + alpha - beta) / 128
    a05 = 7 * (1 - alpha + beta) / 32
    a06 = 7 * (-1 + alpha - beta) / 64
    a07 = (1 - alpha + beta) / 32
    a08 = (-1 + alpha - beta) / 256
    bcoeffs0 = np.array([a00, a01, a02, a03, a04, a05, a06, a07, a08])

    a10 = (1 + 254 * alpha + beta) / 256
    a11 = (31 + 2 * alpha - beta) / 32
    a12 = (7 + 50 * alpha + 7 * beta) / 64
    a13 = (-7 + 14 * alpha + 25 * beta) / 32
    a14 = 35 * (1 - 2 * alpha + beta) / 128
    a15 = 7 * (-1 + 2 * alpha - beta) / 32
    a16 = 7 * (1 - 2 * alpha + beta) / 64
    a17 = (-1 + 2 * alpha - beta) / 32
    a18 = (1 - 2 * alpha + beta) / 256
    bcoeffs1 = np.array([a10, a11, a12, a13, a14, a15, a16, a17, a18])

    a20 = (-1 + 2 * alpha + 254 * beta) / 256
    a21 = (1 + 30 * alpha + 2 * beta) / 32
    a22 = (57 + 14 * alpha - 14 * beta) / 64
    a23 = (7 + 18 * alpha + 14 * beta) / 32
    a24 = (-35 + 70 * alpha + 58 * beta) / 128
    a25 = 7 * (1 - 2 * alpha + 2 * beta) / 32
    a26 = 7 * (-1 + 2 * alpha - 2 * beta) / 64
    a27 = (1 - 2 * alpha + 2 * beta) / 32
    a28 = (-1 + 2 * alpha - 2 * beta) / 256
    bcoeffs2 = np.array([a20, a21, a22, a23, a24, a25, a26, a27, a28])

    a30 = (1 - 2 * alpha + 2 * beta) / 256
    a31 = (-1 + 2 * alpha + 30 * beta) / 32
    a32 = (7 + 50 * alpha + 14 * beta) / 64
    a33 = (25 + 14 * alpha - 14 * beta) / 32
    a34 = (35 + 58 * alpha + 70 * beta) / 128
    a35 = (-7 + 14 * alpha + 18 * beta) / 32
    a36 = 7 * (1 - 2 * alpha + 2 * beta) / 64
    a37 = (-1 + 2 * alpha - 2 * beta) / 32
    a38 = (1 - 2 * alpha + 2 * beta) / 256
    bcoeffs3 = np.array([a30, a31, a32, a33, a34, a35, a36, a37, a38])

    return [bcoeffs0, bcoeffs1, bcoeffs2, bcoeffs3]
