import numpy as np
from scipy.linalg import solve_banded, inv
from enum import Enum
from .bandedLUSolve import *
from .utils import *
from .types import *
import pentapy as pp

CompactFirstDerivatives = [
    DerivType.D1_KP4,
    DerivType.D1_ME44,
    DerivType.D1_ME642,
    DerivType.D1_JT4,
    DerivType.D1_JT6,
    DerivType.D1_JP6,
    DerivType.D1_DSQ6A,
    DerivType.D1_DSQ6B,
    DerivType.D1_BYUP6,
    DerivType.D1_Wm6,
    DerivType.D1_DSQ6B_LEFT,
    DerivType.D1_DSQ6B_RIGHT,
]
CompactSecondDerivatives = [
    DerivType.D2_ME44,
    DerivType.D2_ME642,
    DerivType.D2_JT4,
    DerivType.D2_JT6,
    DerivType.D2_JP6,
    DerivType.D2_DSQ6A,
    DerivType.D2_DSQ6B,
    DerivType.D2_BYUP6,
]
CompactExplicitDerivatives = [
    DerivType.D1_ME44,
    DerivType.D2_ME44,
    DerivType.D1_ME642,
    DerivType.D2_ME642,
]


class NCompactDerivative:
    def __init__(self, N: int, dtype: DerivType, method: CFDSolve, Pmat, Qmat, denom, mask=None):
        """
        Initialize the NCompactDerivative class.
        """
        if dtype in CompactExplicitDerivatives:
            if method != CFDSolve.D_LU and method != CFDSolve.D_INV:
                print(
                    f"Method {method} not supported for explicit derivatives of type {dtype}"
                )
            self.method = CFDSolve.D_LU
        else:
            self.method = method

        self.N = N
        self.dtype = dtype
        self.Pbands = Pmat[1]
        self.Qbands = Qmat[1]
        self.P = Pmat[0]
        self.Q = Qmat[0]
        self.denom = denom
        self.lu_factorization = None
        self.D = None
        self.overwrite = True
        self.checkf = True
        self.mask = mask

        if self.method == CFDSolve.LUSOLVE:
            self.lu_factorization = lu_banded(
                self.Pbands,
                self.P,
                overwrite_ab=self.overwrite,
                check_finite=self.checkf,
            )
        elif self.method == CFDSolve.D_INV:
            if dtype in CompactExplicitDerivatives:
                self.D = self.Q
            else:
                nb = self.Pbands[0]
                A = banded_to_full(self.P, nb, nb, self.N, self.N)
                Ainv = inv(A)
                self.D = np.matmul(Ainv, self.Q)
        elif self.method == CFDSolve.D_LU:
            if dtype in CompactExplicitDerivatives:
                self.D = self.Q
            else:
                self.D = solve_banded(self.Pbands, self.P, self.Q)
        elif self.method == CFDSolve.SCIPY or method == CFDSolve.PENTAPY:
            pass
        else:
            raise ValueError(f"Compact derivative method = {method} not implemented")

    @classmethod
    def bh_deriv(
        cls,
        r,
        dtype0: DerivType,
        dtypeBH: DerivType,
        method: CFDSolve,
        rbh,
        delta_r,
    ):
        """
        Factory method to create combined derivative for BH
        """
        N = len(r)
        dr = r[1] - r[0]
        if dtype0 in CompactFirstDerivatives:
            denom = 1 / dr
        elif dtype0 in CompactSecondDerivatives:
            denom = 1 / (dr * dr)
        else:
            raise ValueError(f"Unsupported first derivative type: {dtype0}")

        mask0 = 1.0
        maskBH = 0.0
        r0 = rbh - 2 * delta_r
        r1 = rbh - delta_r
        r2 = rbh + delta_r
        r3 = rbh + 2 * delta_r

        bg_mask = generalized_transition_profile(
            r, mask0, maskBH, r0, r1, r2, r3, method="tanh"
        )

        Pmat, Qmat = build_bh_derivative(N, dtype0, dtypeBH, bg_mask)
        return cls(N, dtype0, method, Pmat, Qmat, denom, mask=bg_mask)

    @classmethod
    def deriv(cls, r, dtype: DerivType, method: CFDSolve):
        """
        Factory method to create an instance based on the derivative type.
        """
        N = len(r)
        dr = r[1] - r[0]
        if dtype in CompactFirstDerivatives:
            denom = 1 / dr
        elif dtype in CompactSecondDerivatives:
            denom = 1 / (dr * dr)
        else:
            raise ValueError(f"Unsupported first derivative type: {dtype}")

        Pmat, Qmat = build_derivative(N, dtype)
        return cls(N, dtype, method, Pmat, Qmat, denom)

    def get_P(self) -> np.ndarray:
        """
        Get the P matrix.
        """
        return self.P

    def get_Q(self) -> np.ndarray:
        """
        Get the Q matrix.
        """
        return self.Q

    def get_Pbands(self) -> tuple:
        """
        Get the P bands.
        """
        return self.Pbands

    def get_Qbands(self) -> tuple:
        """
        Get the Q bands.
        """
        return self.Qbands

    def get_D(self) -> np.ndarray:
        """
        Get the D matrix.
        """
        if self.D is None:
            raise ValueError("D matrix is not initialized")
        return self.D
    
    def get_mask(self) -> np.ndarray:
        """
        Get the mask array.
        """
        if self.mask is None:
            raise ValueError("Mask is not initialized")
        return self.mask

    def grad(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the function f using the derivative matrices.
        """
        if len(f) != self.N:
            raise ValueError(f"Input array length {len(f)} does not match N {self.N}")

        if self.method == CFDSolve.LUSOLVE:
            rhs = np.matmul(self.Q, f) * self.denom
            dxf = lu_solve_banded(
                self.lu_factorization, rhs, overwrite_b=True, check_finite=self.checkf
            )
        elif self.method == CFDSolve.D_INV or self.method == CFDSolve.D_LU:
            dxf = np.matmul(self.D, f) * self.denom
        elif self.method == CFDSolve.SCIPY:
            rhs = np.matmul(self.Q, f) * self.denom
            dxf = solve_banded(self.Pbands, self.P, rhs)
        elif self.method == CFDSolve.PENTAPY:
            rhs = np.matmul(self.Q, f) * self.denom
            return np.asarray(
                pp.solve(self.P, rhs, is_flat=True, index_row_wise=False, solver=1)
            )
        else:
            raise ValueError(
                f"Compact derivative method = {self.method} not implemented"
            )

        return dxf


def _coeffs_D1_JTT4():
    Lcoeff = np.array([0.25, 1.0, 0.25])
    Rcoeff = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbands = (1, 1)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lbcoeff0], Lbands, 1]
    Rmat = [Rcoeff, [Rbcoeff0], Rbands, -1]
    Rbands = (1, 1)
    return Lmat, Rmat


def _coeffs_D2_JTT4():
    alpha = 1.0 / 10
    a = 6.0 / 5
    Lcoeff = np.array([alpha, 1.0, alpha])
    Rcoeff = np.array([a, -2 * a, a])

    Lbcoeff0 = np.array([1.0, 10.0])
    Rbcoeff0 = np.array([145 / 12, -76 / 3, 29 / 2, -4 / 3, 1 / 12])
    Lbands = (1, 1)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lbcoeff0], Lbands, 1]
    Rmat = [Rcoeff, [Rbcoeff0], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_JTT6():
    alpha = 1 / 3
    Lcoeff = np.array([alpha, 1.0, alpha])
    Rcoeff = np.array([-1 / 36, -14 / 18, 0, 14 / 18, 1 / 36])

    Lbcoeff0 = np.array([1.0, 5.0])
    Lbcoeff1 = np.array([1 / 8, 1.0, 3 / 4])

    Rcoeffb0 = np.array([-197 / 60, -5 / 12, 5, -5 / 3, 5 / 12, -1 / 20])
    Rcoeffb1 = np.array([-43 / 96, -5 / 6, 9 / 8, 1 / 6, -1 / 96])
    Lbands = (1, 1)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, -1]
    return Lmat, Rmat


def _coeffs_D2_JTT6():
    raise NotImplementedError("D2_JTT6 coefficients are not implemented yet.")


def _coeffs_D1_JTP6():
    alpha = 17.0 / 57.0
    beta = -1.0 / 114.0
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])
    a = 30.0 / 19
    Rcoeffs = np.array([-a / 2, 0.0, a / 2])
    Lcoeffb0 = np.array([1.0, 8.0, 6.0])
    Lcoeffb1 = np.array([alpha, 1.0, alpha, beta])
    Rcoeffb0 = np.array([-43 / 12, -20 / 3, 9, 4 / 3, -1 / 12])
    Rcoeffb1 = np.array([-a / 2, 0.0, a / 2])
    Lbands = (2, 2)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeffs, [Rcoeffb0, Rcoeffb1], Rbands, -1]
    return Lmat, Rmat


def _coeffs_D2_JTP6():
    alpha = 12 / 97
    beta = -1 / 194
    a = 120 / 97
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])
    Rcoeff = np.array([a, -2 * a, a])

    Lcoeffb0 = np.array([1.0, 11 / 2, -131 / 4])
    Lcoeffb1 = np.array([alpha, 1.0, alpha, beta])
    Rcoeffb0 = np.array([177 / 16, -507 / 8, 783 / 8, -201 / 4, 81 / 16, -3 / 8])
    Rcoeffb1 = np.array([a, -2 * a, a])
    Lbands = (2, 2)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6A():
    """
    Coefficients for Boris Daszuta's Q_3 first derivative, optimized for eta = 1.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XII.
    """
    Lcoeff = np.array([0.37987923, 1.0, 0.37987923])
    Rcoeff = np.array(
        [
            0.0023272948,
            -0.052602255,
            -0.78165660,
            0.0,
            0.78165660,
            0.052602255,
            -0.0023272948,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_DSQ6A():
    """
    Coefficients for Boris Daszuta's Q_3 second derivative, optimized for eta = 1.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XIII.
    """
    Lcoeff = np.array([0.24246603, 1.0, 0.24246603])
    Rcoeff = np.array(
        [
            -0.0037062571,
            0.14095923,
            0.95445144,
            -2.1834080,
            0.95445144,
            0.14095923,
            -0.0037062571,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B():
    """
    Coefficients for Boris Daszuta's Q_3 first derivative, optimized for eta = 4*pi/5.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XII.
    """
    Lcoeff = np.array([0.41825851, 1.0, 0.41825851])
    Rcoeff = np.array(
        [
            0.0042462587,
            -0.073071204,
            -0.78485488,
            0.0,
            0.78485488,
            0.073071204,
            -0.0042462587,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_DSQ6B():
    """
    Coefficients for Boris Daszuta's Q_3 second derivative, optimized for eta = 4*pi/5.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XIII.
    """
    Lcoeff = np.array([0.28533501, 1.0, 0.28533501])
    Rcoeff = np.array(
        [
            -0.0063260285,
            0.19240201,
            0.85799622,
            -2.0881444,
            0.85799622,
            0.19240201,
            -0.0063260285,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_ME44():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Rcoeffb0 = np.array([-25, 48, -36, 16, -3]) / 12.0
    Rcoeffb1 = np.array([-3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0
    Lbands = (0, 0)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_ME44():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Rcoeffb0 = np.array([45, -154, 214, -156, 61, -10]) / 12.0
    Rcoeffb1 = np.array([10, -15, -4, 14, -6, 1]) / 12.0
    Lbands = (0, 0)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_ME642():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Lcoeffb2 = np.array([1.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (0, 0)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_MEQ642():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Lcoeffb2 = np.array([1.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (0, 0)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_KP4():
    """
    #  This is the 4th-order compact operator defined in Wu and Kim (2024).
    #  The terms are defined in Table 3.
    """
    alpha = 0.5862704032801503
    beta = 9.549533555017055e-2
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])

    a1 = 0.6431406736919156
    a2 = 0.2586011023495066
    a3 = 7.140953479797375e-3
    Rcoeff = np.array([-a3, -a2, -a1, 0.0, a1, a2, a3])

    # i = 0
    alpha01 = 43.65980335321481
    beta02 = 92.40143116322876

    b01 = -86.92242000231872
    b02 = 47.58661913475775
    b03 = 57.30693626084370
    b04 = -13.71254216556246
    b05 = 2.659826729790792
    b06 = -0.2598929200600359

    # i = 1
    alpha10 = 0.08351537442980239
    alpha12 = 1.961483362670730
    beta13 = 0.8789761422182460

    b10 = -0.3199960780333493
    b12 = 0.07735499170041915
    b13 = 1.496612372811008
    b14 = 0.2046919801608821
    b15 = -0.02229717539815850
    b16 = 0.001702365014746567

    # i = 2
    beta20 = 0.008073091519768687
    alpha21 = 0.2162434143850924
    alpha23 = 1.052242062502679
    beta24 = 0.2116022463346598

    b20 = -0.03644974757120792
    b21 = -0.4997030280694729
    b23 = 0.7439822445654316
    b24 = 0.5629384925762924
    b25 = 0.01563884275691290
    b26 = -0.0003043666146108995

    b00 = -(b01 + b02 + b03 + b04 + b05 + b06)
    b11 = -(b10 + b12 + b13 + b14 + b15 + b16)
    b22 = -(b20 + b21 + b23 + b24 + b25 + b26)

    Lcoeffb0 = np.array([1.0, alpha01, beta02])
    Lcoeffb1 = np.array([alpha10, 1.0, alpha12, beta13])
    Lcoeffb2 = np.array([beta20, alpha21, 1.0, alpha23, beta24])

    Rcoeffb0 = np.array([b00, b01, b02, b03, b04, b05, b06])
    Rcoeffb1 = np.array([b10, b11, b12, b13, b14, b15, b16])
    Rcoeffb2 = np.array([b20, b21, b22, b23, b24, b25, b26])
    Lbands = (2, 2)
    Rbands = (3, 3)

    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_BYUP6(optcoeffs1):
    beta = optcoeffs1[1]
    alpha0 = (1.0 + 12.0 * beta) / 3.0  # default value for alpha
    alpha = alpha0 + optcoeffs1[0]
    a06 = optcoeffs1[2]
    a16 = optcoeffs1[3]

    a1 = (-2.0 * (-7.0 + 12.0 * beta)) / 9.0
    a2 = (1.0 + 114.0 * beta) / 9.0
    a00 = (-227.0 + 600.0 * a06) / 60.0
    a01 = (-65.0 + 1644.0 * a06) / 6.0
    a02 = (-5.0 * (-7.0 + 75.0 * a06)) / 3.0
    a03 = (-10.0 * (-1.0 + 60.0 * a06)) / 3.0
    a04 = (5.0 * (-1.0 + 120.0 * a06)) / 12.0
    a05 = (1.0 - 300.0 * a06) / 30.0
    a10 = (-247.0 + 35460.0 * a16) / 900.0
    a11 = (-19.0 - 2592.0 * a16) / 12.0
    a12 = (1.0 - 1755.0 * a16) / 3.0
    a13 = (13.0 + 5760.0 * a16) / 9.0
    a14 = (1.0 + 1620.0 * a16) / 12.0
    a15 = (-1.0 - 4320.0 * a16) / 300.0
    gamma01 = -10.0 * (-1.0 + 12.0 * a06)
    gamma02 = -10.0 * (-1.0 + 30.0 * a06)
    gamma10 = (1.0 - 180.0 * a16) / 15.0
    gamma12 = 2.0 * (1.0 + 270.0 * a16)
    gamma13 = (2.0 * (1.0 + 720.0 * a16)) / 3.0

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [[1.0, gamma01, gamma02], [gamma10, 1.0, gamma12, gamma13]]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = [beta, alpha, 1.0, alpha, beta]

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        [a00, a01, a02, a03, a04, a05, a06],
        [a10, a11, a12, a13, a14, a15, a16],
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = [-a2 / 4.0, -a1 / 2.0, 0.0, a1 / 2.0, a2 / 4.0]

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), 1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), -1]
    return Lmat, Rmat


def _coeffs_D2_BYUP6(optcoeffs2):
    beta = optcoeffs2[1]
    alpha0 = (2.0 * (1.0 + 62.0 * beta)) / 11.0  # default value for alpha
    alpha = alpha0 + optcoeffs2[0]
    a07 = optcoeffs2[2]
    a17 = optcoeffs2[3]

    a1 = (-12.0 * (-1.0 + 26.0 * beta)) / 11.0
    a2 = (3.0 * (1.0 + 194.0 * beta)) / 11.0
    a00 = (48241.0 - 549180.0 * a07) / 900.0
    a01 = (16389.0 - 262820.0 * a07) / 25.0
    a02 = (3.0 * (-10453.0 + 162000.0 * a07)) / 20.0
    a03 = (43622.0 - 667035.0 * a07) / 45.0
    a04 = (-2529.0 + 37220.0 * a07) / 20.0
    a05 = (-3.0 * (-143.0 + 1890.0 * a07)) / 25.0
    a06 = (-1169.0 + 9720.0 * a07) / 900.0
    a10 = (753829.0 - 24461820.0 * a17) / 1114560.0
    a11 = (57209.0 + 4218420.0 * a17) / 20640.0
    a12 = (-58367.0 - 8677836.0 * a17) / 8256.0
    a13 = (172793.0 + 92712420.0 * a17) / 55728.0
    a14 = (4453.0 - 7381500.0 * a17) / 8256.0
    a15 = (-391.0 + 2318580.0 * a17) / 20640.0
    a16 = (529.0 - 15888780.0 * a17) / 1114560.0
    gamma01 = (-36.0 * (-17.0 + 235.0 * a07)) / 5.0
    gamma02 = (-27.0 * (-113.0 + 1740.0 * a07)) / 5.0
    gamma10 = (23.0 - 1620.0 * a17) / 688.0
    gamma12 = (5.0 * (467.0 + 8028.0 * a17)) / 688.0
    gamma13 = (2659.0 - 1702980.0 * a17) / 3096.0

    # boundary elements for P matrix for 2nd derivative
    P2DiagBoundary = [[1.0, gamma01, gamma02], [gamma10, 1.0, gamma12, gamma13]]

    # diagonal elements for P matrix for 2nd derivative
    P2DiagInterior = [beta, alpha, 1.0, alpha, beta]

    # boundary elements for Q matrix for 2nd derivative
    Q2DiagBoundary = [
        [a00, a01, a02, a03, a04, a05, a06, a07],
        [a10, a11, a12, a13, a14, a15, a16, a17],
    ]

    # diagonal elements for Q matrix for 2nd derivative
    t1 = -2.0 * (a1 / 1.0 + a2 / 4.0)
    Q2DiagInterior = [-a2 / 4.0, -a1 / 1.0, t1, a1 / 1.0, a2 / 4.0]

    Lmat = [P2DiagInterior, P2DiagBoundary, (2, 2), 1]
    Rmat = [Q2DiagInterior, Q2DiagBoundary, (2, 2), 1]
    return Lmat, Rmat

def _coeffs_D1_Wm6():
    Lcoeff = np.array([829 / 1200, 1.0, 0.0])
    Rcoeff = np.array(
        [
            29 / 12000,
            -877 / 12000,
            -24271 / 14400,
            151 / 80,
            -307 / 2400,
            -167 / 36000,
            29 / 24000,
        ]
    )

    # JTT4 coefficients for boundary conditions
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbcoeff1 = np.array([0.25, 1.0, 0.25])
    Rbcoeff1 = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])

    # V_m2, O(h^5) coefficients for boundary conditions
    Lbcoeff2 = np.array([0, 2 / 3, 1.0, 0.0])
    Rbcoeff2 = np.array([-3 / 36, -44 / 36, 1, 12 / 36, -1 / 36])

    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1, Lbcoeff2], (1, 1), 1]
    Rmat = [Rcoeff, [Rbcoeff0, Rbcoeff1, Rbcoeff2], (3, 3), -1]
    return Lmat, Rmat

def _coeffs_D1_X6_RIGHT():
    Lcoeff = np.array([829 / 1200, 1.0, 0.0])
    Rcoeff = np.array(
        [
            29 / 12000,
            -877 / 12000,
            -24271 / 14400,
            151 / 80,
            -307 / 2400,
            -167 / 36000,
            29 / 24000,
        ]
    )

    # JTT4 coefficients for boundary conditions
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbcoeff1 = np.array([0.25, 1.0, 0.25])
    Rbcoeff1 = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])

    # V_m2, O(h^5) coefficients for boundary conditions
    Lbcoeff2 = np.array([0, 2 / 3, 1.0, 0.0])
    Rbcoeff2 = np.array([-3 / 36, -44 / 36, 1, 12 / 36, -1 / 36])

    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1, Lbcoeff2], (1, 1), 1]
    Rmat = [Rcoeff, [Rbcoeff0, Rbcoeff1, Rbcoeff2], (3, 3), -1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B_LEFT():
    """
    Coefficients for the left side of the first derivative operator in the DSQ6B scheme.
    """
    Lcoeff = np.array([0.0, 1.0, 0.71391655])
    Rcoeff = np.array(
        [
            0.0044106875,
            -0.035642870,
            0.147435560,
            -0.54741579,
            -0.79751467,
            1.0738269,
            0.16644143,
            -0.011541239,
            0.0,
        ]
    )

    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Lcoeffb3 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

    Rcoeffb0 = np.array([-1.0, 4.0, -3.0]) / 2.0
    Rcoeffb1 = np.array([-1.0, 0.0, 1.0]) / 2.0
    Rcoeffb2 = np.array([-1.0, 6.0, -18.0, 10.0, 3.0])/ 12.0 
    Rcoeffb3 = np.array([-1.0, 6.0, -18.0, 10.0, 3.0]) / 12.0

    Lbands = (1, 1)
    Rbands = (4, 4)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2, Lcoeffb3], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2, Rcoeffb3], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B_RIGHT():
    """
    Coefficients for the left side of the first derivative operator in the DSQ6B scheme.
    """
    Lcoeff = np.array([0.71391655, 1.0, 0.0])
    Rcoeff = np.array(
        [
            0.0,
            0.011541239,
            -0.16644143,
            -1.0738269,
            0.79751467,
            0.54741579,
            -0.147435560,
            0.035642870,
            -0.0044106875,
        ]
    )

    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Lcoeffb3 = np.array([0.0, 0.0, 0.7139165, 1.0 ])

    Rcoeffb0 = np.array([-3.0, 4.0, -1.0]) / 2.0
    Rcoeffb1 = np.array([0.0, -3.0, 4.0, -1.0]) / 2.0
    Rcoeffb2 = np.array([0.0, -3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0
    Rcoeffb3 = Rcoeff[1:]
    Lbands = (1, 1)
    Rbands = (4, 4)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2, Lcoeffb3], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2, Rcoeffb3], Rbands, 1]
    return Lmat, Rmat


def build_derivative(N: int, dtype: DerivType):
    """
    Build the banded matrix for the specified derivative type.

    Parameters:
    N (int): Size of the matrix.
    dtype (DerivType): Type of derivative to build.

    Returns:
    np.ndarray: Banded matrix for the specified derivative.
    """
    if dtype == DerivType.D1_JT4:
        Pmat, Qmat = _coeffs_D1_JTT4()
    elif dtype == DerivType.D2_JT4:
        Pmat, Qmat = _coeffs_D2_JTT4()
    elif dtype == DerivType.D1_JT6:
        Pmat, Qmat = _coeffs_D1_JTT6()
    elif dtype == DerivType.D1_JP6:
        Pmat, Qmat = _coeffs_D1_JTP6()
    elif dtype == DerivType.D2_JP6:
        Pmat, Qmat = _coeffs_D2_JTP6()
    elif dtype == DerivType.D1_KP4:
        Pmat, Qmat = _coeffs_D1_KP4()
    elif dtype == DerivType.D1_BYUP6:
        Pmat, Qmat = _coeffs_D1_BYUP6([0.0, 0.0, 0.0, 0.0])
    elif dtype == DerivType.D2_BYUP6:
        Pmat, Qmat = _coeffs_D2_BYUP6([0.0, 0.0, 0.0, 0.0])
    elif dtype == DerivType.D1_Wm6:
        Pmat, Qmat = _coeffs_D1_Wm6()
    elif dtype == DerivType.D1_DSQ6A:
        Pmat, Qmat = _coeffs_D1_DSQ6A()
    elif dtype == DerivType.D2_DSQ6A:
        Pmat, Qmat = _coeffs_D2_DSQ6A()
    elif dtype == DerivType.D1_DSQ6B:
        Pmat, Qmat = _coeffs_D1_DSQ6B()
    elif dtype == DerivType.D2_DSQ6B:
        Pmat, Qmat = _coeffs_D2_DSQ6B()
    elif dtype == DerivType.D1_ME44:
        Pmat, Qmat = _coeffs_D1_ME44()
    elif dtype == DerivType.D2_ME44:
        Pmat, Qmat = _coeffs_D2_ME44()
    elif dtype == DerivType.D1_ME642:
        Pmat, Qmat = _coeffs_D1_ME642()
    elif dtype == DerivType.D2_ME642:
        Pmat, Qmat = _coeffs_D2_MEQ642()
    elif dtype == DerivType.D1_DSQ6B_LEFT:
        Pmat, Qmat = _coeffs_D1_DSQ6B_LEFT()
    elif dtype == DerivType.D1_DSQ6B_RIGHT:
        Pmat, Qmat = _coeffs_D1_DSQ6B_RIGHT()
    else:
        raise ValueError(f"Unsupported derivative type: {dtype}")

    explicit_derivs = [
        DerivType.D1_ME44,
        DerivType.D2_ME44,
        DerivType.D1_ME642,
        DerivType.D2_ME642,
    ]
    if dtype in explicit_derivs:
        P = np.ones(N)
        pbands = (0, 0)
    else:
        pparity = Pmat[3]
        pbands = Pmat[2]
        Pf = construct_banded_matrix(N, pbands[0], pbands[1], Pmat[0], Pmat[1], pparity)
        P = full_to_banded(Pf, pbands[0], pbands[1])

    qparity = Qmat[3]
    qbands = Qmat[2]
    Q = construct_banded_matrix(N, qbands[0], qbands[1], Qmat[0], Qmat[1], qparity)

    return [P, pbands], [Q, qbands]


def build_bh_derivative(N: int, dtype0: DerivType, dtypeBH: DerivType, mask):
    """
    Build a filter for boundary handling based on the specified type and parameters.
    Parameters:
        N (int): Size of the matrix.
        dtype0 (DerivType): Type of derivative for the interior region.
        dtypeBH (DerivType): Type of derivative for the boundary handling region.
        lambda (float): Mask function.  lambda = 1.0 corresponds to the background
                                        lambda = 0.0 corresponds to the BH region. 
    Returns:
        tuple: Banded matrices for the interior and boundary handling regions.
    """
    P0mat, Q0mat = build_derivative(N, dtype0)
    p0bands = P0mat[1]
    q0bands = Q0mat[1]
    P0 = P0mat[0]
    Q0 = Q0mat[0]

    PBHmat, QBHmat = build_derivative(N, dtypeBH)
    pbhbands = PBHmat[1]
    qbhbands = QBHmat[1]
    PBH = PBHmat[0]
    QBH = QBHmat[0]

    P = np.zeros((N, N), dtype=np.float64)
    Q = np.zeros((N, N), dtype=np.float64)
    if p0bands[0] == 0 and p0bands[1] == 0:
        P0f = np.identity(N, dtype=np.float64)
    else:
        P0f = banded_to_full_slow(P0, p0bands[0], p0bands[1], N, N)

    if pbhbands[0] == 0 and pbhbands[1] == 0:
        PBHf = np.identity(N, dtype=np.float64)
    else:
        # Convert the banded matrix PBH to a full matrix
        PBHf = banded_to_full_slow(PBH, pbhbands[0], pbhbands[1], N, N)

    for i in range(N):
        P[i, :] = mask[i] * P0f[i, :] + (1 - mask[i]) * PBHf[i, :]
        Q[i, :] = mask[i] * Q0[i, :] + (1 - mask[i]) * QBH[i, :]

    pbands = (max(p0bands[0], pbhbands[0]), max(p0bands[1], pbhbands[1]))
    qbands = (max(q0bands[0], qbhbands[0]), max(q0bands[1], qbhbands[1]))
    Pb = full_to_banded(P, pbands[0], pbands[1])

    return [Pb, pbands], [Q, qbands]
