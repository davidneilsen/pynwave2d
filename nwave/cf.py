import numpy as np
from scipy.linalg import solve_banded
from .utils import *
from .filters import Filter1D
from .types import *

CompactFilterTypes = {
    FilterType.JTT4,
    FilterType.JTT6,
    FilterType.JTP6,
    FilterType.JTT8,
    FilterType.JTP8,
    FilterType.KP4,
    FilterType.F2,
}


class NCompactFilter(Filter1D):
    """
    A compact filter class.
    """

    def __init__(
        self,
        N,
        apply_filter: FilterApply,
        filter_type: FilterType,
        method: CFDSolve,
        Pmat,
        Qmat,
        frequency,
        dx,
        mask=None,
    ):
        self.N = N
        self.filter_type = filter_type
        self.apply_filter = apply_filter
        self.method = method
        self.frequency = frequency
        self.P = Pmat[0]
        self.Pbands = Pmat[1]
        self.Q = Qmat[0]
        self.Qbands = Qmat[1]
        self.mask = mask
        self.overwrite = True
        self.checkf = True
        self.dx = dx

        self.Qb = full_to_banded(Qmat[0], Qmat[1][0], Qmat[1][1])
        Qf = banded_to_full(self.Qb, self.Qbands[0], self.Qbands[1], self.N, self.N)
        if not np.allclose(Qf, self.Q):
            raise ValueError(
                "NCompactFilter init: Q matrix is not banded correctly.  Check Qbands and Q matrix."
            )

        super().__init__(dx, apply_filter, filter_type, frequency)

    @classmethod
    def init_filter(
        cls,
        r : np.ndarray,
        filter_type: FilterType,
        apply_filter: FilterApply,
        method: CFDSolve,
        frequency: int,
        filter_bounds : bool,
        alpha : float,
        beta=0.0,
        ko_sigma=0.4,
    ):
        """
        Build a filter based on the specified type and parameters.

        Parameters:
            ftype (FilterType): The type of filter to build.
            alpha (float): Parameter for JTT6, JTP6, JTT8, JTP8 filters.
            beta (float): Parameter for JTX6, JTX8 filters.

        Returns:
            Filter1D: The constructed filter object.
        """
        N = len(r)
        dr = r[1] - r[0]

        if apply_filter == FilterApply.RHS:
            ko_filter = True
        else:
            ko_filter = False

        Pmat, Qmat = build_filter(r, filter_type, filter_bounds, alpha, beta, ko_filter, ko_sigma)
        return cls(N, apply_filter, filter_type, method, Pmat, Qmat, frequency, dr)

    @classmethod
    def init_bh_filter(
        cls,
        r : np.ndarray,
        ftype: FilterType,
        apply_filter: FilterApply,
        method: CFDSolve,
        frequency: int,
        mask_pos : float,
        mask_width : float,
        filter_bounds : bool,
        alpha : float,
        beta=0.0,
        ko_sigma=0.4,
    ):

        N = len(r)
        dr = r[1] - r[0]

        if apply_filter == FilterApply.RHS:
            ko_filter = True
        else:
            ko_filter = False

        mask0 = 1.0
        maskBH = 0.0
        r0 = mask_pos - 2 * mask_width
        r1 = mask_pos - mask_width
        r2 = mask_pos + mask_width
        r3 = mask_pos + 2 * mask_width
        mask = generalized_transition_profile(
            r, mask0, maskBH, r0, r1, r2, r3, method="tanh"
        )
        Pmat, Qmat = build_bh_filter(N, ftype, r, mask, filter_bounds, alpha, beta, ko_filter, ko_sigma)
        return cls(N, apply_filter, ftype, method, Pmat, Qmat, frequency, dr, mask)

    def get_filter_type(self):
        return self.filter_type

    def get_apply_filter(self):
        return self.apply_filter

    def get_frequency(self):
        return self.frequency

    def get_P(self):
        """
        Get the P matrix.
        """
        return self.P

    def get_Pbands(self):
        """
        Get the P bands.
        """
        return self.Pbands

    def get_Q(self):
        """
        Get the Q matrix.
        """
        return self.Q

    def get_Qbands(self):
        """
        Get the Q bands.
        """
        return self.Qbands

    def filter(self, u):
        """
        Apply the filter to the input array u.
        """
        if len(u) != self.N:
            raise ValueError(
                f"Input array length {len(u)} does not match filter size {self.N}."
            )

        if self.method == CFDSolve.SCIPY:
            # rhs = np.matmul(self.Q, u)
            rhs = banded_matvec(self.N, self.Qbands[0], self.Qbands[1], self.Qb, u, 1.0)
            u_f = solve_banded(self.Pbands, self.P, rhs)
        elif self.method == CFDSolve.LUSOLVE:
            # rhs = np.matmul(self.Q, u)
            rhs = banded_matvec(self.N, self.Qbands[0], self.Qbands[1], self.Qb, u, 1.0)
            u_f = solve_banded(
                self.Pbands,
                self.P,
                rhs,
                overwrite_ab=self.overwrite,
                check_finite=self.checkf,
            )
        else:
            raise ValueError(
                f"Unsupported filter application type: {self.apply_filter}"
            )

        return u_f


def _filter_F2_Q(alpha):
    Q_coeffs = 0.5 * np.array(
        [
            (1 + 2 * alpha) / 2,
            (1 + 2 * alpha),
            (1 + 2 * alpha) / 2,
        ]
    )

    bcoeffs0 = np.array([0.5 + 0.5 * alpha, 0.5 + 0.5 * alpha])
    Q_bcoeffs = [bcoeffs0]

    return [Q_coeffs, Q_bcoeffs, (1, 1), 1]


def _filter_F2_P(alpha):
    P_coeffs = np.array([alpha, 1.0, alpha])
    P_bcoeffs = [np.array([1.0, alpha])]

    return [P_coeffs, P_bcoeffs, (1, 1), 1]


def _filter_jtt4_Q(alpha):
    Q_coeffs = 0.5 * np.array(
        [
            (-1 + 2 * alpha) / 8,
            (1 + 2 * alpha) / 2,
            (5 + 6 * alpha) / 4,
            (1 + 2 * alpha) / 2,
            (-1 + 2 * alpha) / 8,
        ]
    )

    a10 = (1 + 14 * alpha) / 16
    a11 = (3 + 2 * alpha) / 4
    a12 = (3 + 2 * alpha) / 8
    a13 = (-1 + 2 * alpha) / 4
    a14 = (1 - 2 * alpha) / 16

    bcoeffs0 = np.array([1.0, 0.0, 0.0])
    bcoeffs1 = np.array([a10, a11, a12, a13, a14])
    Q_bcoeffs = [bcoeffs0, bcoeffs1]

    return [Q_coeffs, Q_bcoeffs, (2, 2), 1]


def _filter_jtt4_P(alpha):
    P_coeffs = np.array([alpha, 1.0, alpha])
    P_bcoeffs0 = np.array([1.0, 0.0])
    P_bcoeffs1 = np.array([alpha, 1.0, alpha])
    P_bcoeffs = [P_bcoeffs0, P_bcoeffs1]

    return [P_coeffs, P_bcoeffs, (1, 1), 1]


def _filter_jtx6_Q(alpha, beta):
    Q_coeffs = 0.5 * np.array(
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

    Q_bcoeffs = [bcoeffs0, bcoeffs1, bcoeffs2]
    return [Q_coeffs, Q_bcoeffs, (3, 3), 1]


def _filter_jtt6_P(alpha):
    P_coeffs = np.array([alpha, 1.0, alpha])
    P_bcoeffs = [np.array([1.0, alpha])]

    return [P_coeffs, P_bcoeffs, (1, 1), 1]


def _filter_jtp6_P(alpha, beta):
    P_coeffs = np.array([beta, alpha, 1.0, alpha, beta])

    P_bcoeffs0 = np.array([1.0, alpha, beta])
    P_bcoeffs1 = np.array([alpha, 1.0, alpha, beta])
    P_bcoeffs = [P_bcoeffs0, P_bcoeffs1]

    return [P_coeffs, P_bcoeffs, (2, 2), 1]


def _filter_jtx8_Q(alpha, beta):
    Q_coeffs = 0.5 * np.array(
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

    Q_bcoeffs = [bcoeffs0, bcoeffs1, bcoeffs2, bcoeffs3]
    return [Q_coeffs, Q_bcoeffs, (4, 4), 1]


def _filter_jtt8_P(alpha):
    P_coeffs = np.array([alpha, 1.0, alpha])
    P_bcoeffs0 = np.array([1.0, alpha])
    P_bcoeffs = [P_bcoeffs0]
    return [P_coeffs, P_bcoeffs, (1, 1), 1]


def _filter_jtp8_P(alpha, beta):
    P_coeffs = np.array([beta, alpha, 1.0, alpha, beta])
    P_bcoeffs0 = np.array([1.0, alpha, beta])
    P_bcoeffs1 = np.array([alpha, 1.0, alpha, beta])
    P_bcoeffs2 = np.array([beta, alpha, 1.0, alpha, beta])
    P_bcoeffs = [P_bcoeffs0, P_bcoeffs1, P_bcoeffs2]
    return [P_coeffs, P_bcoeffs, (2, 2), 1]


def build_filter(
    r, ftype: FilterType, filter_bounds, alpha, beta, kreiss_oliger, ko_sigma
):
    """
    Build a filter based on the specified type and parameters.

    Parameters:
        ftype (FilterType): The type of filter to build.
        alpha (float): Parameter for JTT6, JTP6, JTT8, JTP8 filters.
        beta (float): Parameter for JTX6, JTX8 filters.

    Returns:
        Filter1D: The constructed filter object.
    """
    N = len(r)
    idr = 1.0 / (r[1] - r[0])  # inverse of the grid spacing

    if ftype == FilterType.JTT4:
        Pmat = _filter_jtt4_P(alpha)
        Qmat = _filter_jtt4_Q(alpha)
    elif ftype == FilterType.JTT6:
        Pmat = _filter_jtt6_P(alpha)
        Qmat = _filter_jtx6_Q(alpha, beta)
    elif ftype == FilterType.JTP6:
        Pmat = _filter_jtp6_P(alpha, beta)
        Qmat = _filter_jtx6_Q(alpha, beta)
    elif ftype == FilterType.JTT8:
        Pmat = _filter_jtt8_P(alpha)
        Qmat = _filter_jtx8_Q(alpha, beta)
    elif ftype == FilterType.JTP8:
        Pmat = _filter_jtp8_P(alpha, beta)
        Qmat = _filter_jtx8_Q(alpha, beta)
    elif ftype == FilterType.F2:
        Pmat = _filter_F2_P(alpha)
        Qmat = _filter_F2_Q(alpha)
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

    pcoeffs = Pmat[0]
    qcoeffs = Qmat[0]
    pbounds = Pmat[1]
    qbounds = Qmat[1]

    if kreiss_oliger:
        if filter_bounds:
            raise ValueError(
                "Kreiss-Oliger filter does not support filter bounds. Set filter_bounds to False."
            )
        imid = len(qcoeffs) // 2
        qcoeffs[imid] -= 1.0
        if abs(alpha) > 1.0e-8:
            qcoeffs[imid - 1] -= alpha
            qcoeffs[imid + 1] -= alpha
        if abs(beta) > 1.0e-8:
            qcoeffs[imid - 2] -= beta
            qcoeffs[imid + 2] -= beta
        qcoeffs *= ko_sigma * idr

    if filter_bounds == False:
        pbident = []
        qbident = []
        for i in range(len(pbounds)):
            prow = np.zeros(i + 1, dtype=np.float64)
            prow[i] = 1.0
            pbident.append(prow)
            if kreiss_oliger:
                qrow = np.zeros_like(prow, dtype=np.float64)
                qrow *= ko_sigma * idr
            else:
                qrow = prow.copy()
            qbident.append(qrow)
        pbounds = pbident
        qbounds = qbident

    pbands = Pmat[2]
    print(f"pbands = {pbands}")
    print(f"pbounds = {pbounds}")
    Px = construct_banded_matrix_numba(
        N, pbands[0], pbands[1], pcoeffs, pbounds, Pmat[3]
    )
    P = full_to_banded(Px, pbands[0], pbands[1])

    qbands = Qmat[2]
    qparity = Qmat[3]
    # count the number of bands for the boundary terms in the Q matrix
    nqb = 0
    for iq in range(len(qbounds)):
        nqb = max(nqb, len(qbounds[iq]) - 1 - iq)

    # set the total number of bands for the Q matrix
    qtotbands = (max(nqb, qbands[0]), max(nqb, qbands[1]))
    Q = construct_banded_matrix_numba(
        N, qbands[0], qbands[1], qcoeffs, qbounds, qparity
    )

    return [P, pbands], [Q, qtotbands]


def build_bh_filter(N: int, ftype: FilterType, r, mask, filter_bounds, alpha, beta, kreiss_oliger, ko_sigma):
    """
    Build a filter for boundary handling based on the specified type and parameters.
    """

    P0mat, Q0mat = build_filter(r, ftype, filter_bounds, alpha, beta, kreiss_oliger, ko_sigma)
    p0bands = P0mat[1]
    q0bands = Q0mat[1]
    P0b = P0mat[0]
    P0 = banded_to_full(P0b, p0bands[0], p0bands[1], N, N)
    Q0 = Q0mat[0]

    ftypeBH = FilterType.JTT4
    PBHmat, QBHmat = build_filter(r, ftypeBH, filter_bounds, alpha, beta, kreiss_oliger, ko_sigma)
    PBHb = PBHmat[0]
    pBHbands = PBHmat[1]
    qBHbands = QBHmat[1]
    PBH = banded_to_full(PBHb, pBHbands[0], pBHbands[1], N, N)
    QBH = QBHmat[0]

    Ident = np.identity(N, dtype=np.float64)
    P = np.zeros((N, N), dtype=np.float64)
    Q = np.zeros((N, N), dtype=np.float64)

    blend_filters(P, mask, P0, PBH)
    blend_filters(Q, mask, Q0, QBH)

    pbands = (max(p0bands[0], pBHbands[0]), max(p0bands[1], pBHbands[1]))
    qbands = (max(q0bands[0], qBHbands[0]), max(q0bands[1], qBHbands[1]))
    Pb = full_to_banded(P, pbands[0], pbands[1])

    return [Pb, pbands], [Q, qbands]


@njit
def blend_filters(C, alpha, A, B):
    """
    compute C = alpha * A + (1-alpha) * B
    """
    N = len(alpha)
    for i in range(N):
        for j in range(N):
            C[i, j] = alpha[i] * A[i, j] + (1.0 - alpha[i]) * B[i, j]
