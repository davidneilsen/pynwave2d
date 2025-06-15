import numpy as np
from scipy.linalg import solve_banded
from .utils import *
from .filters import Filter1D
from .types import *


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


def build_filter(N: int, ftype: FilterType, alpha=0.0, beta=0.0):
    """
    Build a filter based on the specified type and parameters.

    Parameters:
        ftype (FilterType): The type of filter to build.
        alpha (float): Parameter for JTT6, JTP6, JTT8, JTP8 filters.
        beta (float): Parameter for JTX6, JTX8 filters.

    Returns:
        Filter1D: The constructed filter object.
    """
    beta0 = 0.0

    if ftype == FilterType.JTT4:
        Pmat = _filter_jtt4_P(alpha)
        Qmat = _filter_jtt4_Q(alpha)
    elif ftype == FilterType.JTT6:
        Pmat = _filter_jtt6_P(alpha)
        Qmat = _filter_jtx6_Q(alpha, beta0)
    elif ftype == FilterType.JTP6:
        Pmat = _filter_jtp6_P(alpha, beta)
        Qmat = _filter_jtx6_Q(alpha, beta)
    elif ftype == FilterType.JTT8:
        Pmat = _filter_jtt8_P(alpha)
        Qmat = _filter_jtx8_Q(alpha, beta0)
    elif ftype == FilterType.JTP8:
        Pmat = _filter_jtp8_P(alpha, beta)
        Qmat = _filter_jtx8_Q(alpha, beta)
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

    pbands = Pmat[2]
    Px = construct_banded_matrix(N, pbands[0], pbands[1], Pmat[0], Pmat[1], Pmat[3])
    P = full_to_banded(Px, pbands[0], pbands[1])

    qbands = Qmat[2]
    Q = construct_banded_matrix(N, qbands[0], qbands[1], Qmat[0], Qmat[1], Qmat[3])

    return [P, pbands], [Q, qbands]


def build_bh_filter(N: int, ftype: FilterType, r, rbh, dr, alpha=0.0, beta=0.0):
    """
    Build a filter for boundary handling based on the specified type and parameters.
    """
    lambda0 = 1.0
    lambda1 = 0.0
    r0 = rbh - 2 * dr
    r1 = rbh - dr
    r2 = rbh + dr
    r3 = rbh + 2 * dr

    lamba = generalized_transition_profile(
        r, lambda0, lambda1, r0, r1, r2, r3, method="linear"
    )

    P0mat, Q0mat = build_filter(N, ftype, alpha, beta)
    p0bands = P0mat[1]
    q0bands = Q0mat[1]
    P0 = P0mat[0]
    Q0 = Q0mat[0]
    Qi = np.identity(N, dtype=np.float64)
    # Pi = full_to_banded(Qi, p0bands[0], p0bands[1])

    Pa = banded_to_full_slow(P0, p0bands[0], p0bands[1], N, N)
    P = np.zeros((N, N), dtype=np.float64)
    Q = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        P[i, :] = lamba[i] * Pa[i, :] + (1 - lamba[i]) * Qi[i, :]
        Q[i, :] = lamba[i] * Q0[i, :] + (1 - lamba[i]) * Qi[i, :]

    Pb = full_to_banded(P, p0bands[0], p0bands[1])

    return [Pb, p0bands], [Q, q0bands]
