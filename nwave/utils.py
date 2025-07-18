import numpy as np
from numba import njit


@njit
def full_to_banded_slow(A, kl, ku):
    """
    This routine converts a banded matrix A to the LAPACK banded matrix
    storage format.

    Inputs:
        A : a M x N matrix
        kl : The number of lower bands
        ku : The number of upper  bands

    Returns:
        AB = The LAPACK banded storage for A

    This is the "slow" version of the routine, because all loops
    are explicit.
    """
    M, N = A.shape
    AB = np.zeros((kl + ku + 1, N), dtype=A.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M - 1, j + kl)
        for i in range(i_min, i_max + 1):
            AB[ku + i - j, j] = A[i, j]
    return AB


@njit
def banded_to_full_slow(AB, kl, ku, M, N):
    """
    This routine converts a banded matrix in the LAPACK banded matrix
    storage format, AB, to a full matrix A of dimension (M, N).

    Inputs:
        AB : a banded matrix in LAPACK storage format
        kl : The number of lower bands
        ku : The number of upper  bands
        M  : matrix dimension
        N  : matrix dimension

    Returns:
        A = The full M x N matrix.  A has dimensions (M, N)

    This is the "slow" version of the routine, because all loops
    are explicit.
    """
    A = np.zeros((M, N), dtype=AB.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M - 1, j + kl)
        for i in range(i_min, i_max + 1):
            A[i, j] = AB[ku + i - j, j]
    return A


@njit
def full_to_banded(A, kl, ku):
    """
    This routine converts a banded matrix A to the LAPACK banded matrix
    storage format.

    Inputs:
        A : a M x N matrix
        kl : The number of lower bands
        ku : The number of upper  bands

    Returns:
        AB = The LAPACK banded storage for A

    This is the "fast" version of the routine, because the inner loop is
    vectorized using a numpy index range.
    """

    M, N = A.shape
    AB = np.zeros((kl + ku + 1, N), dtype=A.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M - 1, j + kl)
        i_indices = np.arange(i_min, i_max + 1)
        AB[ku + i_indices - j, j] = A[i_indices, j]
    return AB


@njit
def banded_to_full(AB, kl, ku, M, N) -> np.ndarray:
    """
    This routine converts a banded matrix in the LAPACK banded matrix
    storage format, AB, to a full matrix A of dimension (M, N).

    Inputs:
        AB : a banded matrix in LAPACK storage format
        kl : The number of lower bands
        ku : The number of upper  bands
        M  : matrix dimension
        N  : matrix dimension

    Returns:
        A = The full M x N matrix.  A has dimensions (M, N)

    This is the "fast" version of the routine, because the inner loop
    vectorized using a numpy index range.
    """

    A = np.zeros((M, N), dtype=AB.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M - 1, j + kl)
        i_indices = np.arange(i_min, i_max + 1)
        A[i_indices, j] = AB[ku + i_indices - j, j]
    return A


@njit
def construct_banded_matrix(N, kl, ku, coeffs, bcoeffs, parity):
    """
    Require boundary coefficients to be indexed from 0!

    Parameters
    ----------
    N : int
        The number of rows and columns in the matrix.
    kl : int
        The number of lower bands for the CENTER STENCIL, coeffs.
    ku : int
        The number of upper bands for the CENTER STENCIL, coeffs.
    coeffs : array_like
        Coefficients for the CENTER STENCIL, with length kl + ku + 1.
    bcoeffs : array_like
        Boundary coefficients for the top and bottom rows, with length p.
    parity : int
        Parity for the boundary coefficients, either 1 or -1.
    """

    assert kl == ku, "This function assumes kl == ku"
    assert len(coeffs) == kl + ku + 1, "coeffs must have length kl + ku + 1"

    A = np.zeros((N, N))
    p = len(bcoeffs)

    for i in range(N):
        if i < p:
            # Apply custom boundary coefficients at the top
            b = bcoeffs[i]
            m = len(b)
            start = 0
            end = min(N, start + m)
            A[i, start:end] = b
        elif i >= N - p:
            # Apply reflected boundary coefficients at the bottom
            idx = N - 1 - i
            b = parity * bcoeffs[idx][::-1]
            m = len(b)
            # end = min(N, i + (m // 2) + 1)
            end = N
            start = end - m
            A[i, start:end] = b
        else:
            # Interior: use banded structure
            # for offset in range(-kl, ku + 1):
            #    j = i + offset
            #    if 0 <= j < N:
            #        A[i, j] = coeffs[offset + kl]
            A[i, i - kl : i + ku + 1] = coeffs[:]
    return A


def Xconstruct_lapack_banded(N, kl, ku, coeffs, bcoeffs, parity):
    """
    This routine was written by ChatGPT, but it doesn't work as expected.
    """
    assert kl == ku, "This function assumes kl == ku"
    assert len(coeffs) == 2 * kl + 1, "coeffs must have length 2*kl + 1"

    ab = np.zeros((2 * kl + 1, N))
    p = len(bcoeffs)

    for i in range(N):
        if i < p:
            # Apply top boundary rows
            b = bcoeffs[i]
            m = len(b)
            # offset_start = max(0, i - (m // 2))
            offset_start = 0
            for j in range(m):
                col = offset_start + j
                row = kl + i - col
                if 0 <= col < N and 0 <= row < 2 * kl + 1:
                    ab[row, col] = b[j]
        elif i >= N - p:
            # Apply bottom boundary rows
            idx = N - 1 - i
            b = parity * bcoeffs[idx][::-1]
            m = len(b)
            # offset_start = i - (m // 2)
            offset_start = N
            for j in range(m):
                col = offset_start + j
                row = kl + i - col
                if 0 <= col < N and 0 <= row < 2 * kl + 1:
                    ab[row, col] = b[j]
        else:
            # Interior rows: use uniform banded coefficients
            for offset in range(-kl, kl + 1):
                row = kl - offset
                col = i
                j = i + offset
                if 0 <= j < N:
                    ab[row, j] = coeffs[kl + offset]

    return ab


def l2norm(u):
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.mean(u**2))


@njit
def l2norm_mask(u, dx, mask):
    """
    Compute the L2 norm of an array.
    """
    sum = 0.0
    for i in range(len(u)):
        if mask[i] > 0.5:
            sum += dx * u[i] ** 2

    return np.sqrt(sum)


def smooth_transition(s, method):
    """
    Compute the transition function f(s) for s in [0, 1]
    """
    if method == "linear":
        return s
    elif method == "cubic":
        return 3 * s**2 - 2 * s**3
    elif method == "tanh":
        # Map s in [0,1] to [-1,1] for symmetric tanh
        return 0.5 * (1 + np.tanh(10 * (s - 0.5)))
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'linear', 'cubic', or 'tanh'."
        )


def generalized_transition_profile(x, a0, a1, x0, x1, x2, x3, method="tanh"):
    """
    Returns a numpy array a(x) with values:
      a0 for x < x0,
      transition a0 -> a1 in (x0, x1),
      a1 for x1 <= x <= x2,
      transition a1 -> a0 in (x2, x3),
      a0 for x > x3.

    Parameters:
        x      : numpy array of coordinates
        a0     : outer value
        a1     : inner plateau value
        x0,x1  : start and end of rising transition
        x2,x3  : start and end of falling transition
        method : 'linear', 'cubic', or 'tanh'

    Returns:
        a : numpy array
    """
    if not (x0 < x1 <= x2 < x3):
        raise ValueError("Require x0 < x1 <= x2 < x3 for valid transitions.")

    a = np.full_like(x, a0, dtype=np.float64)

    # Rising transition: a0 -> a1
    mask_rise = (x >= x0) & (x < x1)
    s_rise = (x[mask_rise] - x0) / (x1 - x0)
    a[mask_rise] = a0 + smooth_transition(s_rise, method) * (a1 - a0)

    # Plateau: a1
    mask_plateau = (x >= x1) & (x <= x2)
    a[mask_plateau] = a1

    # Falling transition: a1 -> a0
    mask_fall = (x > x2) & (x < x3)
    s_fall = (x[mask_fall] - x2) / (x3 - x2)
    a[mask_fall] = a1 - smooth_transition(s_fall, method) * (a1 - a0)

    return a


def write_matrix_to_file(filename, matrix):
    """
    Writes a NumPy 2D array to a text file.

    Each row is written on a single line with elements formatted as '%.1e'.

    Parameters:
        filename (str): Name of the output file.
        matrix (np.ndarray): 2D array to write.
    """
    with open(filename, "w") as f:
        for row in matrix:
            line = " ".join(f"{val:.1e}" for val in row)
            f.write(line + "\n")


@njit
def Xbanded_matvec(N, kl, ku, b: np.ndarray, Ab: np.ndarray, x: np.ndarray):
    # f = Ab x.  Ab is the banded matrix in LAPACK format.
    for i in range(N):
        b[i] = 0.0
        for offset in range(-kl, ku + 1):
            j = i + offset
            if 0 <= j < N:
                b[i] += Ab[offset + kl, j] * x[j]


@njit(fastmath=True, cache=True)
def banded_matvec(N, kl, ku, Ab, x, alpha):
    """
    Perform matrix-vector multiplication with a banded matrix in LAPACK format.

    Returns the result of the operation b = alpha * Ab * x

    Parameters
    ----------
    N : int
        The number of rows and columns in the matrix Ab.
    kl : int
        The number of lower bands for Ab.
    ku : int
        The number of upper bands for Ab.
    Ab : ndarray
        The banded matrix in LAPACK storage format, with shape (kl + ku +
        1, N).
    x : ndarray or matrix
        The input vector of length N.
    alpha : float
        A scaling factor to multiply the result.

    """
    B = np.zeros_like(x)
    if x.ndim == 1:
        for i in range(N):
            for j in range(max(0, i - kl), min(N, i + ku + 1)):
                B[i] += Ab[ku + i - j, j] * x[j] * alpha
    elif x.ndim == 2:
        # x shape: (N, NCOL)
        for col in range(x.shape[1]):
            for i in range(N):
                for j in range(max(0, i - kl), min(N, i + ku + 1)):
                    B[i, col] += Ab[ku + i - j, j] * x[j, col] * alpha
    else:
        raise ValueError("x must be a 1D or 2D array")

    return B


@njit
def construct_banded_matrix_numba(N, kl, ku, coeffs, bcoeffs, parity):
    assert kl == ku
    assert len(coeffs) == kl + ku + 1

    A = np.zeros((N, N))
    p = len(bcoeffs)

    for i in range(N):
        if i < p:
            b = bcoeffs[i]
            m = len(b)
            for j in range(m):
                if j < N:
                    A[i, j] = b[j]
        elif i >= N - p:
            idx = N - 1 - i
            b = bcoeffs[idx]
            m = len(b)
            for j in range(m):
                col = N - m + j
                if col < N:
                    A[i, col] = parity * b[m - 1 - j]
        else:
            for offset in range(-kl, ku + 1):
                j = i + offset
                if 0 <= j < N:
                    A[i, j] = coeffs[offset + kl]

    return A


def linear_interpolation(f, x, x0):
    """
    Perform linear interpolation for the value of f at x0 given arrays f and x.

    Parameters
    ----------
    f : array_like
        Array of function values.
    x : array_like
        Array of x values (must be sorted in ascending order).
    x0 : float
        The x value at which to interpolate.

    Returns
    -------
    float
        Interpolated value f(x0).
    """
    if x0 <= x[0]:
        return f[0]
    if x0 >= x[-1]:
        return f[-1]
    idx = np.searchsorted(x, x0) - 1
    x1, x2 = x[idx], x[idx + 1]
    f1, f2 = f[idx], f[idx + 1]
    return f1 + (f2 - f1) * (x0 - x1) / (x2 - x1)


def write_array(f, array, values_per_line=6):
    """Write values from an array with line breaks for readability and parser safety."""
    for i in range(0, len(array), values_per_line):
        chunk = array[i : i + values_per_line]
        f.write(" ".join(f"{val:.8e}" for val in chunk) + "\n")


def write_vtk_rectilinear_grid(step, data_list, x, y, data_names, time, output_dir):
    """
    Write 2D numpy arrays as legacy VTK rectilinear grid (ASCII) for Paraview.
    """
    nx = len(x)
    ny = len(y)
    npoints = nx * ny

    # Make sure output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/maxwell_{step:05d}.vtk"

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Rectilinear grid data\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")

        # Coordinates
        f.write(f"X_COORDINATES {nx} float\n")
        write_array(f, x)

        f.write(f"Y_COORDINATES {ny} float\n")
        write_array(f, y)

        f.write("Z_COORDINATES 1 float\n")
        f.write("0.0\n")

        # Time field (optional)
        if time is not None:
            f.write(f"FIELD FieldData 1\n")
            f.write(f"TIME 1 1 float\n{time:.8e}\n")

        # Point data
        f.write(f"POINT_DATA {npoints}\n")
        for arr, name in zip(data_list, data_names):
            assert arr.shape == (
                nx,
                ny,
            ), f"Data array {name} must have shape ({nx}, {ny})"
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            arr_flat = arr.flatten(order="F")  # VTK expects Fortran-style ordering
            write_array(f, arr_flat)

        # Ensure a final newline
        f.write("\n")
