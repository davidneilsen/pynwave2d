import numpy as np

def full_to_banded_slow(A, kl, ku):
    '''
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
    '''
    M, N = A.shape
    AB = np.zeros((kl + ku + 1, N), dtype=A.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M-1, j + kl)
        for i in range(i_min, i_max + 1):
            AB[ku + i - j, j] = A[i, j]
    return AB

def banded_to_full_slow(AB, kl, ku, M, N):
    '''
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
    '''
    A = np.zeros((M, N), dtype=AB.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M-1, j + kl)
        for i in range(i_min, i_max + 1):
            A[i, j] = AB[ku + i - j, j]
    return A

def full_to_banded(A, kl, ku):
    '''
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
    '''
 
    M, N = A.shape
    AB = np.zeros((kl + ku + 1, N), dtype=A.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M-1, j + kl)
        i_indices = np.arange(i_min, i_max + 1)
        AB[ku + i_indices - j, j] = A[i_indices, j]
    return AB

def banded_to_full(AB, kl, ku, M, N):
    '''
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
    '''
 
    A = np.zeros((M, N), dtype=AB.dtype)
    for j in range(N):
        i_min = max(0, j - ku)
        i_max = min(M-1, j + kl)
        i_indices = np.arange(i_min, i_max + 1)
        A[i_indices, j] = AB[ku + i_indices - j, j]
    return A

def l2norm(u):
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.sum(u**2)/u.size)