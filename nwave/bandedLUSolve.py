'''----------------------------------------------------------------

This is code to solve banded matrix systems with LU decomposition
using LAPACK routines.  The scipy.linalg library does not include
support for banded systems.  This code is from a proposed extension
to scipy to work directly for banded systems from this page:

    https://github.com/scipy/scipy/issues/20948

Note, the code from github does not define the type LAndUBandCounts,
used in the code, and I replaced these with tuple[int, int].

   -------------------------------------------------------------'''

# === Imports ===

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lapack

# === Models ===

# an Enum class for special banded matrix structures that can be used for the
# determination of the best solver


class BandedMatrixStructures(str, Enum):
    """
    Defines the special banded matrix structures that can be used for the determination
    of the best solver, i.e.,

    - ``DIAGONAL``: a diagonal matrix
    - ``TRIDIAGONAL``: a tridiagonal matrix
    - ``GENERAL``: a general banded matrix

    """

    DIAGONAL = "diagonal"
    TRIDIAGONAL = "tridiagonal"
    GENERAL = "general"


# a dataclass for the factorization of a banded matrix with LU decomposition with p
# partial pivoting


@dataclass()
class BandedLUFactorization:
    """
    A dataclass that holds the partially pivoted LU factorization of a banded matrix.

    Attributes
    ----------
    lub: ndarray of shape (n_rows, n_cols)
        The LU factorization of the matrix ``A`` in banded storage format.
    ipiv: ndarray of shape (n_rows,)
        The pivot indices.
    l_and_u: tuple[int, int]
        The number of lower and upper bands in the LU factorization.
    singular: bool
        If ``True``, the matrix ``A`` is singular.
    structure: BandedMatrixStructures
        The structure of the matrix ``A``.
    shape : (int, int)
        The shape of the matrix ``A`` in dense form.
    n_rows, n_cols : int
        The number of rows and columns of the matrix ``A`` in dense form.
    main_diag_row_idx : int
        The index of the main diagonal in the banded storage format.

    """

    lub: np.ndarray
    ipiv: np.ndarray
    l_and_u: tuple[int, int]
    singular: bool
    structure: BandedMatrixStructures

    shape: tuple[int, int] = field(default=(-1, -1), init=False)
    num_rows: int = field(default=-1, init=False)
    num_cols: int = field(default=-1, init=False)
    main_diagonal_row_index: int = field(default=-1, init=False)

    def __post_init__(self):
        self.shape = self.lub.shape  # type: ignore
        self.num_rows, self.num_cols = self.shape
        self.main_diagonal_row_index = self.num_rows - 1 - self.l_and_u[0]

    def __iter__(self):
        return iter((self.lub, self.ipiv, self.l_and_u, self.singular, self.structure))


# === Auxiliary Functions ===


def _datacopied(arr, original):
    """
    Strictly check for ``arr`` not sharing any data with ``original``, under the
    assumption that ``arr = asarray(original)``

    Was copied from Scipy to be consistent in the LAPACK-wrappers,

    """

    if arr is original:
        return False

    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return False

    return arr.base is None
    

def _evaluate_banded_structure(l_and_u: tuple[int, int]) -> BandedMatrixStructures:
    """
    Evaluates the structure of a banded matrix based on the number of sub- and
    superdiagonals.

    Parameters
    ----------
    l_and_u : (int, int)
        The number of sub- (first) and superdiagonals (second element) aside the main
        diagonal which does not need to be considered here.

    Returns
    -------
    structure : BandedMatrixStructures
        The structure of the banded matrix based on the number of sub- and
        superdiagonals.

    """

    if l_and_u == (0, 0):
        return BandedMatrixStructures.DIAGONAL

    elif l_and_u == (1, 1):
        return BandedMatrixStructures.TRIDIAGONAL

    return BandedMatrixStructures.GENERAL


# === LAPACK-Wrappers for banded LU decomposition ===

def lu_banded(
    l_and_u: tuple[int, int],
    ab: ArrayLike,
    *,
    overwrite_ab: bool = False,
    check_finite: bool = True,
) -> BandedLUFactorization:
    """
    Computes the LU decomposition of a banded matrix ``A`` using LAPACK-routines.
    This function is a wrapper of the LAPACK-routine ``gttrf`` or  ``gbtrf`` which
    each compute the LU decomposition of a banded matrix ``A``. It wraps the routines in
    an analogous way to SciPy's ``scipy.linalg.cholesky_banded``.

    Parameters
    ----------
    l_and_u : (int, int)
        The number of "non-zero" sub- (first) and superdiagonals (second element) aside
        the main diagonal which does not need to be considered here. "Non-zero" can be
        a bit misleading in this context. These numbers should count up to the diagonal
        after which all following diagonals are all zero. Zero-diagonals that come
        before still need to be included.
        Neither of both may exceed ``num_rows``.
        Wrong specification of this can lead to non-zero-diagonals being ignored or
        zero-diagonals being included which corrupts the results or reduces the
        performance.
    ab : array_like of shape (l_and_u[0] + 1 + l_and_u[1], n)
        A 2D-Array resembling the matrix ``A`` in banded storage format (see Notes).
    overwrite_ab : bool, default=False
        If ``True``, the contents of ``ab`` can be overwritten by the routine.
        Otherwise, a copy of ``ab`` is created and overwritten.
        This only takes effect if ``ab`` is either diagonal or tridiagonal.
    check_finite : bool, default=True
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lub_factorization : BandedLUFactorization
        A dataclass containing the LU factorization of the matrix ``A`` as follows:

        - ``lub``: The LU decomposition of ``A`` in banded storage format (see Notes).
        - ``ipiv``: The pivoting indices.
        - ``l_and_u``: The number of sub- and superdiagonals of the matrix ``A`` that
            are non-zero.
        - ``singular``: A boolean indicating whether the matrix is singular.
        - ``structure``: The structure of the matrix ``A`` based on the number of sub-
            and superdiagonals. It is a member of the enumeration
            :class:`BandedMatrixStructures`.

        It can be unpacked as

        ```python
        lub, ipiv, l_and_u, singular, structure = lu_banded(l_and_u, ab)
        ```

    Notes
    -----
    For LAPACK's banded LU decomposition, the matrix ``a`` is stored in ``ab`` using the
    matrix diagonal ordered form:

        ```python
        ab[u + i - j, j] == a[i,j] # see below for u
        ```

    An example of ``ab`` (shape of a is ``(7,7)``, ``u``=3 superdiagonals, ``l``=2
    subdiagonals) looks like:

        ```python
             *    *    *   a03  a14  a25  a36
             *    *   a02  a13  a24  a35  a46
             *   a01  a12  a23  a34  a45  a56   # ^ superdiagonals
            a00  a11  a22  a33  a44  a55  a66   # main diagonal
            a10  a21  a32  a43  a54  a65   *    # v subdiagonals
            a20  a31  a42  a53  a64   *    *
        ```

    where all entries marked with `*` are zero elements although they will be set to
    arbitrary values by this function.

    Internally, the LAPACK routine ``gbtrf`` relies on an expanded version of this
    format to perform inplace operations that adds another ``l`` superdiagonals to the
    matrix in order to overwrite them for the purpose of pivoting. The output is thus an
    expanded version of the LU decomposition of ``A`` in the same format where the main
    diagonal of ``L`` is implicitly taken to be a vector of ones. The output can
    directly be used for the LAPACK-routine ``gbtrs`` to solve linear systems of
    equations based on this decomposition.

    ``gttrf`` uses a similar approach, but it works on a representation of the matrix
    as individual diagonal vectors. It also adds another superdiagonal to the matrix for
    pivoting purposes. The output is as well based on this representation, but it is
    transformed back into the banded storage format for the LU decomposition. This has
    to be considered when feeding the resulting factorisation into the linear solver
    ``gttrs``.

    """

    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        ab = np.asarray_chkfinite(ab)
    else:
        ab = np.asarray(ab)

    # then, the number of lower and upper subdiagonals needs to be checked for being
    # consistent with the shape of ``ab``
    num_subdiagonals, num_superdiagonals = l_and_u
    if num_subdiagonals + num_superdiagonals + 1 != ab.shape[0]:  # pragma: no cover
        raise ValueError(
            f"\nInvalid values for the number of sub- and super "
            f"diagonals: l+u+1 ({num_subdiagonals + num_superdiagonals + 1}) does not "
            f"equal ab.shape[0] ({ab.shape[0]})."
        )

    # the number of columns is also sanitised against the number of diagonals. It is not
    # possible to have less columns (rows in the dense matrix) than the maximum number
    # of diagonals
    if ab.shape[1] < max(num_subdiagonals, num_superdiagonals) + 1:
        raise ValueError(
            f"\nInvalid number of columns: ab.shape[1] ({ab.shape[1]}) is less than "
            f"the maximum number of diagonals "
            f"(max(l,u)+1={max(num_subdiagonals, num_superdiagonals) + 1})."
        )

    # for choosing the best possible routine, the structure of the matrix is evaluated
    structure = _evaluate_banded_structure(l_and_u)

    # if the matrix is diagonal, it is its own LU decomposition
    if structure == BandedMatrixStructures.DIAGONAL:
        if overwrite_ab:
            lub = ab
        else:
            lub = ab.copy()

        # there is no pivoting for a diagonal matrix
        ipiv = np.arange(
            start=0,
            stop=ab.shape[1],
            step=1,
            dtype=np.int64,
        )

        # the matrix is singular if any of the diagonal entries is zero
        info = 0
        singular_entries = ab[0, ::] == 0.0
        if singular_entries.any():
            info = np.where(singular_entries)[0][0] + 1

    # for any other case, the matrix needs to be factorised using a LAPACK-routine
    # for a tridiagonal matrix, the SciPy wrapper for ``gttrf`` should be used
    # NOTE: the special case of a tridiagonal matrix with only 2 columns needs to be
    #       handled as a general matrix because the pivoting with a second superdiagonal
    #       is not possible in this case
    elif structure == BandedMatrixStructures.TRIDIAGONAL and ab.shape[1] > 2:
        lapack_routine = "gttrf"
        (gttrf,) = lapack.get_lapack_funcs((lapack_routine,), (ab,))

        subdiagonal = ab[2, 0:-1]
        main_diagonal = ab[1, ::]
        superdiagonal_1 = ab[0, 1::]

        # ``gttrf`` works directly on the diagonals
        (
            subdiagonal,
            main_diagonal,
            superdiagonal_1,
            superdiagonal_2,
            ipiv,
            info,
        ) = gttrf(
            dl=subdiagonal,
            d=main_diagonal,
            du=superdiagonal_1,
            overwrite_dl=overwrite_ab,
            overwrite_d=overwrite_ab,
            overwrite_du=overwrite_ab,
        )

        # the diagonal structure is packed into a banded matrix
        lub = np.zeros(shape=(4, ab.shape[1]), dtype=ab.dtype)
        lub[0, 2::] = superdiagonal_2
        lub[1, 1::] = superdiagonal_1
        lub[2, ::] = main_diagonal
        lub[3, 0:-1] = subdiagonal
        # NOTE: ipiv seems to be not zero-based?
        ipiv -= 1

    # for a general matrix, the LAPACK-routine ``gbtrf`` is used
    # to make ``ab`` compatible with the shape the LAPACK expects in this case, it
    # needs to be re-written into a larger Array that has zeros elsewhere
    else:
        lapack_routine = "gbtrf"
        (gbtrf,) = lapack.get_lapack_funcs((lapack_routine,), (ab,))
        lpkc_ab = np.row_stack(
            (
                np.zeros((num_subdiagonals, ab.shape[1]), dtype=ab.dtype),
                ab,
            )
        )
        lub, ipiv, info = gbtrf(
            ab=lpkc_ab,
            kl=num_subdiagonals,
            ku=num_superdiagonals,
            overwrite_ab=True,
        )

    # finally, the results needs to be validated and returned
    # Case 1: the factorisation could be completed, which does not imply that the
    # solution can be used for solving a linear system
    if info >= 0:
        return BandedLUFactorization(
            lub=lub,
            ipiv=ipiv,
            l_and_u=l_and_u,
            singular=info > 0,
            structure=structure,
        )

    # Case 2: the factorisation was not completed due to invalid input
    raise ValueError(  # pragma: no cover # noqa: E501
        f"\nIllegal value in {-info}-th argument of internal gbtrf."
    )


def lu_solve_banded(
    lub_factorization: BandedLUFactorization,
    b: ArrayLike,
    *,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """
    Solves a linear system of equations ``Ax=b`` with a banded matrix ``A`` using its
    precomputed LU decomposition.
    This function wraps the LAPACK-routine ``gbtrs`` in an analogous way to SciPy's
    ``scipy.linalg.cho_solve_banded``.

    Parameters
    ----------
    lub_factorization : BandedLUFactorization
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.
    b : ndarray of shape (n,)
        A 1D-Array containing the right-hand side of the linear system of equations.
    overwrite_b : bool, default=False
        If ``True``, the contents of ``b`` can be overwritten by the routine. Otherwise,
        a copy of ``b`` is created and overwritten.
    check_finite : bool, default=True
        Whether to check that the input contains only finite numbers. Disabling may give
        a performance gain, but may result in problems (crashes, non-termination) if the
        inputs do contain infinities or NaNs.

    Returns
    -------
    x : ndarray of shape (n,)
        The solution to the system ``A x = b``.

    Raises
    ------
    LinAlgError
        If the system to solve is singular.

    """

    # if the matrix is singular, the solution cannot be computed
    if lub_factorization.singular:
        raise np.linalg.LinAlgError("\nSystem is singular.")

    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        lub_factorization.lub = np.asarray_chkfinite(lub_factorization.lub)
        lub_factorization.ipiv = np.asarray_chkfinite(lub_factorization.ipiv)
        b_internal = np.asarray_chkfinite(b)
    else:
        lub_factorization.lub = np.asarray(lub_factorization.lub)
        lub_factorization.ipiv = np.asarray(lub_factorization.ipiv)
        b_internal = np.asarray(b)

    overwrite_b = overwrite_b or _datacopied(b_internal, b)

    # then, the shapes of the LU decomposition and ``b`` need to be validated against
    # each other
    if lub_factorization.num_cols != b_internal.shape[0]:  # pragma: no cover
        raise ValueError(
            f"\nShapes of lub ({lub_factorization.num_cols}) and b "
            f"({b_internal.shape[0]}) are not compatible."
        )

    # now, the best routine is picked based on the structure of the matrix
    # for a diagonal matrix, the solution is just the division of the right-hand side by
    # the main diagonal
    if lub_factorization.structure == BandedMatrixStructures.DIAGONAL:
        if b_internal.ndim == 1:
            x = b_internal / lub_factorization.lub[0, ::]
        else:
            x = b_internal / lub_factorization.lub[0, ::].reshape((-1, 1))

        info = 0

    # for a tridiagonal matrix, the SciPy wrapper for ``gttrs`` should be used
    # NOTE: the special case of a tridiagonal matrix with only 2 columns needs to be
    #       handled as a general matrix because the pivoting with a second superdiagonal
    #       is not possible in this case
    elif (
        lub_factorization.structure == BandedMatrixStructures.TRIDIAGONAL
        and lub_factorization.num_cols > 2
    ):
        lapack_routine = "gttrs"
        (gttrs,) = lapack.get_lapack_funcs(
            (lapack_routine,), (lub_factorization.lub, b_internal)
        )

        # the solution is computed
        x, info = gttrs(
            dl=lub_factorization.lub[3, 0:-1],
            d=lub_factorization.lub[2, ::],
            du=lub_factorization.lub[1, 1::],
            du2=lub_factorization.lub[0, 2::],
            # NOTE: ipiv seems to be not zero-based?
            ipiv=lub_factorization.ipiv + 1,
            b=b_internal,
        )

    # for a general matrix, the LAPACK-routine ``gbtrs`` is used
    else:
        (gbtrs,) = lapack.get_lapack_funcs(
            ("gbtrs",), (lub_factorization.lub, b_internal)
        )
        x, info = gbtrs(
            ab=lub_factorization.lub,
            kl=lub_factorization.l_and_u[0],
            ku=lub_factorization.l_and_u[1],
            b=b_internal,
            ipiv=lub_factorization.ipiv,
            overwrite_b=overwrite_b,
        )

    # finally, the results needs to be validated and returned
    # Case 1: the solution could be computed truly successfully, i.e., without any
    # NaN-values
    if info == 0 and not np.isnan(x).any():
        return x

    # Case 2: the solution was computed, but there were NaN-values in it
    elif info == 0:
        raise np.linalg.LinAlgError("\nMatrix is singular.")

    # Case 3: the solution could not be computed due to invalid input
    elif info < 0:  # pragma: no cover
        raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrs.")

    # Case 4: unexpected error
    raise AssertionError(  # pragma: no cover
        f"\nThe internal gbtrs returned info > 0 ({info}) which should not happen."
    )

