import numpy as np
import tomllib
import matplotlib.pyplot as plt
import sys
import os
import time

"""
This script is to get performance data (timing) for different 
derivative types and ways of solving the linear systems.
The speed of different solution methods does depend on the array
sizes.  

In general, it is difficult to consistently beat SCIPY by a meaningful
amount of time.

Inverting the matrix P to build P^{-1}Q becomes very expensive as N > 10000.

Calling the BLAS banded matrix-vector multiply (DGBMV) through SCIPY 
is almost always slow.  numpy's matmul is very fast.  My own banded matrix
multiplier is about the same as np.matmul, which has the advantage
of contiguous memory access for vector instructions.

"""

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import *

DEBUG = False

def func1(x):
    k1 = 3.0
    k2 = 1.0
    f = np.cos(x) + np.sin(k1 * x)
    dxf = k1 * np.cos(k1 * x) - np.sin(x)
    dxxf = -np.cos(x) - k1**2 * np.sin(k1 * x)
    return f, dxf, dxxf


def func2(x):
    """Simpler test for testing 'symmetry' to make sure x and y give the same values"""
    freq = (1.0 / 2.0) * np.pi
    f = np.cos(freq * x)

    # first order analytical derivative
    dxf = -freq * np.sin(freq * x)

    # second order analytical derivative
    dxxf = -freq * freq * np.cos(freq * x)

    return f, dxf, dxxf

def func3(x):
    """Test function similar to BSSN variables"""
    epsilon = 1.72345769e-5
    f = -2/(x + epsilon)
    dxf = 2 / ((x + epsilon) ** 2)
    dxxf = -4 / ((x + epsilon) ** 3)
    return f, dxf, dxxf


# with open("params1D.toml", "rb") as f:
#    params = tomllib.load(f)

nx0 = 20
level = 9
nx = int((2**level)*nx0 + 1)
params = { "Nx": nx, "Xmin": -4.0, "Xmax": 4.0 }

g = Grid1D(params)
dx = g.dx[0]
x = g.xi[0]

D1 = ExplicitFirst642_1D(dx)

C1 = CompactFirst1D(x, DerivType.D1_JP6, CFDSolve.LUSOLVE)
V1 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.SCIPY)
V2 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.PENTAPY)
V3 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.LUSOLVE)
V4 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.D_LU)
V5 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.D_INV)


f, dxf, dxxf = func1(x)

DXF = D1.grad(f)
t0 = time.perf_counter()
CXF = C1.grad(f)
t1 = time.perf_counter()
V1XF = V1.grad(f)
t2 = time.perf_counter()
V2XF = V2.grad(f)
t3 = time.perf_counter()
V3XF = V3.grad(f)
t4 = time.perf_counter()
V4XF = V4.grad(f)
t5 = time.perf_counter()
V5XF = V5.grad(f)
t6 = time.perf_counter()

et0 = t1 - t0
et1 = t2 - t1
et2 = t3 - t2
et3 = t4 - t3
et4 = t5 - t4
et5 = t6 - t5
print(f"Nx     : {nx}")
print("Timing for Pentadiagonal Systems")
print(f"Time C1: {et0:.2e} s.")
print(f"Time V1 SCIPY: {et1:.2e} s.")
print(f"Time V2 PENTAPY: {et2:.2e} s.")
print(f"Time V3 LUSOLVE: {et3:.2e} s.")
print(f"Time V4 DLU: {et4:.2e} s.")
print(f"Time V5 DINV (should be same as DLU): {et5:.2e} s.")

#---------------
#  Tridiagonal
#
#---------------

T1 = CompactFirst1D(x, DerivType.D1_JT6, CFDSolve.LUSOLVE)
U1 = NCompactDerivative.deriv(x, DerivType.D1_JT6, CFDSolve.SCIPY)
U3 = NCompactDerivative.deriv(x, DerivType.D1_JT6, CFDSolve.LUSOLVE)
U4 = NCompactDerivative.deriv(x, DerivType.D1_JT6, CFDSolve.D_LU)
U5 = NCompactDerivative.deriv(x, DerivType.D1_JT6, CFDSolve.D_INV)

t10 = time.perf_counter()
TXF = T1.grad(f)
t11 = time.perf_counter()
U1XF = U1.grad(f)
t12 = time.perf_counter()
U3XF = U3.grad(f)
t13 = time.perf_counter()
U4XF = U4.grad(f)
t14 = time.perf_counter()
U5XF = U5.grad(f)
t15 = time.perf_counter()

et10 = t11 - t10
et11 = t12 - t11
et12 = t13 - t12
et13 = t14 - t13
et14 = t15 - t14
print(f"Nx     : {nx}")
print("Timing for Tridiagonal Systems")
print(f"Time T1: {et10:.2e} s.")
print(f"Time U1 SCIPY: {et11:.2e} s.")
print(f"Time U3 LUSOLVE: {et12:.2e} s.")
print(f"Time U4 DLU: {et13:.2e} s.")
print(f"Time U5 D_INV: {et14:.2e} s.")

Q = V3.get_Q()
qb = V3.get_Qbands()
kl = qb[0]
ku = qb[1]
Qx = full_to_banded(Q, kl, ku)


t20 = time.perf_counter()
rhs = np.matmul(Q,f)
t21 = time.perf_counter()
rhs2 = banded_matvec(nx, kl, ku, Qx, f)
t22 = time.perf_counter()

print(f"Full MV multiply: {t21-t20:.2e} s.")
print(f"Banded MV multiply: {t22-t21:.2e} s.")


"""
p1 = V1.get_P()
p1bands = V1.get_Pbands()
kl = p1bands[0]
ku = p1bands[1]
p1f = banded_to_full_slow(p1, kl, ku, nx, nx)
q1 = V1.get_Q()

if nx < 21 and DEBUG:
    write_matrix_to_file("P1mat.dat", p1f)
    write_matrix_to_file("Q1mat.dat", q1)

err_dxf = DXF - dxf
err_cxf = CXF - dxf
err_v1xf = V1XF - dxf
err_v2xf = V2XF - dxf
err_v3xf = V3XF - dxf
err_v4xf = V4XF - dxf

print("Error in x compact derivative:", np.linalg.norm(err_cxf))
print("Error in x derivative:", np.linalg.norm(err_dxf))

c1d = np.abs(err_cxf)
e1d = np.abs(err_dxf)
v1d = np.abs(err_v1xf)

fig, ax = plt.subplots()
ax.semilogy(x, e1d, marker=".", label="Error EFD")
ax.semilogy(x, c1d, marker="x", label="Error CFD")
ax.semilogy(x, v1d, marker="+", label="Error VFD")
plt.legend()
plt.title("Error dxf 1D (x)")
plt.show()

fig, ax = plt.subplots()
ax.semilogy(x, np.abs(V2XF-V1XF), marker=".", label="Error Pent")
ax.semilogy(x, np.abs(V3XF-V1XF), marker="+", label="Error LU")
ax.semilogy(x, np.abs(V4XF-V1XF), marker="x", label="Error D")
plt.legend()
plt.title("Error dxf 1D (x)")
plt.show()
"""
