import numpy as np
import tomllib
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import *

DEBUG = False

def func1(x):
    k1 = 11.0
    k2 = 7.0
    f = np.cos(k2 * x) + np.sin(k1 * x)
    dxf = k1 * np.cos(k1 * x) - k2 * np.sin(k2 * x)
    dxxf = -k2**2 * np.cos(k2 * x) - k1**2 * np.sin(k1 * x)
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

nx0 = 14
level = 2
nx = int((2**level)*nx0 + 1)
params = { "Nx": nx, "Xmin": -1.0, "Xmax": 3.0 }

g = Grid1D(params)
dx = g.dx[0]
x = g.xi[0]

D1 = ExplicitFirst642_1D(dx)
D2 = ExplicitSecond642_1D(dx)

dnames = [ 
    "ME44",
    "ME642",
    "JT4",
#    "JT6",
    "JP6",
#    "DSQ6A",
#    "DSQ6B",
    "BYUP4_Op18",
    "BYUC6_Op10",
]

dnames2 = [
    "ME44",
    "ME642",
    "JT4",
    "JP6",
#    "DSQ6A",
#    "DSQ6B"
]


f, dxf, dxxf = func1(x)

fig, ax = plt.subplots()
ax.plot(x, f, marker=".", label="f")
#ax.plot(x, dxf, marker=".", label="dxf")
#ax.plot(x, dxxf, marker=".", label="dxxf")
plt.legend()
plt.title("f")
plt.savefig("func.pdf")
plt.show()

#--------------------------------
#  First Derivatives
#--------------------------------
fig, ax = plt.subplots()

#CD = NCompactDerivative.deriv(x, DerivType.D1_KP4, CFDSolve.SCIPY)
#CDxf = CD.grad(f)
#err = np.abs(CDxf - dxf)
#ax.semilogy(x, err, marker=".", label="KP4")

for dn in dnames:
    dd = d1_type_map[dn]
    print(f"...constructing {dn}")
    CD = NCompactDerivative.deriv(x, dd, CFDSolve.SCIPY)
    CDxf = CD.grad(f)
    err = np.abs(CDxf - dxf)
    ax.semilogy(x, err, marker=".", label=dn)

#ax.axvline(-1.0, ls='--', c='0.6')
#ax.axvline(-0.5, ls='-', c='0.8')
#ax.axvline(0.5, ls='-', c='0.8')
#ax.axvline(1.0, ls='--', c='0.6')
plt.legend(ncol=2)
plt.title("Error dxf 1D (x)")
plt.savefig("d1_err.pdf")
plt.show()

#--------------------------------
#  Second Derivatives
#--------------------------------
fig, ax = plt.subplots()

#CD = NCompactDerivative.deriv(x, DerivType.D1_KP4, CFDSolve.SCIPY)
#CDxf = CD.grad(f)
#CDxxf = CD.grad(CDxf)
#err = np.abs(CDxxf - dxxf)
#ax.semilogy(x, err, marker=".", label="D1_KP4^2")

for dn in dnames2:
    dd = d2_type_map[dn]
    print(f"...constructing {dn}")
    CD = NCompactDerivative.deriv(x, dd, CFDSolve.SCIPY)
    CDxf = CD.grad(f)
    err = np.abs(CDxf - dxxf)
    ax.semilogy(x, err, marker=".", label=dn)

#ax.axvline(-1.0, ls='--', c='0.6')
#ax.axvline(-0.5, ls='-', c='0.8')
#ax.axvline(0.5, ls='-', c='0.8')
#ax.axvline(1.0, ls='--', c='0.6')
plt.legend(ncol=2)
plt.title("Error dxxf 1D (x)")
plt.savefig("d2_err.pdf")
plt.show()

"""
#--------------------------------
#  Excision Derivatives
#--------------------------------
fig, ax = plt.subplots()

x_bh = 0.0
dx_bh = 0.15

for dd, dn in zip(derivs, dnames):
    print(f"...constructing {dn}")
    CD = NCompactDerivative.bh_deriv(x, dd, DerivType.D1_ME44, CFDSolve.SCIPY, x_bh, dx_bh)
    CDxf = CD.grad(f)
    err = np.abs(CDxf - dxf)
    ax.semilogy(x, err, marker=".", label=dn)

ax.axvline(-1.0, ls='--', c='0.6')
ax.axvline(-0.5, ls='-', c='0.8')
ax.axvline(0.5, ls='-', c='0.8')
ax.axvline(1.0, ls='--', c='0.6')

plt.legend()
plt.title("Error dxf with excision 1D (x)")
plt.show()
"""




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

