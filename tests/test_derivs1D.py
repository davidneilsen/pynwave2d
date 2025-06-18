import numpy as np
import tomllib
import matplotlib.pyplot as plt
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nwave import *


def func(x):
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



nx0 = 14
level = 4
nx = int((2**level)*nx0 + 1)
params = { "Nx": nx, "Xmin": -4.0, "Xmax": 4.0 }

g = Grid1D(params)
dx = g.dx[0]
x = g.xi[0]

D1 = ExplicitFirst44_1D(dx)
D2 = ExplicitSecond44_1D(dx)

C1 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.SCIPY)
C2 = NCompactDerivative.deriv(x, DerivType.D2_JP6, CFDSolve.SCIPY)

C3 = NCompactDerivative.deriv(x, DerivType.D1_JP6, CFDSolve.LUSOLVE)

f, dxf, dxxf = func(x)

DXF = D1.grad(f)
DXXF = D2.grad2(f)

CXF = C1.grad(f)
CXXF = C2.grad(f)
C3XF = C3.grad(f)
e3 = np.abs(C3XF - CXF)

err_x = DXF - dxf
err_xx = DXXF - dxxf

crr_x = CXF - dxf
crr_xx = CXXF - dxxf

print("Error in x compact derivative:", np.linalg.norm(crr_x))
print("Error in xx compact derivative:", np.linalg.norm(crr_xx))

print("Error in x derivative:", np.linalg.norm(err_x))
print("Error in xx derivative:", np.linalg.norm(err_xx))

# 1D plots
c1d = np.abs(crr_xx)
e1d = np.abs(err_xx)
fig, ax = plt.subplots()
ax.semilogy(x,e1d,marker=".",label="FD")
ax.semilogy(x,c1d,marker=".",label="CFD")
plt.legend()
plt.title("Error dxxf 1D (x)")
plt.show()

c1d = np.abs(crr_x)
e1d = np.abs(err_x)
fig, ax = plt.subplots()
ax.semilogy(x,e1d,marker=".",label="FD")
ax.semilogy(x,c1d,marker=".",label="CFD")
plt.legend()
plt.title("Error dxf 1D (x)")
plt.show()

fig, ax = plt.subplots()
#ax.plot(x,DXF,marker=".",label="Explicit")
#ax.plot(x,CXF,marker=".",label="CX")
#ax.plot(x,C3XF,marker=".",label="C3X")
ax.semilogy(x,e3,marker=".",label="CFD Solve Methods")
plt.legend()
plt.title("Error in LU solved system")
plt.show()

