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


with open("params1D.toml", "rb") as f:
    params = tomllib.load(f)

g = Grid1D(params)
dx = g.dx[0]
x = g.xi[0]

D1 = ExplicitFirst44_1D(dx)
D2 = ExplicitSecond44_1D(dx)

C1 = CompactFirst1D(x, "D1_SP4")
C2 = CompactSecond1D(x, "D2_JTT4")

f, dxf, dxxf = func(x)

DXF = D1.grad(f)
DXXF = D2.grad2(f)

CXF = C1.grad(f)
CXXF = C2.grad2(f)

ab = C1.dxf.get_Abanded()
q = C1.dxf.get_B()
sys = LinearSolveLU(ab)
rhs = np.matmul(q, f)
cxf = sys.solve(rhs) / dx
e3 = np.abs(cxf - CXF)

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

c1d = np.abs(crr_x)
e1d = np.abs(err_x)
fig, ax = plt.subplots()
ax.semilogy(x,e1d,marker=".",label="FD")
ax.semilogy(x,c1d,marker=".",label="CFD")
plt.legend()
plt.title("Error dxf 1D (x)")

fig, ax = plt.subplots()
ax.semilogy(x,e3,marker=".",label="Solvers")
plt.legend()
plt.title("Error in LU solved system")





plt.show()

