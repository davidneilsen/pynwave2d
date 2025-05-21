import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nwave import Grid, Grid2D, CompactFirst2D, CompactSecond2D, ExplicitFirst44_2D, ExplicitSecond44_2D


def func(x, y):
    k1 = 3.0
    k2 = 1.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.cos(X - k2 * Y) + np.sin(k1 * X + Y)
    dxf = k1 * np.cos(k1 * X + Y) - np.sin(X - k2 * Y)
    dyf = np.cos(k1 * X + Y) + k2 * np.sin(X - k2 * Y)
    dxxf = -np.cos(X - k2 * Y) - k1**2 * np.sin(k1 * X + Y)
    dyyf = -(k2**2) * np.cos(X - k2 * Y) - np.sin(k1 * X + Y)
    return f, dxf, dyf, dxxf, dyyf


def func2(x, y):
    """Simpler test for testing 'symmetry' to make sure x and y give the same values"""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    freq = (1.0 / 2.0) * np.pi

    f = np.cos(freq * xx) * np.cos(freq * yy)

    # first order analytical derivative
    dxf = -freq * np.sin(freq * xx) * np.cos(freq * yy)
    dyf = -freq * np.cos(freq * xx) * np.sin(freq * yy)

    # second order analytical derivative
    dxxf = -freq * freq * np.cos(freq * xx) * np.cos(freq * yy)
    dyyf = -freq * freq * np.cos(freq * xx) * np.cos(freq * yy)

    return f, dxf, dyf, dxxf, dyyf


with open("params2D.json") as f:
    params = json.load(f)

g = Grid2D(params)
dx = g.dx[0]
dy = g.dx[1]
x = g.xi[0]
y = g.xi[1]

D1 = ExplicitFirst44_2D(dx, dy)
D2 = ExplicitSecond44_2D(dx, dy)

C1 = CompactFirst2D(x, y, "D1_SP4")
C2 = CompactSecond2D(x, y, "D2_JTT4")

f, dxf, dyf, dxxf, dyyf = func(x, y)

DXF = D1.grad_x(f)
DYF = D1.grad_y(f)
DXXF = D2.grad_xx(f)
DYYF = D2.grad_yy(f)

CXF = C1.grad_x(f)
CYF = C1.grad_y(f)
CXXF = C2.grad_xx(f)
CYYF = C2.grad_yy(f)

err_x = DXF - dxf
err_y = DYF - dyf
err_xx = DXXF - dxxf
err_yy = DYYF - dyyf

crr_x = CXF - dxf
crr_y = CYF - dyf
crr_xx = CXXF - dxxf
crr_yy = CYYF - dyyf

print("Error in x compact derivative:", np.linalg.norm(crr_x))
print("Error in y compact derivative:", np.linalg.norm(crr_y))
print("Error in xx compact derivative:", np.linalg.norm(crr_xx))
print("Error in yy compact derivative:", np.linalg.norm(crr_yy))

print("Error in x derivative:", np.linalg.norm(err_x))
print("Error in y derivative:", np.linalg.norm(err_y))
print("Error in xx derivative:", np.linalg.norm(err_xx))
print("Error in yy derivative:", np.linalg.norm(err_yy))

# 1D plots
ii = (params["Nx"]-1) // 2
c1d = np.abs(crr_xx[:,ii])
e1d = np.abs(err_xx[:,ii])
fig, ax = plt.subplots()
ax.semilogy(x,e1d,marker=".",label="FD")
ax.semilogy(x,c1d,marker=".",label="CFD")
plt.legend()
plt.title("Error dxxf 1D (x)")

c1d = np.abs(crr_x[:,ii])
e1d = np.abs(err_x[:,ii])
fig, ax = plt.subplots()
ax.semilogy(x,e1d,marker=".",label="FD")
ax.semilogy(x,c1d,marker=".",label="CFD")
plt.legend()
plt.title("Error dxf 1D (x)")


"""
# 2D Plots


f1d = np.abs(crr_xx[ii,:])
fig, ax = plt.subplots()
ax.semilogy(y,f1d,marker=".")
plt.title("crr_xx 1D (y)")

X, Y = np.meshgrid(x, y, indexing="ij")
fig, ax = plt.subplots()
CS = ax.contourf(X, Y, f, levels=10)
fig.colorbar(CS)
plt.title("f")

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, dxf, levels=10)
fig.colorbar(CS)
plt.title("dxf")

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, dyf, levels=10)
fig.colorbar(CS)
plt.title("dxf")


# Error plots
fig, ax = plt.subplots()
CS = ax.contourf(X, Y, err_x, levels=10)
fig.colorbar(CS)
plt.title("Error dxf")


fig, ax = plt.subplots()
CS = ax.contourf(X, Y, err_y, levels=10)
fig.colorbar(CS)
plt.title("Error dyf")

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, err_xx, levels=10)
fig.colorbar(CS)
plt.title("Error dxxf")


fig, ax = plt.subplots()
CS = ax.contourf(X, Y, err_yy, levels=10)
fig.colorbar(CS)
plt.title("Error dyyf")

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, crr_xx, levels=10)
fig.colorbar(CS)
plt.title("Error compact dxxf")


fig, ax = plt.subplots()
CS = ax.contourf(X, Y, crr_yy, levels=10)
fig.colorbar(CS)
plt.title("Error compact dyyf")
"""


plt.show()

