import numpy as np
from numba import njit
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nwave import *


class Maxwell2D(Equations):
    def __init__(self, NU, g, bctype):
        if bctype == "SOMMERFELD":
            apply_bc = BCType.RHS
        elif bctype == "WAVEGUIDE":
            apply_bc = BCType.FUNCTION
        else:
            raise ValueError(
                "Invalid boundary condition type. Use 'SOMMERFELD' or 'WAVEGUIDE'."
            )

        self.bound_cond = bctype
        self.U_EX = 0
        self.U_EY = 1
        self.U_HZ = 2
        self.u_names = ["Ex", "Ey", "Hz"]
        super().__init__(NU, g, apply_bc)

    def rhs(self, dtu, u, g):
        """
        Compute the right-hand side of the Maxwell equations for Transverse Electric (TE) mode in 2D.
        """
        dtEx = dtu[self.U_EX]
        dtEy = dtu[self.U_EY]
        dtHz = dtu[self.U_HZ]
        Ex = u[self.U_EX]
        Ey = u[self.U_EY]
        Hz = u[self.U_HZ]

        dyEx = g.D1.grad_y(Ex)
        dxEy = g.D1.grad_x(Ey)
        dxHz = g.D1.grad_x(Hz)
        dyHz = g.D1.grad_y(Hz)

        dtEx[:, :] = dyHz[:, :]
        dtEy[:, :] = -dxHz[:, :]
        dtHz[:, :] = dyEx[:, :] - dxEy[:, :]

        if self.apply_bc == BCType.RHS and self.bound_cond == "SOMMERFELD":
            # Sommerfeld boundary conditions
            # print("Applying Sommerfeld boundary conditions")
            x = g.xi[0]
            y = g.xi[1]
            dxEx = g.D1.grad_x(Ex)
            dyEy = g.D1.grad_y(Ey)
            bc_sommerfeld(dtEx, Ex, dxEx, dyEx, 1.0, 1, x, y)
            bc_sommerfeld(dtEy, Ey, dxEy, dyEy, 1.0, 1, x, y)
            bc_sommerfeld(dtHz, Hz, dxHz, dyHz, 1.0, 1, x, y)

    def initialize(self, g, params):
        if params["id_type"] == "waveguide":
            if np.abs(g.xi[0][0]) > 1.0e-8 or np.abs(g.xi[1][0]) > 1.0e-8:
                raise ValueError("Grid must start at (0, 0) for waveguide solution.")
            waveguide_solution(self.u, g, 0.0, params)
        elif params["id_type"] == "gaussian":
            self.gaussian_pulse(g, params)
        else:
            raise ValueError(
                "Invalid initial condition type. Use 'waveguide' or 'gaussian'."
            )

    def gaussian_pulse(self, g, params):
        x = g.xi[0]
        y = g.xi[1]
        x0, y0 = params["id_x0"], params["id_y0"]
        B0, sigma = params["id_B0"], params["id_sigma"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.u[self.U_EX][:, :] = 0.0
        self.u[self.U_EY][:, :] = 0.0
        self.u[self.U_HZ][:, :] = B0 * np.exp(
            -((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)
        )

    def apply_bcs(self, u, g):
        if self.apply_bc == BCType.FUNCTION and self.bound_cond == "WAVEGUIDE":
            bc_waveguide(u[0], u[1], u[2])
        else:
            raise NotImplementedError(
                f"Boundary condition '{self.bound_cond}' is not implemented."
            )


def waveguide_solution(u, g, t, params):
    # Initial data for a rectangular waveguide mode at the critical frequency.
    # This is a TE mode solution.
    x = g.xi[0]
    y = g.xi[1]
    B0 = params["id_B0"]
    mode_m = params["id_mode_m"]
    mode_n = params["id_mode_n"]
    a = x[-1] - x[0]
    b = y[-1] - y[0]
    omn = np.pi * np.sqrt((mode_m / a) ** 2 + (mode_n / b) ** 2)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u[0][:] = (
        -B0
        * mode_n
        * np.pi
        * np.cos((mode_m * np.pi * X) / a)
        * np.sin(omn * t)
        * np.sin((mode_n * np.pi * Y) / b)
        / (b * omn)
    )
    u[1][:] = (
        B0
        * mode_m
        * np.pi
        * np.cos((mode_n * np.pi * Y) / b)
        * np.sin(omn * t)
        * np.sin((mode_m * np.pi * X) / a)
        / (a * omn)
    )
    u[2][:] = (
        B0
        * np.cos((mode_m * np.pi * X) / a)
        * np.cos((mode_n * np.pi * Y) / b)
        * np.cos(omn * t)
    )


def cal_constraints(u, divE, uexact, uerr, g, time, params):
    """
    Calculate the constraints for the Maxwell equations.
    divE = 0 and divH = 0
    """
    Ex = u[0]
    Ey = u[1]

    divE[:] = g.D1.grad_x(Ex) + g.D1.grad_y(Ey)
    if params["id_type"] == "waveguide":
        waveguide_solution(uexact, g, time, params)
        for i in range(3):
            uerr[i][:] = uexact[i][:] - u[i][:]


def bc_waveguide_old(Ex, Ey, Hz):
    """
    Apply waveguide boundary conditions for a TE mode in a rectangular waveguide.
    These BCs require the computational domain to be x = [0, a] and y = [0, b].
    """

    Ex[:, 0] = 0.0
    Ey[:, 0] = (
        48.0 * Ey[:, 1] - 36.0 * Ey[:, 2] + 16.0 * Ey[:, 3] - 3.0 * Ey[:, 4]
    ) / 25.0
    Hz[:, 0] = (
        48.0 * Hz[:, 1] - 36.0 * Hz[:, 2] + 16.0 * Hz[:, 3] - 3.0 * Hz[:, 4]
    ) / 25.0

    Ex[:, -1] = 0.0
    Ey[:, -1] = (
        48.0 * Ey[:, -2] - 36.0 * Ey[:, -3] + 16.0 * Ey[:, -4] - 3.0 * Ey[:, -5]
    ) / 25.0
    Hz[:, -1] = (
        48.0 * Hz[:, -2] - 36.0 * Hz[:, -3] + 16.0 * Hz[:, -4] - 3.0 * Hz[:, -5]
    ) / 25.0

    Ey[0, :] = 0.0
    Ex[0, :] = (
        48.0 * Ex[1, :] - 36.0 * Ex[2, :] + 16.0 * Ex[3, :] - 3.0 * Ex[4, :]
    ) / 25.0
    Hz[0, :] = (
        48.0 * Hz[1, :] - 36.0 * Hz[2, :] + 16.0 * Hz[3, :] - 3.0 * Hz[4, :]
    ) / 25.0

    Ey[-1, :] = 0.0
    Ex[-1, :] = (
        48.0 * Ex[-2, :] - 36.0 * Ex[-3, :] + 16.0 * Ex[-4, :] - 3.0 * Ex[-5, :]
    ) / 25.0
    Hz[-1, :] = (
        48.0 * Hz[-2, :] - 36.0 * Hz[-3, :] + 16.0 * Hz[-4, :] - 3.0 * Hz[-5, :]
    ) / 25.0


@njit
def bc_waveguide(Ex, Ey, Hz):
    """
    Apply waveguide boundary conditions for a TE mode in a rectangular waveguide.
    These BCs require the computational domain to be x = [0, a] and y = [0, b].
    """
    # print("Applying waveguide boundary conditions")
    nx = Ex.shape[0]
    ny = Ex.shape[1]

    LTRACE = False
    if LTRACE:
        print(f"@@@ WG:1 |Ex|={l2norm(Ex)}, |Ey|={l2norm(Ey)}, |Hz|={l2norm(Hz)}")

    j = 0
    for i in range(nx):
        Ex[i, j] = 0.0
        Ey[i, j] = (
            48.0 * Ey[i, j + 1]
            - 36.0 * Ey[i, j + 2]
            + 16.0 * Ey[i, j + 3]
            - 3.0 * Ey[i, j + 4]
        ) / 25.0
        Hz[i, j] = (
            48.0 * Hz[i, j + 1]
            - 36.0 * Hz[i, j + 2]
            + 16.0 * Hz[i, j + 3]
            - 3.0 * Hz[i, j + 4]
        ) / 25.0

    j = ny - 1
    for i in range(nx):
        Ex[i, j] = 0.0
        Ey[i, j] = (
            48.0 * Ey[i, j - 1]
            - 36.0 * Ey[i, j - 2]
            + 16.0 * Ey[i, j - 3]
            - 3.0 * Ey[i, j - 4]
        ) / 25.0
        Hz[i, j] = (
            48.0 * Hz[i, j - 1]
            - 36.0 * Hz[i, j - 2]
            + 16.0 * Hz[i, j - 3]
            - 3.0 * Hz[i, j - 4]
        ) / 25.0

    i = 0
    for j in range(ny):
        Ey[i, j] = 0.0
        Ex[i, j] = (
            48.0 * Ex[i + 1, j]
            - 36.0 * Ex[i + 2, j]
            + 16.0 * Ex[i + 3, j]
            - 3.0 * Ex[i + 4, j]
        ) / 25.0
        Hz[i, j] = (
            48.0 * Hz[i + 1, j]
            - 36.0 * Hz[i + 2, j]
            + 16.0 * Hz[i + 3, j]
            - 3.0 * Hz[i + 4, j]
        ) / 25.0

    i = nx - 1
    for j in range(ny):
        Ey[i, j] = 0.0
        Ex[i, j] = (
            48.0 * Ex[i - 1, j]
            - 36.0 * Ex[i - 2, j]
            + 16.0 * Ex[i - 3, j]
            - 3.0 * Ex[i - 4, j]
        ) / 25.0
        Hz[i, j] = (
            48.0 * Hz[i - 1, j]
            - 36.0 * Hz[i - 2, j]
            + 16.0 * Hz[i - 3, j]
            - 3.0 * Hz[i - 4, j]
        ) / 25.0

    if LTRACE:
        print(f"@@@ WG:2 |Ex|={l2norm(Ex)}, |Ey|={l2norm(Ey)}, |Hz|={l2norm(Hz)}")


@njit
def bc_sommerfeld(dtf, f, dxf, dyf, falloff, ngz, x, y):
    Nx = len(x)
    Ny = len(y)
    for j in range(Ny):
        for i in range(ngz):
            # xmin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
        for i in range(Nx - ngz, Nx):
            # xmax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )

    for i in range(Nx):
        for j in range(ngz):
            # ymin boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
        for j in range(Ny - ngz, Ny):
            # ymax boundary
            inv_r = 1.0 / np.sqrt(x[i] ** 2 + y[j] ** 2)
            dtf[i, j] = (
                -(x[i] * dxf[i, j] + y[j] * dyf[i, j] + falloff * f[i, j]) * inv_r
            )
