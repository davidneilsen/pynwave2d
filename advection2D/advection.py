import numpy as np
from numba import njit
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nwave import Equations, Grid2D


class Advection(Equations):
    def __init__(self, NU, g: Grid2D, apply_bc=None):
        super().__init__(NU, g, apply_bc)
        self.U_PHI = 0
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def rhs(self, dtu, u, g: Grid2D):
        dtphi = dtu[0]
        phi = u[0]
        X, Y = np.meshgrid(g.xi[0], g.xi[1], indexing="ij")

        dxphi = g.D1.grad_x(phi)
        dyphi = g.D1.grad_y(phi)
        dxxphi = g.D2.grad_xx(phi)
        dyyphi = g.D2.grad_yy(phi)
        dtphi[:] = (
            Y[:] * dxphi[:] - X[:] * dyphi[:] + self.alpha * (dxxphi[:] + dyyphi[:])
        )

        # BCs
        dtphi[:, -1] = 0.0
        dtphi[0, :] = 0.0
        dtphi[:, 0] = 0.0
        dtphi[-1, :] = 0.0

    def initialize(self, g: Grid2D, params):
        x = g.xi[0]
        y = g.xi[1]
        x0, y0 = params["id_x0"], params["id_y0"]
        amp, omega = params["id_amp"], params["id_omega"]
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.u[0][:, :] = np.exp(-omega * ((X - x0) ** 2 + (Y - y0) ** 2))

    def apply_bcs(self, u, g: Grid2D):
        print("no bcs")
