import numpy as np
import sys
import os
from numba import njit

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nwave import Equations, Grid1D

class BSSN(Equations):
    """
    The BSSN equations in 1D.  

    The equations are from the paper "BSSN in Spherical Symmetry" by J. David Brown (2008).
    """
    def __init__(self, NU, g, M, eta, apply_bc=None):
        super().__init__(NU, g, apply_bc)
        self.eta = eta
        self.M = M
        self.v = 0

        self.U_CHI = 0
        self.U_GRR = 1
        self.U_GTT = 2
        self.U_ARR = 3
        self.U_K   = 4
        self.U_GT  = 5
        self.U_ALPHA  = 6
        self.U_SHIFT  = 7
        self.U_GB     = 8

        nx = g.shp[0]
        self.Nc = 3
        self.C = []
        for i in range(self.Nc):
            self.C.append(np.zeros(nx))
        
        self.C_HAM = 0
        self.MOM = 1
        self.GTR = 2

    def rhs(self, dtu, u, g : Grid1D):
        v = self.v

        chi = u[self.U_CHI]
        g_rr = u[self.U_GRR]
        g_tt = u[self.U_GTT]
        A_rr = u[self.U_ARR]
        K = u[self.U_K]
        Gamma_r = u[self.U_GT]
        alpha = u[self.U_ALPHA]
        beta_r = u[self.U_SHIFT]
        B_r = u[self.U_GB]

        chi_rhs = dtu[self.U_CHI]
        g_rr_rhs = dtu[self.U_GRR]
        g_tt_rhs = dtu[self.U_GTT]
        A_rr_rhs = dtu[self.U_ARR]
        K_rhs = dtu[self.U_K]
        Gamma_r_rhs = dtu[self.U_GT]
        alpha_rhs = dtu[self.U_ALPHA]
        beta_r_rhs = dtu[self.U_SHIFT]
        B_r_rhs = dtu[self.U_GB]

        d_chi = g.D1.grad_x(chi)
        d_g_rr = g.D1.grad_x(g_rr)
        d_g_tt = g.D1.grad_x(g_tt)
        d_A_rr = g.D1.grad_x(A_rr)
        d_K = g.D1.grad_x(K)
        d_Gamma_r = g.D1.grad_x(Gamma_r)
        d_alpha = g.D1.grad_x(alpha)
        d_beta_r = g.D1.grad_x(beta_r)
        d_B_r = g.D1.grad_x(B_r)

        d2_chi = g.D2.grad_xx(chi)
        d2_g_rr = g.D2.grad_xx(g_rr)
        d2_g_tt = g.D2.grad_xx(g_tt)
        d2_alpha = g.D2.grad_xx(alpha)
        d2_beta_r = g.D2.grad_xx(beta_r)

        BSSN.rhs_chi(chi_rhs, alpha, K, chi, g_rr, g_tt, beta_r, d_beta_r,
                d_chi, d_g_rr, d_g_tt, v)
        BSSN.rhs_g_rr(g_rr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v)
        BSSN.rhs_g_tt(g_tt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v)
        BSSN.rhs_A_rr(A_rr_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
             d_alpha, d2_alpha, d_beta_r, d_chi, d2_chi, d_g_rr, d2_g_rr,
             d_g_tt, d2_g_tt, d_A_rr, d_Gamma_r, v)
        BSSN.rhs_K(K_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
          d_K, d_alpha, d_chi, d_g_rr, d_g_tt, d2_alpha)
        BSSN.rhs_Gamma_r(Gamma_r_rhs, alpha, A_rr, chi, 
                g_rr, g_tt, beta_r, 
                d_K, d_alpha, d_chi,
                d_g_rr, d2_g_rr, d_g_tt, d2_g_tt,
                d_beta_r, d2_beta_r, d_Gamma_r, v)
        BSSN.rhs_gauge(alpha_rhs, beta_r_rhs, B_r_rhs, alpha, beta_r, 
                B_r, K, d_alpha, d_beta_r, d_B_r, 
                d_Gamma_r, Gamma_r_rhs, self.eta)

    @njit
    def rhs_A_rr(A_rr_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
             d_alpha, d2_alpha, d_beta_r, d_chi, d2_chi, d_g_rr, d2_g_rr,
             d_g_tt, d2_g_tt, d_A_rr, d_Gamma_r, v):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt
        inv_6chi = 1.0 / (6.0 * chi)

        A_rr_rhs = (-2.0 * alpha * A_rr**2 * inv_g_rr
               + K * alpha * A_rr
               - v * beta_r * d_g_rr * A_rr * inv_g_rr / 3.0
               - 2.0 * v * beta_r * d_g_tt * A_rr * inv_g_tt / 3.0
               - (2.0 / 3.0) * v * d_beta_r * A_rr
               + 2.0 * d_beta_r * A_rr
               + 2.0 * alpha * chi * d_g_rr**2 * inv_g_rr2 / 3.0
               - alpha * chi * d_g_tt**2 * inv_g_tt2 / 3.0
               - alpha * d_chi**2 * inv_6chi
               - (2.0 / 3.0) * g_rr * alpha * chi * inv_g_tt
               + beta_r * d_A_rr
               + (2.0 / 3.0) * g_rr * alpha * chi * d_Gamma_r
               - (alpha * chi * d_g_rr * d_g_tt) * inv_g_rr * inv_g_tt / 2.0
               + chi * d_g_rr * d_alpha * inv_g_rr / 3.0
               + chi * d_g_tt * d_alpha * inv_g_tt / 3.0
               - (alpha * d_g_rr * d_chi) * inv_g_rr / 6.0
               - (alpha * d_g_tt * d_chi) * inv_g_tt / 6.0
               - (2.0 / 3.0) * d_chi * d_alpha
               - alpha * chi * d2_g_rr * inv_g_rr / 3.0
               + alpha * chi * d2_g_tt * inv_g_tt / 3.0
               - (2.0 / 3.0) * d2_alpha
               + alpha * d2_chi / 3.0)


    @njit
    def rhs_Gamma_r(Gamma_r_rhs, alpha, A_rr, chi, 
                g_rr, g_tt, beta_r, 
                d_K, d_alpha, d_chi,
                d_g_rr, d2_g_rr, d_g_tt, d2_g_tt,
                d_beta_r, d2_beta_r, d_Gamma_r, v):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr *  inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt

        Gamma_r_rhs = (-v * beta_r * d_g_tt**2 * inv_g_rr * inv_g_tt2
               + A_rr * alpha * d_g_tt * inv_g_rr2 * inv_g_tt
               - v * d_beta_r * d_g_tt * inv_g_rr * inv_g_tt / 3.0
               + d_beta_r * d_g_tt * inv_g_rr * inv_g_tt
               + beta_r * d_Gamma_r
               + A_rr * alpha * d_g_rr * inv_g_rr2 * inv_g_rr
               - (4.0 / 3.0) * alpha * d_K * inv_g_rr
               - 2.0 * A_rr * d_alpha * inv_g_rr2
               + v * d_g_rr * d_beta_r * inv_g_rr2 / 2.0
               - d_g_rr * d_beta_r * inv_g_rr2 / 2.0
               - 3.0 * A_rr * alpha * d_chi * inv_g_rr2 / chi
               + v * beta_r * d2_g_rr * inv_g_rr2 / 6.0
               + v * beta_r * d2_g_tt * inv_g_rr * inv_g_tt / 3.0
               + v * d2_beta_r * inv_g_rr / 3.0
               + d2_beta_r * inv_g_rr)

    @njit
    def rhs_K(K_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
          d_K, d_alpha, d_chi, d_g_rr, d_g_tt, d2_alpha):

        g_rr2 = g_rr * g_rr
        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = 1.0 / g_rr2

        K_rhs = (3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * d_K
            + chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            - chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            + d_alpha * d_chi * inv_g_rr / 2
            - chi * d2_alpha * inv_g_rr)


    @njit
    def rhs_g_rr(grr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v):

        inv_g_tt = 1.0 / g_tt
        grr_rhs = (-2 * A_rr * alpha
            - (1 / 3) * v * beta_r * d_g_rr
            + beta_r * d_g_rr
            - 2 * g_rr * v * beta_r * d_g_tt * inv_g_tt / 3
            + 2 * g_rr * d_beta_r
            - 2 / 3 * g_rr * v * d_beta_r)

    @njit
    def rhs_g_tt(gtt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v):

        inv_g_rr = 1.0 / g_rr
        gtt_rhs = (A_rr * g_tt * alpha * inv_g_rr
            - g_tt * v * beta_r * d_g_rr * inv_g_rr / 3
            - 2 / 3 * v * beta_r * d_g_tt
            + beta_r * d_g_tt
            - 2 / 3 * g_tt * v * d_beta_r)

    @njit
    def rhs_chi(chi_rhs, alpha, K, chi, g_rr, g_tt, beta_r, d_beta_r,
                d_chi, d_g_rr, d_g_tt, v):

        chi_rhs = (2 * K * alpha * chi / 3
            - v * beta_r * d_g_rr * chi / (3 * g_rr)
            - 2 * v * beta_r * d_g_tt * chi / (3 * g_tt)
            - 2 / 3 * v * d_beta_r * chi + beta_r * d_chi)

    @njit
    def rhs_gauge(alpha_rhs, beta_r_rhs, B_r_rhs, alpha, beta_r, 
            B_r, K, d_alpha, d_beta_r, d_B_r,
            d_Gamma_r, dt_Gamma_r, eta):
        alpha_rhs = beta_r * d_alpha - 2 * alpha * K
        beta_r_rhs = (3.0/4.0) * B_r + beta_r * d_beta_r
        B_r_rhs = dt_Gamma_r + beta_r * d_B_r - beta_r * d_Gamma_r - eta * B_r

    def initialize(self, g: Grid1D, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        r = g.xi[0]
        self.u[self.U_CHI][:] = (1.0 + self.M /(2*r))**(-4)
        self.u[self.U_GRR][:] = 1.0
        self.u[self.U_GTT][:] = r**2
        self.u[self.U_ARR][:] = 0.0
        self.u[self.U_K][:] = 0.0
        self.u[self.U_GT][:] = 0.0
        self.u[self.U_ALPHA][:] = 1.0
        self.u[self.U_SHIFT][:] = 0.0
        self.u[self.U_GB][:] = 0.0

    def apply_bcs(self, u, g: Grid1D):
        """
        Regularity conditions at the origin.  (This routine called from RK4.)
        """
        gtt = u[self.U_GTT]
        b = u[self.U_SHIFT]
        gtt[0] = (-315*gtt[1] + 210*gtt[2] - 126*gtt[3] + 45*gtt[4] - 7*gtt[5])/63
        b[0] = (-315*b[1] + 210*b[2] - 126*b[3] + 45*b[4] - 7*b[5])/63
