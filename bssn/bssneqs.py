import numpy as np
import sys
import os
from numba import njit

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import Equations, Grid, l2norm

DEBUG = False


class GBSSNSystem:
    """
    Class to define the system of equations for the generalized BSSN equations.

    Parameters
    ----------
    eqs : int
        System of equations to use. 0 for Eulerian, 1 for Lagrangian.
    lapse : int
        Lapse system to use. 0 for no advection terms, 1 to include advection terms.
    shift : int
        Shift system to use. 0 for no advection terms, 1 to include advection terms.
    """

    def __init__(self, eqs, lapse, shift):
        if eqs not in [0, 1]:
            raise ValueError("Invalid equations system. Must be 0 or 1.")
        if lapse not in [0, 1]:
            raise ValueError("Invalid lapse system. Must be 0 or 1.")
        if shift not in [0, 1]:
            raise ValueError("Invalid shift system. Must be 0 or 1.")

        self.eqs = eqs
        self.lapse = lapse
        self.shift = shift


class BSSN(Equations):
    """
    The BSSN equations in 1D.

    The equations are from the paper "BSSN in Spherical Symmetry" by J. David Brown (2008).
    """

    def __init__(self, g, M, eta, extended_domain, apply_bc=None, gbssn_system=GBSSNSystem(1, 1, 1)):
        """
        Initialize the BSSN equations.
        Parameters
        ----------
        g : Grid1D
            Grid object.
        M : float
            Mass of the black hole.
        eta : float
            Damping parameter.
        apply_bc : "FUNCTION" or None
            If "FUNCTION", apply the boundary conditions using the function in each post-RK step.
        gbssn_system : GBSSNSystem
            System of equations to use.
        """

        NU = 9
        super().__init__(NU, g, apply_bc)

        self.gbssn_system = gbssn_system
        self.eta = eta
        self.M = M
        self.extended_domain = extended_domain

        self.U_CHI = 0
        self.U_GRR = 1
        self.U_GTT = 2
        self.U_ARR = 3
        self.U_K = 4
        self.U_GT = 5
        self.U_ALPHA = 6
        self.U_SHIFT = 7
        self.U_GB = 8

        nx = g.shp[0]
        self.Nc = 3
        self.C = []
        for i in range(self.Nc):
            self.C.append(np.zeros(nx))

        self.C_HAM = 0
        self.MOM = 1
        self.GAMCON = 2
        self.u_names = ["chi", "grr", "gtt", "Arr", "K", "Gt", "alpha", "beta", "gB"]
        self.c_names = ["ham", "mom", "gamcom"]

    def initialize(self, g: Grid, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        if params["initial_data"] == "Puncture":
            self.initial_data_puncture(g, params)
        elif params["initial_data"] == "EddingtonFinkelstein":
            if self.gbssn_system.eqs == 0:
                raise ValueError("Eddington-Finkelstein initial data is not valid for Eulerian system.")
            self.initial_data_ef(g)
        else:
            raise ValueError("Invalid initial data. Must be 'puncture' or 'EddingtonFinkelstein'.")

    def initial_data_puncture(self, g: Grid, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        r = g.xi[0]
        if params["collapsed_lapse"] == True:
            self.u[self.U_ALPHA][:] = 1.0 / (1.0 + self.M / (2 * abs(r))) ** 2
        else:
            self.u[self.U_ALPHA].fill(1.0)

        self.u[self.U_CHI][:] = 1.0 / (1.0 + self.M / (2 * abs(r))) ** (4)
        self.u[self.U_GRR][:] = 1.0
        self.u[self.U_GTT][:] = r**2
        self.u[self.U_ARR][:] = 0.0
        self.u[self.U_K][:] = 0.0
        self.u[self.U_GT][:] = -2.0 / r
        self.u[self.U_SHIFT][:] = 0.0
        self.u[self.U_GB][:] = 0.0

    def initial_data_ef(self, g: Grid):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in Eddington-Finkelstein coordinates.
        This can be used to verify the BSSN equations for LAGRANGIAN
        systems.
        """
        r = np.abs(g.xi[0])
        H = np.zeros_like(r)
        m = self.M

        H[:] = 2.0*m / r
        self.u[self.U_ALPHA][:] = 1.0/np.sqrt(1.0 + H)
        self.u[self.U_CHI][:] = 1.0 / (1.0 + H)**(1.0/3.0)
        self.u[self.U_GRR][:] = (1.0 + H)**(2.0/3.0)
        self.u[self.U_GTT][:] = r**2/(1.0 + H)**(1.0/3.0)
        self.u[self.U_ARR][:] = -4.0*m*(1.0 + H)**(1.0/6.0)*(3.0*m + 2.0*r) / (3.0*r*r*(r + 2.0*m))
        self.u[self.U_K][:] = 2*m*(r + 3.0*m) / (r*r*(2.0*m + r)*np.sqrt(1.0 + H))
        self.u[self.U_GT][:] = -2.0*(8*m + 3*r)*(1.0 + H)**(1.0/3.0) / (3.0*(2.0*m + r)**2)
        self.u[self.U_SHIFT][:] = H / (1 + H)
        self.u[self.U_GB][:] = 0.0

    def rhs(self, dtu, u, g: Grid):
        v = self.gbssn_system.eqs
        lambda_lapse = self.gbssn_system.lapse
        lambda_shift = self.gbssn_system.shift

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

        if g.D1 is None or g.D2 is None:
            raise AttributeError(
                "Grid object 'g' must have non-None D1 and D2 attributes with 'grad' and 'grad2' methods."
            )

        d_alpha = g.D1.grad(alpha)
        d_beta_r = g.D1.grad(beta_r)
        d_chi = g.D1.grad(chi)
        d_g_rr = g.D1.grad(g_rr)
        d_g_tt = g.D1.grad(g_tt)
        d_K = g.D1.grad(K)
        d_Gamma_r = g.D1.grad(Gamma_r)

        USE_ADVECTION = True
        if USE_ADVECTION and g.D1.HAVE_ADVECTIVE_DERIV:
            # Advection terms
            dr = g.dx[0]
            ad_alpha = np.empty_like(alpha)
            ad_beta_r = np.empty_like(beta_r)
            ad_B_r = np.empty_like(B_r)
            ad_chi = np.empty_like(chi)
            ad_g_rr = np.empty_like(g_rr)
            ad_g_tt = np.empty_like(g_tt)
            ad_K = np.empty_like(K)
            ad_A_rr = np.empty_like(A_rr)
            ad_Gamma_r = np.empty_like(Gamma_r)

            g.D1.advec_grad(ad_B_r, B_r, beta_r, dr)
            g.D1.advec_grad(ad_alpha, alpha, beta_r, dr)
            g.D1.advec_grad(ad_beta_r, beta_r, beta_r, dr)
            g.D1.advec_grad(ad_g_rr, g_rr, beta_r, dr)
            g.D1.advec_grad(ad_g_tt, g_tt, beta_r, dr)
            g.D1.advec_grad(ad_A_rr, A_rr, beta_r, dr)
            g.D1.advec_grad(ad_K, K, beta_r, dr)
            g.D1.advec_grad(ad_Gamma_r, Gamma_r, beta_r, dr)
            g.D1.advec_grad(ad_chi, chi, beta_r, dr)
        else:
            ad_alpha = d_alpha
            ad_beta_r = d_beta_r
            ad_B_r = g.D1.grad(B_r)
            ad_chi = d_chi
            ad_g_rr = d_g_rr
            ad_g_tt = d_g_tt
            ad_K = d_K
            ad_A_rr = g.D1.grad(A_rr)
            ad_Gamma_r = d_Gamma_r

        d2_chi = g.D2.grad2(chi)
        d2_g_rr = g.D2.grad2(g_rr)
        d2_g_tt = g.D2.grad2(g_tt)
        d2_alpha = g.D2.grad2(alpha)
        d2_beta_r = g.D2.grad2(beta_r)

        BSSN.rhs_chi(
            chi_rhs,
            alpha,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_beta_r,
            ad_chi,
            d_g_rr,
            d_g_tt,
            v,
        )
        BSSN.rhs_g_rr(
            g_rr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, ad_g_rr, d_g_tt, d_beta_r, v
        )
        BSSN.rhs_g_tt(
            g_tt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, ad_g_tt, d_beta_r, v
        )
        BSSN.rhs_A_rr(
            A_rr_rhs,
            alpha,
            A_rr,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_alpha,
            d2_alpha,
            d_beta_r,
            d_chi,
            d2_chi,
            d_g_rr,
            d2_g_rr,
            d_g_tt,
            d2_g_tt,
            ad_A_rr,
            d_Gamma_r,
            v,
        )
        BSSN.rhs_K(
            K_rhs,
            alpha,
            A_rr,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            ad_K,
            d_alpha,
            d_chi,
            d_g_rr,
            d_g_tt,
            d2_alpha,
        )
        BSSN.rhs_Gamma_r(
            Gamma_r_rhs,
            alpha,
            A_rr,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_K,
            d_alpha,
            d_chi,
            d_g_rr,
            d2_g_rr,
            d_g_tt,
            d2_g_tt,
            d_beta_r,
            d2_beta_r,
            ad_Gamma_r,
            v,
        )
        BSSN.rhs_gauge(
            alpha_rhs,
            beta_r_rhs,
            B_r_rhs,
            alpha,
            beta_r,
            B_r,
            K,
            ad_alpha,
            ad_beta_r,
            ad_B_r,
            d_Gamma_r,
            Gamma_r_rhs,
            self.eta,
            lambda_lapse,
            lambda_shift,
        )

        # Apply outer boundary conditions at r = rmax
        ng = g.nghost
        alpha_rhs[-ng:] = 0.0
        beta_r_rhs[-ng:] = 0.0
        B_r_rhs[-ng:] = 0.0
        chi_rhs[-ng:] = 0.0
        g_rr_rhs[-ng:] = 0.0
        g_tt_rhs[-ng:] = 0.0
        A_rr_rhs[-ng:] = 0.0
        K_rhs[-ng:] = 0.0
        Gamma_r_rhs[-ng:] = 0.0
        if self.extended_domain:
            # Apply boundary conditions at r = -rmax
            alpha_rhs[:ng] = 0.0
            beta_r_rhs[:ng] = 0.0
            B_r_rhs[:ng] = 0.0
            chi_rhs[:ng] = 0.0
            g_rr_rhs[:ng] = 0.0
            g_tt_rhs[:ng] = 0.0
            A_rr_rhs[:ng] = 0.0
            K_rhs[:ng] = 0.0
            Gamma_r_rhs[:ng] = 0.0

        if DEBUG:
            print("v = ",v)
            print("dt_alpha = ", l2norm(alpha_rhs[10:]))
            print("dt_beta_r = ", l2norm(beta_r_rhs[10:]))
            print("dt_chi = ", l2norm(chi_rhs[10:]))
            print("dt_g_rr = ", l2norm(g_rr_rhs[10:]))
            print("dt_g_tt = ", l2norm(g_tt_rhs[10:]))
            print("dt_A_rr = ", l2norm(A_rr_rhs[10:]))
            print("dt_K = ", l2norm(K_rhs[10:]))
            print("dt_Gamma_r = ", l2norm(Gamma_r_rhs[10:]))
            sys.exit(0)

    @staticmethod
    @njit
    def rhs_A_rr(
        A_rr_rhs,
        alpha,
        A_rr,
        K,
        chi,
        g_rr,
        g_tt,
        beta_r,
        d_alpha,
        d2_alpha,
        d_beta_r,
        d_chi,
        d2_chi,
        d_g_rr,
        d2_g_rr,
        d_g_tt,
        d2_g_tt,
        ad_A_rr,
        d_Gamma_r,
        v,
    ):
        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt
        inv_6chi = 1.0 / (6.0 * chi)

        for i in range(A_rr_rhs.shape[0]):
            A_rr_rhs[i] = (
                -2.0 * alpha[i] * A_rr[i] ** 2 * inv_g_rr[i]
                + K[i] * alpha[i] * A_rr[i]
                - v * beta_r[i] * d_g_rr[i] * A_rr[i] * inv_g_rr[i] / 3.0
                - 2.0 * v * beta_r[i] * d_g_tt[i] * A_rr[i] * inv_g_tt[i] / 3.0
                - (2.0 / 3.0) * v * d_beta_r[i] * A_rr[i]
                + 2.0 * d_beta_r[i] * A_rr[i]
                + 2.0 * alpha[i] * chi[i] * d_g_rr[i] ** 2 * inv_g_rr2[i] / 3.0
                - alpha[i] * chi[i] * d_g_tt[i] ** 2 * inv_g_tt2[i] / 3.0
                - alpha[i] * d_chi[i] ** 2 * inv_6chi[i]
                - (2.0 / 3.0) * g_rr[i] * alpha[i] * chi[i] * inv_g_tt[i]
                + beta_r[i] * ad_A_rr[i]
                + (2.0 / 3.0) * g_rr[i] * alpha[i] * chi[i] * d_Gamma_r[i]
                - (alpha[i] * chi[i] * d_g_rr[i] * d_g_tt[i])
                * inv_g_rr[i]
                * inv_g_tt[i]
                / 2.0
                + chi[i] * d_g_rr[i] * d_alpha[i] * inv_g_rr[i] / 3.0
                + chi[i] * d_g_tt[i] * d_alpha[i] * inv_g_tt[i] / 3.0
                - (alpha[i] * d_g_rr[i] * d_chi[i]) * inv_g_rr[i] / 6.0
                - (alpha[i] * d_g_tt[i] * d_chi[i]) * inv_g_tt[i] / 6.0
                - (2.0 / 3.0) * d_chi[i] * d_alpha[i]
                - alpha[i] * chi[i] * d2_g_rr[i] * inv_g_rr[i] / 3.0
                + alpha[i] * chi[i] * d2_g_tt[i] * inv_g_tt[i] / 3.0
                - (2.0 / 3.0) * chi[i] * d2_alpha[i]
                + alpha[i] * d2_chi[i] / 3.0
            )

    @staticmethod
    @njit
    def rhs_Gamma_r(
        Gamma_r_rhs,
        alpha,
        A_rr,
        chi,
        g_rr,
        g_tt,
        beta_r,
        d_K,
        d_alpha,
        d_chi,
        d_g_rr,
        d2_g_rr,
        d_g_tt,
        d2_g_tt,
        d_beta_r,
        d2_beta_r,
        ad_Gamma_r,
        v,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt

        Gamma_r_rhs[:] = (
            -v * beta_r * d_g_tt**2 * inv_g_rr * inv_g_tt2
            + A_rr * alpha * d_g_tt * inv_g_rr2 * inv_g_tt
            - v * d_beta_r * d_g_tt * inv_g_rr * inv_g_tt / 3.0
            + d_beta_r * d_g_tt * inv_g_rr * inv_g_tt
            + beta_r * ad_Gamma_r
            + A_rr * alpha * d_g_rr * inv_g_rr2 * inv_g_rr
            - (4.0 / 3.0) * alpha * d_K * inv_g_rr
            - 2.0 * A_rr * d_alpha * inv_g_rr2
            + v * d_g_rr * d_beta_r * inv_g_rr2 / 2.0
            - d_g_rr * d_beta_r * inv_g_rr2 / 2.0
            - 3.0 * A_rr * alpha * d_chi * inv_g_rr2 / chi
            + v * beta_r * d2_g_rr * inv_g_rr2 / 6.0
            + v * beta_r * d2_g_tt * inv_g_rr * inv_g_tt / 3.0
            + v * d2_beta_r * inv_g_rr / 3.0
            + d2_beta_r * inv_g_rr
        )

    @staticmethod
    @njit
    def rhs_K(
        K_rhs,
        alpha,
        A_rr,
        K,
        chi,
        g_rr,
        g_tt,
        beta_r,
        ad_K,
        d_alpha,
        d_chi,
        d_g_rr,
        d_g_tt,
        d2_alpha,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        K_rhs[:] = (
            3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * ad_K
            + chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            - chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            + d_alpha * d_chi * inv_g_rr / 2
            - chi * d2_alpha * inv_g_rr
        )
        """
        print("chi       = ", l2norm(chi))
        print("alpha     = ", l2norm(alpha))
        print("d2_alpha  = ", l2norm(d2_alpha))
        print("K_rhs     = ", l2norm(K_rhs))
        """

    @staticmethod
    @njit
    def rhs_g_rr(
        grr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, ad_g_rr, d_g_tt, d_beta_r, v
    ):

        inv_g_tt = 1.0 / g_tt
        grr_rhs[:] = (
            -2 * A_rr * alpha
            - (1 / 3) * v * beta_r * ad_g_rr
            + beta_r * ad_g_rr
            - 2 * g_rr * v * beta_r * d_g_tt * inv_g_tt / 3
            + 2 * g_rr * d_beta_r
            - 2 / 3 * g_rr * v * d_beta_r
        )

    @staticmethod
    @njit
    def rhs_g_tt(
        gtt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, ad_g_tt, d_beta_r, v
    ):

        inv_g_rr = 1.0 / g_rr
        gtt_rhs[:] = (
            A_rr * g_tt * alpha * inv_g_rr
            - g_tt * v * beta_r * d_g_rr * inv_g_rr / 3
            - 2 / 3 * v * beta_r * ad_g_tt
            + beta_r * ad_g_tt
            - 2 / 3 * g_tt * v * d_beta_r
        )

    @staticmethod
    @njit
    def rhs_chi(
        chi_rhs, alpha, K, chi, g_rr, g_tt, beta_r, d_beta_r, ad_chi, d_g_rr, d_g_tt, v
    ):

        chi_rhs[:] = (
            2 * K[:] * alpha[:] * chi[:] / 3
            - v * beta_r[:] * d_g_rr[:] * chi[:] / (3 * g_rr[:])
            - 2 * v * beta_r[:] * d_g_tt[:] * chi[:] / (3 * g_tt[:])
            - 2 / 3 * v * d_beta_r[:] * chi[:]
            + beta_r[:] * ad_chi[:]
        )
        """
        print("v       = ", v)
        print("chi     = ", l2norm(chi))
        print("alpha   = ", l2norm(alpha))
        print("d_g_tt  = ", l2norm(d_g_tt))
        print("chi_rhs = ", l2norm(chi_rhs))
        """

    @staticmethod
    @njit
    def rhs_gauge(
        alpha_rhs,
        beta_r_rhs,
        B_r_rhs,
        alpha,
        beta_r,
        B_r,
        K,
        ad_alpha,
        ad_beta_r,
        ad_B_r,
        d_Gamma_r,
        dt_Gamma_r,
        eta,
        lambda_lapse,
        lambda_shift,
    ):

        alpha_rhs[:] = lambda_lapse * beta_r * ad_alpha - 2 * alpha * K
        beta_r_rhs[:] = (3.0 / 4.0) * B_r + lambda_shift * beta_r * ad_beta_r
        B_r_rhs[:] = (
            dt_Gamma_r
            + lambda_shift * beta_r * ad_B_r
            - lambda_shift * beta_r * d_Gamma_r
            - eta * B_r
        )

    def apply_bcs(self, u, g: Grid):
        """
        Regularity conditions at the origin.  (This routine called from RK4.)
        """
        """       
        Apply regularity conditions at the origin.  Brown uses guard
        cells to apply the regularity conditions to some variables.
        These conditions are used to set some variables or their
        derivatives to zero at the origin.

        gtt = u[self.U_GTT]
        b = u[self.U_SHIFT]
        gtt[0] = (-315*gtt[1] + 210*gtt[2] - 126*gtt[3] + 45*gtt[4] - 7*gtt[5])/63
        b[0] = (-315*b[1] + 210*b[2] - 126*b[3] + 45*b[4] - 7*b[5])/63
        # print("Applying regularity conditions at the origin")

        # Apply regularity conditions at the origin
        if self.gbssn_system.eqs == 0:
            # Lagrangian
            self.u[self.U_GTT][0] = extrapolate_func(self.u[self.U_GTT], order=5)
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)
        elif self.gbssn_system.eqs == 1:
            # Eulerian
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)
        """
        # print("Applying regularity conditions at the origin")
        ng = g.nghost

        if self.extended_domain == False:
            set_inner_regularity(self.u[self.U_CHI], ng, 1)
            set_inner_regularity(self.u[self.U_GRR], ng, 1)
            set_inner_regularity(self.u[self.U_GTT], ng, 1)
            set_inner_regularity(self.u[self.U_ARR], ng, 1)
            set_inner_regularity(self.u[self.U_K], ng, 1)
            set_inner_regularity(self.u[self.U_GT], ng, -1)
            set_inner_regularity(self.u[self.U_ALPHA], ng, 1)
            set_inner_regularity(self.u[self.U_SHIFT], ng, -1)
            set_inner_regularity(self.u[self.U_GB], ng, -1)

    def cal_constraints(self, u, g: Grid):
        """
        Calculate the constraints for the BSSN equations.
        """
        Ham = self.C[self.C_HAM]
        Mom = self.C[self.MOM]
        Gamcon = self.C[self.GAMCON]

        chi = u[self.U_CHI]
        g_rr = u[self.U_GRR]
        g_tt = u[self.U_GTT]
        A_rr = u[self.U_ARR]
        K = u[self.U_K]
        Gamma_r = u[self.U_GT]

        if g.D1 is None or g.D2 is None:
            raise AttributeError(
                "Grid object 'g' must have non-None D1 and D2 attributes with 'grad' and 'grad2' methods."
            )

        d_chi = g.D1.grad(chi)
        d_g_rr = g.D1.grad(g_rr)
        d_g_tt = g.D1.grad(g_tt)
        d_A_rr = g.D1.grad(A_rr)
        d_K = g.D1.grad(K)

        d2_chi = g.D2.grad2(chi)
        d2_g_tt = g.D2.grad2(g_tt)

        BSSN.get_constraints(
            Ham,
            Mom,
            Gamcon,
            chi,
            g_rr,
            g_tt,
            A_rr,
            K,
            Gamma_r,
            d_chi,
            d_g_rr,
            d_g_tt,
            d_A_rr,
            d_K,
            d2_chi,
            d2_g_tt,
        )

    @staticmethod
    @njit
    def get_constraints(
        Ham,
        Mom,
        Gamcon,
        chi,
        g_rr,
        g_tt,
        A_rr,
        K,
        Gamma_r,
        d_chi,
        d_g_rr,
        d_g_tt,
        d_A_rr,
        d_K,
        d2_chi,
        d2_g_tt,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        Ham[:] = (
            -(3.0 / 2.0) * A_rr**2 * inv_g_rr2
            + (2.0 / 3.0) * K**2
            - (5.0 / 2.0) * d_chi**2 * inv_g_rr / chi
            + 2.0 * d2_chi * inv_g_rr
            + 2.0 * chi * inv_g_tt
            - 2.0 * chi * d2_g_tt * inv_g_rr * inv_g_tt
            + 2.0 * d_chi * d_g_tt * inv_g_rr * inv_g_tt
            + chi * d_g_rr * d_g_tt * inv_g_rr2 * inv_g_tt
            - d_chi * d_g_rr * inv_g_rr2
            + 0.5 * chi * d_g_tt**2 * inv_g_rr * inv_g_tt * inv_g_tt
        )

        Mom[:] = (
            d_A_rr * inv_g_rr
            - (2.0 / 3.0) * d_K
            - (3.0 / 2.0) * A_rr * d_chi * inv_g_rr / chi
            + (3.0 / 2.0) * A_rr * d_g_tt * inv_g_rr * inv_g_tt
            - A_rr * d_g_rr * inv_g_rr2
        )

        Gamcon[:] = -0.5 * d_g_rr * inv_g_rr2 + Gamma_r + d_g_tt * inv_g_rr * inv_g_tt


def extrapolate_func(u, order=4):
    """
    To apply regularity at the origin. This function assumes that
    the grid is defined at points r = (i*h/2) for i = 0, 1, ..., N.

    The interpolating polynomial of order n is
        P_n(r) = a_0 + a_1*r + a_2*r^2 + ... + a_n*r^n

    The function at h/2 is set such that the extrapolated value at
    r = 0 is zero.
    """
    if order == 4:
        return (420 * u[1] - 378 * u[2] + 180 * u[3] - 35 * u[4]) / 315
    elif order == 5:
        return (1155 * u[1] - 1386 * u[2] + 990 * u[3] - 385 * u[4] + 63 * u[5]) / 693
    elif order == 6:
        return (
            6006 * u[1]
            - 9009 * u[2]
            + 8580 * u[3]
            - 5005 * u[4]
            + 1638 * u[5]
            - 231 * u[6]
        ) / 3003
    else:
        raise ValueError("Invalid order. Must be 4, 5 or 6.")


def extrapolate_deriv(u, order=4):
    """
    To apply regularity at the origin. This function assumes that
    the grid is defined at points r = (i*h/2) for i = 0, 1, ..., N.

    The interpolating polynomial of order n is
        P_n(r) = a_0 + a_1*r + a_2*r^2 + ... + a_n*r^n

    The function at h/2 is set such that the extrapolated value of
    dP_n/dr at r = 0 is zero.
    """
    if order == 4:
        return (229 * u[1] - 225 * u[2] + 111 * u[3] - 22 * u[4]) / 93
    elif order == 5:
        return (
            26765 * u[1] - 34890 * u[2] + 25770 * u[3] - 10205 * u[4] + 1689 * u[5]
        ) / 9129
    elif order == 6:
        return (
            36527 * u[1]
            - 59295 * u[2]
            + 58310 * u[3]
            - 34610 * u[4]
            + 11451 * u[5]
            - 1627 * u[6]
        ) / 10756
    else:
        raise ValueError("Invalid order. Must be 4, 5 or 6.")


def set_inner_regularity(u, ng, parity=1):
    """
    Apply inner boundary condition at the origin.
    """
    u[:ng] = parity * u[2 * ng - 1 : ng - 1 : -1]

