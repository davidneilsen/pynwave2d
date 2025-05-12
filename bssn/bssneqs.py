import numpy as np
import sys
import os
from numba import njit

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nwave import Equations, Grid1D, l2norm

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
        if (eqs not in [0, 1]):
            raise ValueError("Invalid equations system. Must be 0 or 1.")
        if (lapse not in [0, 1]):
            raise ValueError("Invalid lapse system. Must be 0 or 1.")
        if (shift not in [0, 1]):
            raise ValueError("Invalid shift system. Must be 0 or 1.")

        self.eqs = eqs
        self.lapse = lapse
        self.shift = shift

class BSSN(Equations):
    """
    The BSSN equations in 1D.  

    The equations are from the paper "BSSN in Spherical Symmetry" by J. David Brown (2008).
    """

    def __init__(self, g, M, eta, apply_bc=None, gbssn_system=GBSSNSystem(1, 1, 1)):
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
        apply_bc : function
            Function to apply boundary conditions.
        gbssn_system : GBSSNSystem
            System of equations to use.
        """

        NU = 9
        super().__init__(NU, g, apply_bc)

        self.gbssn_system = gbssn_system
        self.eta = eta
        self.M = M

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
        self.GAMCON = 2

    def rhs(self, dtu, u, g : Grid1D):
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
                d_Gamma_r, Gamma_r_rhs, self.eta, lambda_lapse, lambda_shift)
        
        # Apply outer boundary conditions
        alpha_rhs[-2:] = 0.0
        beta_r_rhs[-2:] = 0.0
        B_r_rhs[-2:] = 0.0
        chi_rhs[-2:] = 0.0
        g_rr_rhs[-2:] = 0.0
        g_tt_rhs[-2:] = 0.0
        A_rr_rhs[-2:] = 0.0
        K_rhs[-2:] = 0.0
        Gamma_r_rhs[-2:] = 0.0

        if DEBUG:
            print("d_alpha = ", l2norm(alpha_rhs))
            print("d_beta_r = ", l2norm(beta_r_rhs))
            print("d_chi = ", l2norm(chi_rhs))
            print("d_g_rr = ", l2norm(g_rr_rhs))
            print("d_g_tt = ", l2norm(g_tt_rhs))
            print("d_A_rr = ", l2norm(A_rr_rhs))
            print("d_K = ", l2norm(K_rhs))
            print("d_Gamma_r = ", l2norm(Gamma_r_rhs))

    @njit
    def rhs_A_rr(A_rr_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
             d_alpha, d2_alpha, d_beta_r, d_chi, d2_chi, d_g_rr, d2_g_rr,
             d_g_tt, d2_g_tt, d_A_rr, d_Gamma_r, v):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt
        inv_6chi = 1.0 / (6.0 * chi)

        A_rr_rhs[:] = (-2.0 * alpha * A_rr**2 * inv_g_rr
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

        Gamma_r_rhs[:] = (-v * beta_r * d_g_tt**2 * inv_g_rr * inv_g_tt2
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

    # FIXME njit
    def rhs_K(K_rhs, alpha, A_rr, K, chi, g_rr, g_tt, beta_r,
          d_K, d_alpha, d_chi, d_g_rr, d_g_tt, d2_alpha):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        K_rhs[:] = (3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * d_K
            + chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            - chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            + d_alpha * d_chi * inv_g_rr / 2
            - chi * d2_alpha * inv_g_rr)
        """
        print("chi       = ", l2norm(chi))
        print("alpha     = ", l2norm(alpha))
        print("d2_alpha  = ", l2norm(d2_alpha))
        print("K_rhs     = ", l2norm(K_rhs))
        """

    @njit
    def rhs_g_rr(grr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v):

        inv_g_tt = 1.0 / g_tt
        grr_rhs[:] = (-2 * A_rr * alpha
            - (1 / 3) * v * beta_r * d_g_rr
            + beta_r * d_g_rr
            - 2 * g_rr * v * beta_r * d_g_tt * inv_g_tt / 3
            + 2 * g_rr * d_beta_r
            - 2 / 3 * g_rr * v * d_beta_r)

    @njit
    def rhs_g_tt(gtt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, d_g_tt, d_beta_r, v):

        inv_g_rr = 1.0 / g_rr
        gtt_rhs[:] = (A_rr * g_tt * alpha * inv_g_rr
            - g_tt * v * beta_r * d_g_rr * inv_g_rr / 3
            - 2 / 3 * v * beta_r * d_g_tt
            + beta_r * d_g_tt
            - 2 / 3 * g_tt * v * d_beta_r)

    @njit
    def rhs_chi(chi_rhs, alpha, K, chi, g_rr, g_tt, beta_r, d_beta_r,
                d_chi, d_g_rr, d_g_tt, v):

        chi_rhs[:] = (2 * K * alpha * chi / 3
            - v * beta_r * d_g_rr * chi / (3 * g_rr)
            - 2 * v * beta_r * d_g_tt * chi / (3 * g_tt)
            - 2 / 3 * v * d_beta_r * chi + beta_r * d_chi)
        """
        print("v       = ", v)
        print("chi     = ", l2norm(chi))
        print("alpha   = ", l2norm(alpha))
        print("d_g_tt  = ", l2norm(d_g_tt))
        print("chi_rhs = ", l2norm(chi_rhs))
        """

    @njit
    def rhs_gauge(alpha_rhs, beta_r_rhs, B_r_rhs, alpha, beta_r, 
            B_r, K, d_alpha, d_beta_r, d_B_r,
            d_Gamma_r, dt_Gamma_r, eta, lambda_lapse, lambda_shift):

        alpha_rhs[:] = lambda_lapse * beta_r * d_alpha - 2 * alpha * K
        beta_r_rhs[:] = (3.0/4.0) * B_r + lambda_shift * beta_r * d_beta_r
        B_r_rhs[:] = dt_Gamma_r + lambda_shift * beta_r * d_B_r - lambda_shift * beta_r * d_Gamma_r - eta * B_r

    def initialize(self, g: Grid1D, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        r = g.xi[0]
        self.u[self.U_CHI][:] = 1.0/(1.0 + self.M /(2 * r))**(4)
        self.u[self.U_GRR][:] = 1.0
        self.u[self.U_GTT][:] = r**2
        self.u[self.U_ARR][:] = 0.0
        self.u[self.U_K][:] = 0.0
        self.u[self.U_GT][:] = -2.0 / r
        self.u[self.U_ALPHA][:] = (1.0 - self.M / (2*r)) / (1.0 + self.M / (2*r))
        self.u[self.U_SHIFT][:] = 0.0
        self.u[self.U_GB][:] = 0.0

    def apply_bcs(self, u, g: Grid1D):
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
        """
        # Apply regularity conditions at the origin
        if self.gbssn_system.eqs == 0:
            # Lagrangian
            self.u[self.U_GTT][0] = extrapolate_func(self.u[self.U_GTT], order=5)
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)
        elif self.gbssn_system.eqs == 1:
            # Eulerian
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)

    def cal_constraints(self, u, g):
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

        d_chi = g.D1.grad_x(chi)
        d_g_rr = g.D1.grad_x(g_rr)
        d_g_tt = g.D1.grad_x(g_tt)
        d_A_rr = g.D1.grad_x(A_rr)
        d_K = g.D1.grad_x(K)

        d2_chi = g.D2.grad_xx(chi)
        d2_g_tt = g.D2.grad_xx(g_tt)
 
        BSSN.get_constraints(Ham, Mom, Gamcon, chi, g_rr, g_tt, A_rr, K, Gamma_r,
            d_chi, d_g_rr, d_g_tt, d_A_rr, d_K, d2_chi, d2_g_tt)

    @njit
    def get_constraints(Ham, Mom, Gamcon, chi, g_rr, g_tt, A_rr, K, Gamma_r, 
            d_chi, d_g_rr, d_g_tt, d_A_rr, d_K, d2_chi, d2_g_tt):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        Ham[:] = ( -(3.0 / 2.0) * A_rr**2 * inv_g_rr2 
            + (2.0 / 3.0) * K**2 
            - (5.0 / 2.0) * d_chi**2 * inv_g_rr / chi 
            + 2.0 * d2_chi * inv_g_rr 
            + 2.0 * chi * inv_g_tt 
            - 2.0 * chi * d2_g_tt * inv_g_rr * inv_g_tt 
            + 2.0 * d_chi * d_g_tt * inv_g_rr * inv_g_tt 
            + chi * d_g_rr * d_g_tt * inv_g_rr2 * inv_g_tt 
            - d_chi * d_g_rr * inv_g_rr2 
            + 0.5 * chi * d_g_tt**2 * inv_g_rr * inv_g_tt * inv_g_tt )
   
        Mom[:] = ( d_A_rr * inv_g_rr 
            - (2.0 / 3.0) * d_K 
            - (3.0 / 2.0) * A_rr * d_chi * inv_g_rr / chi 
            + (3.0 / 2.0) * A_rr * d_g_tt * inv_g_rr * inv_g_tt 
            - A_rr * d_g_rr * inv_g_rr2 )

        Gamcon[:] = ( -0.5 * d_g_rr * inv_g_rr2 
            + Gamma_r 
            + d_g_tt * inv_g_rr * inv_g_tt )

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
        return (420*u[1] - 378*u[2] + 180*u[3] - 35*u[4]) / 315
    elif order == 5:
        return (1155*u[1] - 1386*u[2] + 990*u[3] - 385*u[4] + 63*u[5]) / 693
    elif order == 6:
        return (6006*u[1] - 9009*u[2] + 8580*u[3] - 5005*u[4] + 1638*u[5] - 231*u[6]) / 3003
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
        return (229*u[1] - 225*u[2] + 111*u[3] - 22*u[4]) / 93
    elif order == 5:
        return (26765*u[1] - 34890*u[2] + 25770*u[3] - 10205*u[4] + 1689*u[5]) / 9129
    elif order == 6:
        return (36527*u[1] - 59295*u[2] + 58310*u[3] - 34610*u[4] + 11451*u[5] - 1627*u[6]) / 10756
    else:
        raise ValueError("Invalid order. Must be 4, 5 or 6.")
