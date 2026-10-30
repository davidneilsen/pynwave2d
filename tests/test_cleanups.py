"""Regressions for the upstream-cleanups changes: imports without the
typing.Any workaround, optional pentapy, and the laplacian_alpha refactor.

Run: python -m unittest discover -s tests -p test_cleanups.py -v
"""
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bssn'))
from bssneqs import BSSN


def run_python(code):
    return subprocess.run([sys.executable, '-c', code, str(ROOT)],
                          capture_output=True, text=True, timeout=120)


class ImportTests(unittest.TestCase):
    def test_bssn_modules_import_cleanly(self):
        # bssnutils used typing.Any without importing it.
        p = run_python("import sys\n"
                       "sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]\n"
                       "import bssnutils, solver")
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_pentapy_is_optional(self):
        # sys.modules[name] = None makes 'import pentapy' raise ImportError.
        code = """import sys
sys.modules['pentapy'] = None
sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]
import numpy as np
import bssnutils, solver
from nwave import NCompactDerivative, DerivType, CFDSolve
r = np.linspace(-10., 10., 61)
f = np.sin(r)
NCompactDerivative.deriv(r, DerivType.D1_JP6, CFDSolve.LUSOLVE).grad(f)
try:
    NCompactDerivative.deriv(r, DerivType.D1_JP6, CFDSolve.PENTAPY).grad(f)
except ImportError as err:
    print('PENTAPY:', err)
"""
        p = run_python(code)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('PENTAPY: pentapy is required', p.stdout)


class LaplacianAlphaTests(unittest.TestCase):
    r = np.linspace(0.5, 6.0, 41)

    # Smooth, non-trivial metric and lapse, with exact first and second
    # derivatives.
    @staticmethod
    def fields(r):
        chi = 1.0 / (1.0 + 0.5 / r) ** 4
        g_rr = 1.0 + 0.3 * np.exp(-r)
        g_tt = r**2 * (1.0 + 0.2 * np.sin(r))
        alpha = 1.0 - 0.4 * np.exp(-0.5 * r**2) + 0.1 * np.cos(2 * r)
        return chi, g_rr, g_tt, alpha

    def derivs(self, r):
        psi = 1.0 + 0.5 / r
        d_chi = 2.0 * psi**-5 / r**2
        d_g_rr = -0.3 * np.exp(-r)
        d_g_tt = 2 * r * (1.0 + 0.2 * np.sin(r)) + 0.2 * r**2 * np.cos(r)
        d_alpha = 0.4 * r * np.exp(-0.5 * r**2) - 0.2 * np.sin(2 * r)
        d2_alpha = 0.4 * (1 - r**2) * np.exp(-0.5 * r**2) - 0.4 * np.cos(2 * r)
        return d_chi, d_g_rr, d_g_tt, d_alpha, d2_alpha

    def laplacian(self, r):
        chi, g_rr, g_tt, _ = self.fields(r)
        d_chi, d_g_rr, d_g_tt, d_alpha, d2_alpha = self.derivs(r)
        return BSSN.laplacian_alpha(chi, g_rr, g_tt, d_alpha, d_chi,
                                    d_g_rr, d_g_tt, d2_alpha)

    def test_matches_divergence_form(self):
        """D^2 alpha = (1/sqrt(gamma)) d_r (sqrt(gamma) gamma^rr d_r alpha),
        gamma_ij = chi^-1 diag(g_rr, g_tt, g_tt sin^2), by central differences."""
        def flux(r):
            chi, g_rr, g_tt, _ = self.fields(r)
            d_alpha = self.derivs(r)[3]
            return chi**-0.5 * g_rr**-0.5 * g_tt * d_alpha  # sqrt(gamma) gamma^rr alpha'

        h = 1e-5
        r = self.r
        chi, g_rr, g_tt, _ = self.fields(r)
        sqrt_gamma = chi**-1.5 * g_rr**0.5 * g_tt
        expected = (flux(r + h) - flux(r - h)) / (2 * h) / sqrt_gamma
        np.testing.assert_allclose(self.laplacian(r), expected, rtol=1e-8, atol=1e-10)

    def test_flat_space(self):
        r = self.r
        one = np.ones_like(r)
        _, _, _, d_alpha, d2_alpha = self.derivs(r)
        lap = BSSN.laplacian_alpha(one, one, r**2, d_alpha, 0 * r, 0 * r, 2 * r, d2_alpha)
        np.testing.assert_allclose(lap, d2_alpha + 2 * d_alpha / r, rtol=1e-14)

    def test_rhs_K_matches_pre_refactor_expression(self):
        """rhs_K now takes lap_alpha; agreement with the previous inline
        expression is to roundoff, not bitwise (terms are regrouped)."""
        rng = np.random.default_rng(4)
        r = self.r
        chi, g_rr, g_tt, alpha = self.fields(r)
        d_chi, d_g_rr, d_g_tt, d_alpha, d2_alpha = self.derivs(r)
        A_rr, K, beta_r, ad_K = (rng.normal(size=r.size) for _ in range(4))
        inv_g_rr, inv_g_tt = 1.0 / g_rr, 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        previous = (
            3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * ad_K
            + chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            - chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            + d_alpha * d_chi * inv_g_rr / 2
            - chi * d2_alpha * inv_g_rr
        )
        K_rhs = np.empty_like(r)
        BSSN.rhs_K(K_rhs, alpha, A_rr, K, g_rr, beta_r, ad_K, self.laplacian(r))
        np.testing.assert_allclose(K_rhs, previous, rtol=1e-12, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
