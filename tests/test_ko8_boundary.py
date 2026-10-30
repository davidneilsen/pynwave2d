"""Regressions for the KO8 left-endpoint stencil.

Run: python -m unittest discover -s tests -p test_ko8_boundary.py -v
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nwave.filters import KreissOligerFilterO8_1D
from nwave.rk import RK4


class KO8Boundary(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-2, 2, 81)
        self.dx = self.x[1] - self.x[0]
        self.f = KreissOligerFilterO8_1D(self.dx, 0.5, True)

    def test_annihilates_low_polynomials(self):
        # The boundary rows annihilate degrees 0..2; this does not claim
        # eighth-order accuracy of the lower-order boundary closure.
        for p in range(3):
            with self.subTest(power=p):
                self.assertLess(np.max(abs(self.f.filter(self.x**p))), 1e-12)

    def test_reflection_for_nonsymmetric_data(self):
        u = np.random.default_rng(11).normal(size=81)
        np.testing.assert_allclose(
            self.f.filter(u[::-1]), self.f.filter(u)[::-1], atol=1e-13, rtol=1e-14)

    def test_linearity(self):
        rng = np.random.default_rng(2)
        u, v = rng.normal(size=81), rng.normal(size=81)
        np.testing.assert_allclose(
            self.f.filter(2 * u - 3 * v),
            2 * self.f.filter(u) - 3 * self.f.filter(v), atol=1e-13, rtol=1e-13)

    def test_endpoint_is_reflected_right_formula(self):
        u = np.random.default_rng(3).normal(size=81)
        expected = 0.5 * (-u[4] + 4 * u[3] - 6 * u[2] + 4 * u[1] - u[0]) / (17 / 48 * 256 * self.dx)
        self.assertAlmostEqual(self.f.filter(u)[0], expected, places=14)

    def test_only_boundary_enabled_endpoint_changes(self):
        # Interior rows must agree exactly with the boundary-off filter.
        u = np.random.default_rng(7).normal(size=81)
        off = KreissOligerFilterO8_1D(self.dx, 0.5, False).filter(u)
        on = self.f.filter(u)
        np.testing.assert_array_equal(on[4:-4], off[4:-4])
        np.testing.assert_array_equal(off[:4], np.zeros(4))
        np.testing.assert_array_equal(off[-4:], np.zeros(4))

    def test_active_dissipation_evolution_preserves_parity(self):
        class Eq:
            Nu = 1
            apply_bc = None
            shp = [81]

            def __init__(self, x):
                self.u = [np.exp(-x * x) * np.cos(8 * x)]

            def rhs(self, out, u, g):
                out[0][:] = 0

        e = Eq(self.x)
        initial = e.u[0].copy()
        g = SimpleNamespace(num_filters=1, Filter=[self.f])
        rk = RK4(e, g)
        for _ in range(100):
            rk.step(e, g, 0.01 * self.dx)
        self.assertTrue(np.isfinite(e.u[0]).all())
        np.testing.assert_allclose(e.u[0], e.u[0][::-1], atol=3e-14, rtol=0)
        self.assertLess(np.linalg.norm(e.u[0]), np.linalg.norm(initial))


if __name__ == '__main__':
    unittest.main()
