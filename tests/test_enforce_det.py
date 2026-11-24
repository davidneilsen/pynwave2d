"""Opt-in det(gtilde) = 1 enforcement after each RK stage.

Run: python -m unittest discover -s tests -p test_enforce_det.py -v
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(os.environ.get("BSSN_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
import bssneqs as b
from nwave import Grid1D


def grid(ext):
    xmin = -2. if ext else 0.
    return Grid1D({'Nx': 41, 'NGhost': 3, 'Xmin': xmin, 'Xmax': 2.}, cell_centered=True)


def perturbed_metric(r, seed=7):
    rng = np.random.default_rng(seed)
    a = 1. + 0.3 * rng.uniform(-1, 1, r.size)
    q = r * r * (1. + 0.3 * rng.uniform(-1, 1, r.size))
    return a, q


class EnforceDetTests(unittest.TestCase):
    def test_unit_det_and_isotropic(self):
        for ext in (False, True):
            g = grid(ext); r = g.xi[0]
            a, q = perturbed_metric(r)
            ratio = a * r * r / q
            b.enforce_unit_det(a, q, r)
            np.testing.assert_allclose(a * (q / (r * r)) ** 2, 1., rtol=0, atol=1e-14)
            np.testing.assert_allclose(a * r * r / q, ratio, rtol=1e-14)

    def test_idempotent(self):
        r = grid(True).xi[0]
        a, q = perturbed_metric(r)
        b.enforce_unit_det(a, q, r)
        a1, q1 = a.copy(), q.copy()
        b.enforce_unit_det(a, q, r)
        np.testing.assert_allclose(a, a1, rtol=1e-15)
        np.testing.assert_allclose(q, q1, rtol=1e-15)

    def test_bad_det_raises(self):
        r = grid(True).xi[0]
        for bad in (-1., 0., np.nan):
            a, q = perturbed_metric(r)
            a[5] = bad
            with self.assertRaises(FloatingPointError):
                b.enforce_unit_det(a, q, r)

    def test_apply_bcs_hook(self):
        for ext in (False, True):
            g = grid(ext); r = g.xi[0]
            on = b.BSSN(g, 1., 2., ext, enforce_det=True)
            off = b.BSSN(g, 1., 2., ext)
            a, q = perturbed_metric(r)
            st_on = [np.ones_like(r) for _ in on.u]
            st_on[on.U_GRR][:], st_on[on.U_GTT][:] = a, q
            st_off = [v.copy() for v in st_on]
            on.apply_bcs(st_on, g)
            off.apply_bcs(st_off, g)
            det = st_on[on.U_GRR] * (st_on[on.U_GTT] / (r * r)) ** 2
            np.testing.assert_allclose(det, 1., rtol=0, atol=1e-14)
            self.assertFalse(np.allclose(st_off[off.U_GRR] * (st_off[off.U_GTT] / (r * r)) ** 2, 1.))

    def test_default_off_and_validated(self):
        g = grid(True)
        self.assertFalse(b.BSSN(g, 1., 2., True).enforce_det)
        with self.assertRaises(ValueError):
            b.BSSN(g, 1., 2., True, enforce_det="yes")


if __name__ == "__main__":
    unittest.main()
