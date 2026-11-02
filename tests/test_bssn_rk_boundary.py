"""Regressions for BSSN.apply_bcs acting on the supplied RK stage.

Run: python -m unittest discover -s tests -p test_bssn_rk_boundary.py -v
BSSN_TEST_ROOT optionally selects an unmodified source tree as a baseline.
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


class StageBoundaryTests(unittest.TestCase):
    def test_stage_regularization(self):
        g = Grid1D({'Nx': 21, 'NGhost': 3, 'Xmin': 0., 'Xmax': 2.}, cell_centered=True)
        e = b.BSSN(g, 1., 2., False)
        for v in e.u:
            v[:] = np.arange(len(v))
        base = [v.copy() for v in e.u]
        stage = [v.copy() + 100 for v in e.u]
        e.apply_bcs(stage, g)
        for original, current in zip(base, e.u):
            np.testing.assert_array_equal(current, original)
        for j in range(9):
            sign = -1 if j in (e.U_GT, e.U_SHIFT, e.U_GB) else 1
            np.testing.assert_array_equal(stage[j][:3], sign * stage[j][5:2:-1])
            np.testing.assert_array_equal(stage[j][3:], base[j][3:] + 100)

    def test_extended_unchanged(self):
        g = Grid1D({'Nx': 21, 'NGhost': 3, 'Xmin': -2., 'Xmax': 2.}, cell_centered=True)
        e = b.BSSN(g, 1., 2., True)
        stage = [np.arange(len(v), dtype=float) + 100 for v in e.u]
        before = [v.copy() for v in stage]
        e.apply_bcs(stage, g)
        for a, z in zip(before, stage):
            np.testing.assert_array_equal(a, z)


if __name__ == "__main__":
    unittest.main()
