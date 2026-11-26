"""Optional lapse floor applied after each RK stage (alpha_floor).

Run: python -m unittest discover -s tests -p test_alpha_floor.py -v
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


def state(e, r):
  # A lapse that dips negative next to r = 0 and is even in r.
  u = [np.ones_like(r) for _ in e.u]
  u[e.U_GTT][:] = r * r
  u[e.U_ALPHA][:] = 0.3 * r * r - 5e-3
  return u


class AlphaFloorTests(unittest.TestCase):
  def test_validated(self):
    g = grid(True)
    self.assertEqual(b.BSSN(g, 1., 2., True).alpha_floor, 0.0)
    self.assertEqual(b.BSSN(g, 1., 2., True, alpha_floor=1e-12).alpha_floor, 1e-12)
    for bad in (-1e-12, np.nan, np.inf, True, "1e-12", None):
      with self.assertRaises(ValueError):
        b.BSSN(g, 1., 2., True, alpha_floor=bad)

  def test_off_leaves_alpha(self):
    for ext in (False, True):
      g = grid(ext); r = g.xi[0]
      e = b.BSSN(g, 1., 2., ext, enforce_det=False)
      u = state(e, r); a0 = u[e.U_ALPHA].copy()
      e.apply_bcs(u, g)
      if ext:
        np.testing.assert_array_equal(u[e.U_ALPHA], a0)
      self.assertLess(u[e.U_ALPHA].min(), 0.)

  def test_floor_clips_only_below(self):
    for ext in (False, True):
      g = grid(ext); r = g.xi[0]
      off = b.BSSN(g, 1., 2., ext, enforce_det=False)
      on = b.BSSN(g, 1., 2., ext, enforce_det=False, alpha_floor=1e-6)
      uo, un = state(off, r), state(on, r)
      off.apply_bcs(uo, g); on.apply_bcs(un, g)
      ao, an = uo[off.U_ALPHA], un[on.U_ALPHA]
      self.assertEqual(an.min(), 1e-6)
      keep = ao >= 1e-6
      np.testing.assert_array_equal(an[keep], ao[keep])
      np.testing.assert_array_equal(an[~keep], 1e-6)
      for i in range(len(un)):
        if i != on.U_ALPHA:
          np.testing.assert_array_equal(un[i], uo[i])

  def test_half_domain_parity_kept(self):
    g = grid(False); r = g.xi[0]; ng = g.nghost
    e = b.BSSN(g, 1., 2., False, enforce_det=False, alpha_floor=1e-6)
    u = state(e, r)
    e.apply_bcs(u, g)
    a = u[e.U_ALPHA]
    np.testing.assert_array_equal(a[:ng], a[2 * ng - 1:ng - 1:-1])


if __name__ == "__main__":
  unittest.main()
