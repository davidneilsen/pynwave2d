"""Opt-in log KO on the conformal metric (log_ko).

Run: python -m unittest discover -s tests -p test_log_ko.py -v
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(os.environ.get("BSSN_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
import bssneqs as b
import bssnutils as butil
from nwave import Grid1D, BCType, RK4

PARAMS = {'D1': 'E642', 'D2': 'E642', 'BHMask': False, 'Mass': 1.0,
          'collapsed_lapse': True, 'initial_data': 'Puncture',
          'filters': [{'type': 'KO6', 'apply': 'Rhs', 'sigma': 0.5,
                       'filter_boundary': False}]}


def setup(ext, log_ko, enforce_det=False, nx=161, xmax=8.0):
  p = dict(PARAMS, Nx=nx, NGhost=3, Xmin=-xmax if ext else 0.0, Xmax=xmax)
  g = Grid1D(p, cell_centered=True)
  r = g.xi[0]
  D1, D2 = butil.init_derivative_operators(r, p)
  g.set_D1(D1)
  g.set_D2(D2)
  g.set_filter(butil.init_filter(r, p))
  e = b.BSSN(g, 1.0, 2.0, ext, BCType.FUNCTION,
             enforce_det=enforce_det, log_ko=log_ko)
  e.initialize(g, p)
  return e, g


def ko_terms(e, g, a, q):
  fx = g.Filter[0]
  return e.filter_rhs(e.U_GRR, a, fx), e.filter_rhs(e.U_GTT, q, fx)


def det(e, g):
  r = g.xi[0]
  return e.u[e.U_GRR] * (e.u[e.U_GTT] / (r * r)) ** 2


class LogKOTests(unittest.TestCase):
  def test_flag_validated(self):
    g = setup(True, False)[1]
    with self.assertRaises(ValueError):
      b.BSSN(g, 1.0, 2.0, True, BCType.FUNCTION, log_ko=1)

  def test_off_is_plain_filter(self):
    # Default: every field gets fx.filter(u), bitwise.
    e, g = setup(True, False)
    self.assertFalse(e.log_ko)
    fx = g.Filter[0]
    for i, ui in enumerate(e.u):
      np.testing.assert_array_equal(e.filter_rhs(i, ui, fx), fx.filter(ui))

  def test_on_only_touches_metric(self):
    e, g = setup(True, True)
    fx = g.Filter[0]
    for i, ui in enumerate(e.u):
      if i in (e.U_GRR, e.U_GTT):
        continue
      np.testing.assert_array_equal(e.filter_rhs(i, ui, fx), fx.filter(ui))

  def test_unit_det_is_kept(self):
    # On det = 1 data the log-KO terms are det-tangent:
    # Ka/a + 2 Kq/q = KO(ln det) = 0; standard KO is not.
    rng = np.random.default_rng(3)
    for ext in (False, True):
      on, g = setup(ext, True)
      off = setup(ext, False)[0]
      r = g.xi[0]
      a = 1.0 + 0.3 * rng.uniform(-1, 1, r.size)
      q = r * r / np.sqrt(a)
      Ka, Kq = ko_terms(on, g, a, q)
      self.assertLess(np.max(np.abs(Ka / a + 2 * Kq / q)), 1e-12)
      Ka, Kq = ko_terms(off, g, a, q)
      self.assertGreater(np.max(np.abs(Ka / a + 2 * Kq / q)), 1e-3)

  def test_det_rate_is_ko_of_ln_det(self):
    # Off det = 1: the filter's d ln det / dt is exactly KO(ln det).
    rng = np.random.default_rng(5)
    e, g = setup(True, True)
    r = g.xi[0]
    a = 1.0 + 0.3 * rng.uniform(-1, 1, r.size)
    q = r * r * (1.0 + 0.3 * rng.uniform(-1, 1, r.size))
    Ka, Kq = ko_terms(e, g, a, q)
    expect = g.Filter[0].filter(np.log(a * (q / (r * r)) ** 2))
    np.testing.assert_allclose(Ka / a + 2 * Kq / q, expect, rtol=0,
                               atol=1e-10 * np.max(np.abs(expect)))

  def test_evolution_det_drift_smaller(self):
    # Puncture, no det projection: log KO drifts det far less near r = 0.
    drift = {}
    for log_ko in (False, True):
      e, g = setup(False, log_ko)
      rk = RK4(e, g)
      for _ in range(40):
        rk.step(e, g, 0.1 * g.dx[0])
      self.assertTrue(all(np.isfinite(x).all() for x in e.u))
      drift[log_ko] = np.max(np.abs(det(e, g) - 1.0))
    self.assertLess(drift[True], 0.1 * drift[False])


if __name__ == '__main__':
  unittest.main()
