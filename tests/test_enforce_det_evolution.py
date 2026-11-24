"""det(gtilde) = 1 enforcement in evolution: RK steps, parity, the off path,
initial-data re-splitting, and short end-to-end solver runs.

Run: python -m unittest discover -s tests -p test_enforce_det_evolution.py -v
PYNWAVE_BASE_ROOT=<tree without enforce_det> also checks that
enforce_det = false is bitwise identical to that tree.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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


def setup(ext, enforce_det, nx=161, xmax=8.0, initial_data='Puncture'):
    p = dict(PARAMS, Nx=nx, NGhost=3, Xmin=-xmax if ext else 0.0, Xmax=xmax,
             initial_data=initial_data)
    g = Grid1D(p, cell_centered=True)
    r = g.xi[0]
    D1, D2 = butil.init_derivative_operators(r, p)
    g.set_D1(D1)
    g.set_D2(D2)
    g.set_filter(butil.init_filter(r, p))
    e = b.BSSN(g, 1.0, 2.0, ext, BCType.FUNCTION, enforce_det=enforce_det)
    e.initialize(g, p)
    return e, g


def det(e, g):
    r = g.xi[0]
    return e.u[e.U_GRR] * (e.u[e.U_GTT] / (r * r)) ** 2


def evolve(e, g, n, cfl=0.1):
    rk = RK4(e, g)
    for _ in range(n):
        rk.step(e, g, cfl * g.dx[0])


class StepTests(unittest.TestCase):
    def test_unit_det_after_steps(self):
        for ext in (False, True):
            e, g = setup(ext, True)
            evolve(e, g, 20)
            # Interior points; mirror ghosts carry their own round-off in r.
            ng = g.nghost
            np.testing.assert_allclose(det(e, g)[ng:-ng], 1.0, rtol=0, atol=1e-14)
            off, g2 = setup(ext, False)
            evolve(off, g2, 20)
            # Without the projection the first cells drift (KO6 on g_rr, g_tt).
            self.assertGreater(np.max(np.abs(det(off, g2) - 1.0)), 1e-7)

    def test_origin_parity(self):
        # Half domain: the mirror ghosts are exact after enforcement.
        e, g = setup(False, True)
        evolve(e, g, 20)
        ng = g.nghost
        for k in (e.U_CHI, e.U_GRR, e.U_GTT, e.U_ARR, e.U_K, e.U_ALPHA):
            np.testing.assert_array_equal(e.u[k][:ng], e.u[k][2 * ng - 1:ng - 1:-1])
        for k in (e.U_GT, e.U_SHIFT, e.U_GB):
            np.testing.assert_array_equal(e.u[k][:ng], -e.u[k][2 * ng - 1:ng - 1:-1])
        # Extended domain: the evolved state stays reflection symmetric.
        e, g = setup(True, True)
        evolve(e, g, 20)
        for k, parity in ((e.U_GRR, 1), (e.U_GTT, 1), (e.U_CHI, 1), (e.U_GT, -1)):
            u = e.u[k]
            np.testing.assert_allclose(u, parity * u[::-1], rtol=1e-11,
                                       atol=1e-11 * np.max(np.abs(u)))

    def test_half_and_extended_agree(self):
        # Same physical problem: the mirrored half domain matches the
        # extended domain's r > 0 half with enforcement on.
        h, gh = setup(False, True, nx=81)
        x, gx = setup(True, True, nx=161)
        evolve(h, gh, 20)
        evolve(x, gx, 20)
        rh, rx = gh.xi[0], gx.xi[0]
        ih = np.flatnonzero((rh > 0) & (rh < 4.0))
        ix = np.abs(rx[None, :] - rh[ih, None]).argmin(axis=1)
        np.testing.assert_allclose(rx[ix], rh[ih], rtol=0, atol=1e-12)
        for k in (h.U_GRR, h.U_GTT, h.U_ALPHA):
            np.testing.assert_allclose(h.u[k][ih], x.u[k][ix], rtol=1e-6)


class OffPathTests(unittest.TestCase):
    def test_off_never_projects(self):
        with mock.patch.object(b, 'enforce_unit_det') as stage, \
             mock.patch.object(b.BSSN, 'project_initial_det') as initial:
            for ext in (False, True):
                e, g = setup(ext, False)
                evolve(e, g, 2)
            stage.assert_not_called()
            initial.assert_not_called()

    @unittest.skipUnless(os.environ.get('PYNWAVE_BASE_ROOT'),
                         'set PYNWAVE_BASE_ROOT to a tree without enforce_det')
    def test_off_bitwise_equals_base(self):
        code = r'''import sys, numpy as np
root, ext, off, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4]
sys.path[:0] = [root + "/bssn", root]
import bssneqs as b, bssnutils as butil
from nwave import Grid1D, BCType, RK4
p = {"D1": "E642", "D2": "E642", "BHMask": False, "Mass": 1.0, "collapsed_lapse": True,
     "initial_data": "Puncture", "Nx": 161, "NGhost": 3, "Xmin": -8.0 if ext else 0.0,
     "Xmax": 8.0, "filters": [{"type": "KO6", "apply": "Rhs", "sigma": 0.5,
                               "filter_boundary": False}]}
g = Grid1D(p, cell_centered=True); r = g.xi[0]
D1, D2 = butil.init_derivative_operators(r, p); g.set_D1(D1); g.set_D2(D2)
g.set_filter(butil.init_filter(r, p))
kw = {"enforce_det": False} if off else {}
e = b.BSSN(g, 1.0, 2.0, ext, BCType.FUNCTION, **kw); e.initialize(g, p)
rk = RK4(e, g)
for _ in range(40):
    rk.step(e, g, 0.1 * g.dx[0])
np.save(out, np.array(e.u))
'''
        base = os.environ['PYNWAVE_BASE_ROOT']
        with tempfile.TemporaryDirectory() as d:
            for ext in ('0', '1'):
                outs = []
                for root, off in ((str(ROOT), '1'), (base, '0')):
                    out = f'{d}/{Path(root).name}_{ext}.npy'
                    subprocess.run([sys.executable, '-c', code, root, ext, off, out],
                                   check=True, capture_output=True, timeout=300)
                    outs.append(np.load(out))
                np.testing.assert_array_equal(outs[0], outs[1])


class InitialDataTests(unittest.TestCase):
    def test_unit_det_data_untouched(self):
        for ext, idata in ((False, 'Puncture'), (True, 'Puncture'),
                           (False, 'EddingtonFinkelstein')):
            on, _ = setup(ext, True, initial_data=idata)
            off, g = setup(ext, False, initial_data=idata)
            if np.max(np.abs(det(off, g) - 1.0)) > b.BSSN.INITIAL_DET_TOL:
                continue  # e.g. EF data with a blended interior (#14)
            for u1, u2 in zip(on.u, off.u):
                np.testing.assert_array_equal(u1, u2)

    def resplit_error(self, nx):
        # EF data (on det = 1) given a different conformal split with
        # det = f^-3: the projection must restore the original split, Gt to
        # the D1 truncation error.  ref is itself projected, so this holds
        # for any EF data, blended or not.
        ref, g = setup(False, True, nx=nx, xmax=10.0, initial_data='EddingtonFinkelstein')
        e, _ = setup(False, True, nx=nx, xmax=10.0, initial_data='EddingtonFinkelstein')
        r = g.xi[0]
        f = 1.0 + 0.2 * np.exp(-(r - 4.0) ** 2)
        dlnf = -0.4 * (r - 4.0) * np.exp(-(r - 4.0) ** 2) / f
        g_rr0 = ref.u[ref.U_GRR]
        for k in (e.U_GRR, e.U_GTT, e.U_CHI, e.U_ARR):
            e.u[k][:] = ref.u[k] / f
        e.u[e.U_GT][:] = f * ref.u[ref.U_GT] + 0.5 * dlnf * f / g_rr0
        e.project_initial_det(g)
        # Points already within INITIAL_DET_TOL of det = 1 are left alone.
        tol = b.BSSN.INITIAL_DET_TOL
        np.testing.assert_allclose(det(e, g), 1.0, rtol=0, atol=tol)
        for k in (e.U_GRR, e.U_GTT, e.U_CHI, e.U_ARR):
            np.testing.assert_allclose(e.u[k], ref.u[k], rtol=2 * tol)
        m = (r > 1.5) & (r < 8.5)
        return np.max(np.abs(e.u[e.U_GT][m] - ref.u[ref.U_GT][m]))

    def test_resplit_restores_data_and_converges(self):
        e1, e2 = self.resplit_error(201), self.resplit_error(401)
        self.assertLess(e2, 1e-6)
        self.assertGreater(e1 / e2, 2 ** 5)   # E642 interior is 6th order

    def test_bad_initial_det_raises(self):
        e, g = setup(True, True)
        e.u[e.U_GRR][4] = -1.0
        with self.assertRaises(ValueError):
            e.project_initial_det(g)


SOLVER_PAR = """Nx = {nx}
Nt = 40
NGhost = 3
Xmin = {xmin}
Xmax = 10.0
cfl = 0.1
D1 = "E642"
D2 = "E642"
BHMask = false
Mass = 1.0
collapsed_lapse = true
initial_data = "Puncture"
enforce_det = {on}
output_interval = 40
print_interval = 20
bhmask_outer_radius = 9.0
output_dir = "{out}"

[shift_damping]
type = "constant"
eta = 2.0

[[filters]]
type = "KO6"
apply = "Rhs"
sigma = 0.5
filter_boundary = false
"""


def read_curve(path, name):
    vals, cur = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('# TIME'):
            continue
        if line.startswith('# '):
            cur = line[2:].strip()
            vals[cur] = []
        else:
            vals[cur].append([float(x) for x in line.split()])
    return np.array(vals[name])


class SolverSmokeTests(unittest.TestCase):
    def test_solver_end_to_end(self):
        for xmin, nx in ((0.0, 100), (-10.0, 200)):
            with tempfile.TemporaryDirectory() as d:
                res = {}
                for on in ('true', 'false'):
                    par = Path(d) / f'p_{on}.toml'
                    par.write_text(SOLVER_PAR.format(nx=nx, xmin=xmin, on=on,
                                                     out=f'{d}/{on}'))
                    p = subprocess.run([sys.executable, str(ROOT / 'bssn' / 'solver.py'), str(par)],
                                       capture_output=True, text=True, timeout=300,
                                       cwd=str(ROOT / 'bssn'))
                    self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                    curve = f'{d}/{on}_0/bssn_0040.curve'
                    grr, gtt = read_curve(curve, 'grr'), read_curve(curve, 'gtt')
                    r = grr[:, 0]
                    res[on] = (r, grr[:, 1] * (gtt[:, 1] / r ** 2) ** 2,
                               np.loadtxt(f'{d}/{on}_0/bssn_constraints.dat',
                                          delimiter=',', skiprows=1, usecols=(1, 2, 7)))
                r, det_on, con_on = res['true']
                _, det_off, con_off = res['false']
                # %.8e curve output limits the check to ~1e-7.
                np.testing.assert_allclose(det_on, 1.0, rtol=0, atol=3e-7)
                self.assertTrue(np.all(np.isfinite(con_on)))
                # Away from the puncture the two runs agree closely.
                far = np.abs(r) > 3.0
                self.assertLess(np.max(np.abs(det_off[far] - 1.0)), 1e-5)
                np.testing.assert_allclose(con_on[-1], con_off[-1], rtol=0.5)


if __name__ == "__main__":
    unittest.main()
