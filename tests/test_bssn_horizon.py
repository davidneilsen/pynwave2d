"""Regressions for horizon root finding and recovery after a failed search.

Run: python -m unittest discover -s tests -p test_bssn_horizon.py -v
BSSN_TEST_ROOT optionally selects an unmodified source tree as a baseline.
"""
import builtins
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import typing
import unittest

import numpy as np

ROOT = Path(os.environ.get("BSSN_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
# Upstream bssnutils uses Any without importing it (fixed in #2); remove this
# line, and the copy in the subprocess snippet, once #2 merges.
builtins.Any = typing.Any
import bssneqs as b
import solver

CONFIG = """Nx=80
Nt=2
NGhost=3
Xmin=-20.0
Xmax=20.0
cfl=0.1
eta=2.0
D1="E642"
D2="E642"
BHMask=false
Mass=1.0
collapsed_lapse=true
initial_data="Puncture"
output_interval=1
print_interval=1
bhmask_outer_radius=15.0
"""


class HorizonTests(unittest.TestCase):
    def test_horizon_roots(self):
        x = np.linspace(-2, 2, 17)
        for root in (0.5, 0.51, -0.5, -0.51):
            lo, hi = (0.1, 1.5) if root > 0 else (-1.5, -0.1)
            self.assertAlmostEqual(b.find_ah_zero(x - root, x, lo, hi), root)
        self.assertAlmostEqual(b.find_ah_zero((x - 0.5) * (x - 1.), x, 0.1, 1.5), 1.)
        self.assertIsNone(b.find_ah_zero(np.ones_like(x), x, 0.1, 1.5))
        self.assertIsNone(b.find_ah_zero(x - 0.5, x, 0.6, 1.5))

    def test_horizon_tracking(self):
        good = solver.update_horizon_tracking(0.5, 1., 0.7, 0.)
        self.assertEqual(good, (0.5, 1., 0.5, 0.5))
        failed = solver.update_horizon_tracking(None, None, good[2], good[3])
        self.assertEqual(failed, (0., 0., 0.5, 0.5))
        recovered = solver.update_horizon_tracking(0.51, 1., failed[2], failed[3])
        self.assertEqual(recovered, (0.51, 1., 0.51, 0.51))

    def test_solver_recovers_and_reports_mask(self):
        with tempfile.TemporaryDirectory() as d:
            par = Path(d) / "params.toml"
            par.write_text(CONFIG + f'output_dir="{d}/out"\n')
            # Injected detections: 0.5, then a failure, then 0.51 only if the
            # search guess survived the failure.
            code = """import sys, builtins, typing, json
import numpy as np
builtins.Any = typing.Any
sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]
import solver
calls = []
def find(self, g, guess):
    calls.append(float(guess))
    n = len(calls)
    radius = 0.5 if n == 1 else None if n == 2 else 0.51 if guess > 0 else None
    return radius, None if radius is None else 1., np.zeros_like(g.xi[0])
solver.bssn.BSSN.find_horizon = find
solver.main(sys.argv[2])
print('GUESSES=' + json.dumps(calls))
"""
            p = subprocess.run([sys.executable, '-c', code, str(ROOT), str(par)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            guesses = json.loads(p.stdout.split('GUESSES=')[-1])
            self.assertEqual(guesses, [0.7, 0.5, 0.5])
            with (Path(d) / 'out_0/bssn_constraints.dat').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['horizon_found'] for r in rows], ['True', 'False', 'True'])
            self.assertEqual([float(r['mask_inner_radius']) for r in rows], [0.5, 0.5, 0.51])
            self.assertEqual([float(r['radius_bh']) for r in rows], [0.5, 0., 0.51])


if __name__ == "__main__":
    unittest.main()
