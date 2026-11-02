"""Regression for the Gamcon constraint column.

Run: python -m unittest discover -s tests -p test_constraint_diagnostics.py -v
"""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

CONFIG = """Nx=80
Nt=4
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
output_interval=10
print_interval=2
"""
RUN = """import sys
sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]
import solver
solver.main(sys.argv[2])
"""


def run_solver():
    d = tempfile.TemporaryDirectory()
    par = Path(d.name) / "params.toml"
    par.write_text(CONFIG + f'output_dir="{d.name}/out"\n')
    p = subprocess.run([sys.executable, '-c', RUN, str(ROOT), str(par)],
                       capture_output=True, text=True, timeout=120)
    return d, p


class ConstraintDiagnostics(unittest.TestCase):
    def test_gamcon_column(self):
        d, p = run_solver()
        with d:
            self.assertEqual(p.returncode, 0, p.stderr)
            with (Path(d.name) / 'out_0/bssn_constraints.dat').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)  # t = 0 and two print intervals
            for row in rows:  # truncation error makes it nonzero even at t = 0
                g = float(row['Gamcon'])
                self.assertTrue(math.isfinite(g) and g > 0.0, row)
                self.assertNotIn(g, (float(row['Ham']), float(row['Mom'])), row)
            self.assertIn('|Gamcon|=', p.stdout)


if __name__ == '__main__':
    unittest.main()
