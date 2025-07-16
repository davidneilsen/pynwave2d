import numpy as np
import sys
import os
import csv

# from pdb import set_trace as bp
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bssneqs as bssn
import tomllib
from nwave import *

nx = 12
pars = {"Nx": nx, "NGhost": 2, "Xmin": -5.0, "Xmax": 5.0}
grid = Grid1D(pars, cell_centered=False)

ftype = FilterType.JTT8
alpha = 1.0
beta = 0.0
Pb, Q = init_JT_filter(ftype, alpha, beta, nx)

print(Pb)
print(Q)
