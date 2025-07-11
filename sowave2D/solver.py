import numpy as np
import sys
import os
import tomllib

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sowave
import sowaveutils as sutils
from nwave import *
import nwave.ioxdmf as iox


def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]
    dx = g.dx[0]
    dy = g.dx[1]
    D1, D2 = sutils.init_derivative_operators(x, y, params)
    g.set_D1(D1)
    g.set_D2(D2)

    eqs = sowave.ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dx
    rk4 = RK4(eqs, g)

    time = 0.0
    func_names = ["phi", "chi"]
    iox.write_hdf5(0, eqs.u, x, y, func_names, output_dir)

    Nt = params["Nt"]
    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.2f}")
        if i % output_interval == 0:
            iox.write_hdf5(i, eqs.u, x, y, func_names, output_dir)

    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)

