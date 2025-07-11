import numpy as np
import sys
import os
import tomllib

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import advection
import advectutils as autil
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

    D1, D2 = autil.init_derivative_operators(x, y, params)
    g.set_D1(D1)
    g.set_D2(D2)

    """
    Need to write 2D filters

    F1 = KreissOligerFilterO6_2D( dx, 0.1, apply_diss_boundaries=False)
    g.set_filter(F1)
    print(f"Filter type: {g.Filter.get_filter_type()}")
    print(f"Filter apply: {g.Filter.get_apply_filter()}")
    print(f"Filter sigma: {F1.get_sigma()}")
    """

    eqs = advection.Advection(1, g)
    eqs.initialize(g, params)
    eqs.set_alpha(params["diss_alpha"])

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    # dt = params["cfl"] * dx
    dt = 1.0e-3
    rk4 = RK4(eqs, g)

    time = 0.0

    func_names = ["phi"]
    iox.write_hdf5(0, eqs.u, x, y, func_names, output_dir)

    Nt = params["Nt"]
    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.2f}")
        if i % output_interval == 0:
            iox.write_hdf5(i, eqs.u, x, y, func_names, output_dir)

    Nx = g.shp[0]
    Ny = g.shp[1]
    iox.write_xdmf(output_dir, Nt, Nx, Ny, func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
