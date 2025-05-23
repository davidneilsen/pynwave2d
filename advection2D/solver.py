import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import advection
import json
from nwave import *
import nwave.ioxdmf as iox


def main():
    # Read parameters
    with open("params.json") as f:
        params = json.load(f)

    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]
    dx = g.dx[0]
    dy = g.dx[1]
    #    D1 = ExplicitFirst44_2D(dx, dy)
    #    D2 = ExplicitSecond44_2D(dx, dy)
    D1 = CompactFirst2D(x, y, "D1_JTP6")
    D2 = CompactSecond2D(x, y, "D2_JTP6")
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
    main()
