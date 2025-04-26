import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sowave
import json
from nwave import Grid, RK4, Equations, CompactFirst2D, CompactSecond2D
import nwave.ioxdmf as iox

def main():
    # Read parameters
    with open("params.json") as f:
        params = json.load(f)

    g = Grid(params)
#    D1 = fd.ExplicitFirst44_2D(g.dx, g.dy)
#    D2 = fd.ExplicitSecond44_2D(g.dx, g.dy)
    D1 = CompactFirst2D(g.x, g.y, "D1_JTP6", use_banded=False)
    D2 = CompactSecond2D(g.x, g.y, "D2_JTP6", use_banded=False)
    g.set_D1(D1)
    g.set_D2(D2)

    eqs = sowave.ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * g.dx
    rk4 = RK4(eqs, g, params["bctype"])

    time = 0.0

    iox.write_hdf5(0, eqs.u, g.x, g.y, output_dir)

    Nt = params["Nt"]
    for i in range(1,Nt+1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.2f}")
        iox.write_hdf5(i, eqs.u, g.x, g.y, output_dir)

    iox.write_hdf5(Nt, eqs.u, g.x, g.y, output_dir)
    iox.write_xdmf(output_dir, Nt, g.Nx, g.Ny, output_interval, dt)


if __name__ == "__main__":
    main()
