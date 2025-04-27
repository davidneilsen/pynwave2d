import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import advection
import json
from nwave import Grid, RK4, Equations, CompactFirst2D, CompactSecond2D, ExplicitFirst44_2D, ExplicitSecond44_2D
import nwave.ioxdmf as iox

def main():
    # Read parameters
    with open("params.json") as f:
        params = json.load(f)

    g = Grid(params)
    D1 = ExplicitFirst44_2D(g.dx, g.dy)
    D2 = ExplicitSecond44_2D(g.dx, g.dy)
#    D1 = CompactFirst2D(g.x, g.y, "D1_DE4", use_banded=False)
#    D2 = CompactSecond2D(g.x, g.y, "D2_JTP6", use_banded=False)
    g.set_D1(D1)
    g.set_D2(D2)

    eqs = advection.Advection(1, g)
    eqs.initialize(g, params)
    eqs.set_alpha(params["diss_alpha"])

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    #dt = params["cfl"] * g.dx
    dt = 1.0e-3
    rk4 = RK4(eqs, g)

    time = 0.0

    func_names = [ "phi" ]
    iox.write_hdf5(0, eqs.u, g.x, g.y, func_names, output_dir)

    Nt = params["Nt"]
    for i in range(1,Nt+1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.2f}")
        if i % output_interval == 0:
            iox.write_hdf5(i, eqs.u, g.x, g.y, func_names, output_dir)

    iox.write_xdmf(output_dir, Nt, g.Nx, g.Ny, func_names, output_interval, dt)


if __name__ == "__main__":
    main()
