import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bssneqs as bssn
import json
from nwave import Grid1D, RK4, Equations, ExplicitFirst44_1D, ExplicitSecond44_1D

def main():
    # Read parameters
    with open("params.json") as f:
        params = json.load(f)

    g = Grid1D.BH_grid(params["Nr"], params["Rmax"])
    r = g.xi[0]
    dr = g.dx[0]
    D1 = ExplicitFirst44_1D(dr)
    D2 = ExplicitSecond44_1D(dr)
    g.set_D1(D1)
    g.set_D2(D2)

    eqs = bssn.BSSN(9, g, params["Mass"], params["eta"], params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dr
    rk4 = RK4(eqs, g)

    time = 0.0
    func_names = [ "chi", "grr", "gtt", "Arr", "K", "Gt", "alpha", "beta", "gB" ]
    Nt = params["Nt"]
    for i in range(1,Nt+1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.2f}")


if __name__ == "__main__":
    main()
