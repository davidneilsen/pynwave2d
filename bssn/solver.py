import numpy as np
import sys
import os
#from pdb import set_trace as bp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bssneqs as bssn
import json
from nwave import (
    Grid1D,
    RK4,
    Equations,
    ExplicitFirst44_1D,
    ExplicitSecond44_1D,
    CompactDerivative,
    l2norm,
)


def write_curve(filename, time, x, eqs):
    constraint_names = ["Ham", "Mom"]
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(eqs.u_names)):
            f.write(f"# {eqs.u_names[m]}\n")
            for xi, di in zip(x, eqs.u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")
        for m in range(len(eqs.c_names)):
            f.write(f"# {eqs.c_names[m]}\n")
            for xi, di in zip(x, eqs.C[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")


def main():
    # Read parameters
    with open("params.json") as f:
        params = json.load(f)

    g = Grid1D(params, cell_centered=True)
    r = g.xi[0]
    dr = g.dx[0]
    D1 = ExplicitFirst44_1D(dr)
    D2 = ExplicitSecond44_1D(dr)
    # D1 = CompactDerivative(r, "D1_JTP6", method="LUSOLVE")
    # D2 = CompactDerivative(r, "D2_JTP6", method="LUSOLVE")
    g.set_D1(D1)
    g.set_D2(D2)

    # GBSSN system: (sys, lapse advection, shift advection)
    #    sys = 0 (Eulerian), 1 (Lagrangian)
    sys = bssn.GBSSNSystem(1, 1, 1)
    eqs = bssn.BSSN(g, params["Mass"], params["eta"], "FUNCTION", sys)
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    print_interval = params["print_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dr
    rk4 = RK4(eqs, g)

    time = 0.0
    Nt = params["Nt"]

    eqs.cal_constraints(eqs.u, g)
    step = 0
    fname = f"{output_dir}/bssn_{step:04d}.curve"
    write_curve(fname, 0.0, g.xi[0], eqs)

    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        time += dt
        if i % print_interval == 0 or i % output_interval == 0:
            eqs.cal_constraints(eqs.u, g)
        if i % print_interval == 0:
            print(
                f"Step {i:d}, t={time:.2e}, |chi|={l2norm(eqs.u[0]):.2e}, |grr|={l2norm(eqs.u[1]):.2e}, |gtt|={l2norm(eqs.u[2]):.2e}, |Arr|={l2norm(eqs.u[3]):.2e}, |K|={l2norm(eqs.u[4]):.2e}, |Gt|={l2norm(eqs.u[5]):.2e}, |alpha|={l2norm(eqs.u[6]):.2e}, |beta|={l2norm(eqs.u[7]):.2e}, |gB|={l2norm(eqs.u[8]):.2e}, |Ham|={l2norm(eqs.C[0]):.2e}, |Mom|={l2norm(eqs.C[1]):.2e}"
            )
        if i % output_interval == 0:
            fname = f"{output_dir}/bssn_{i:04d}.curve"
            write_curve(fname, time, g.xi[0], eqs)


if __name__ == "__main__":
    main()
