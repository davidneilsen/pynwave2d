import numpy as np
import sys
import os
import tomllib

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maxwell import Maxwell2D, cal_constraints
import mutils
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
    D1, D2 = mutils.init_derivative_operators(x, y, params)
    g.set_D1(D1)
    g.set_D2(D2)

    if params["id_type"] == "waveguide":
        bound_cond = "WAVEGUIDE"
    elif params["id_type"] == "gaussian":
        bound_cond = "SOMMERFELD"
    else:
        raise ValueError(
            "Invalid initial condition type. Use 'waveguide' or 'gaussian'."
        )

    time = 0.0
    eqs = Maxwell2D(3, g, bound_cond)
    eqs.initialize(g, params)
    divE = np.zeros(g.shp)
    uexact = [np.zeros(g.shp) for _ in range(eqs.Nu)]
    uerr = [np.zeros(g.shp) for _ in range(eqs.Nu)]
    cal_constraints(eqs.u, divE, uexact, uerr, g, time, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dx
    rk4 = RK4(eqs, g)

    func_names = eqs.u_names + [
        "divE",
        "Ex_exact",
        "Ey_exact",
        "Hz_exact",
        "Ex_error",
        "Ey_error",
        "Hz_error",
    ]
    allfuncs = eqs.u + [divE] + uexact + uerr
    iox.write_hdf5(0, allfuncs, x, y, func_names, output_dir)
    # write_vtk_rectilinear_grid(0, allfuncs, x, y, func_names, time, output_dir)

    Nt = params["Nt"]
    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        time += dt
        print(f"Step {i:d}  t={time:.4f}")
        if i % output_interval == 0:
            cal_constraints(eqs.u, divE, uexact, uerr, g, time, params)
            iox.write_hdf5(i, allfuncs, x, y, func_names, output_dir)
            # write_vtk_rectilinear_grid(i, allfuncs, x, y, func_names, time, output_dir)

    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
