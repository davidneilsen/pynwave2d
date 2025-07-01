import numpy as np
import sys
import os
import csv

# from pdb import set_trace as bp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bssneqs as bssn
import bssnutils as butil
import tomllib
from nwave import *


def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Verify parameters
    butil.verify_params(params)

    # Set up grid
    if "Xmin" in params and params["Xmin"] < 0.0:
        extended_domain = True
        cellgrid = False
        if params["Nx"] % 2 != 0:
            params["Nx"] += 1
            print(f"Adjusted Nx to {params['Nx']} (must be even)")
    else:
        extended_domain = False
        cellgrid = True

    nghost = params["NGhost"]
    g = Grid1D(params, cell_centered=cellgrid)

    r = g.xi[0]
    dr = g.dx[0]
    Nr = len(r)

    D1, D2 = butil.init_derivative_operators(r, params)
    g.set_D1(D1)
    g.set_D2(D2)
    print(f"D1 type: {g.D1.get_type()}")
    print(f"D2 type: {g.D2.get_type()}")

    F1 = butil.init_filter(r, params)
    g.set_filter(F1)

    if g.num_filters > 0:
        for f in g.Filter:
            print(f"f type = {type(f)}")
            print(f"Filter type: {f.get_filter_type()}")
            print(f"Filter apply: {f.get_apply_filter()}")

    # GBSSN system: (sys, lapse advection, shift advection)
    #    sys = 0 (Eulerian), 1 (Lagrangian)
    sys = bssn.GBSSNSystem(1, 1, 1)
    eqs = bssn.BSSN(
        g,
        params["Mass"],
        params["eta"],
        extended_domain,
        BCType.FUNCTION,
        sys,
        have_d2=True,
    )
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    print_interval = params["print_interval"]

    os.makedirs(output_dir, exist_ok=True)

    Nt = params["Nt"]
    time = 0.0
    dt = params["cfl"] * dr
    rk4 = RK4(eqs, g)

    eqs.cal_constraints(eqs.u, g)
    rbh_guess = 0.7
    rbh, mbh, rTheta = eqs.find_horizon(g, rbh_guess)
    if rbh == None:
        rbh = 0.0
        mbh = 0.0
    eqs.set_ah(rbh, mbh)
    bhpts = int(rbh * len(r) / (r[-1] - r[0]))
    print(f"Horizon(s) found at {rbh:03f} with masses {mbh:03f} and bhpts={bhpts:d}")

    step = 0
    fname = f"{output_dir}/bssn_{step:04d}.curve"
    butil.write_curve(fname, 0.0, g.xi[0], eqs)

    fname = f"{output_dir}/rtheta_{step:04d}.curve"
    butil.write_curve_functions(fname, ["rTheta"], time, g.xi[0], [rTheta])

    # Create a CSV file for constraint norms
    conname = f"{output_dir}/bssn_constraints.dat"
    confile = open(conname, mode="w", newline="")
    writer = csv.writer(confile)
    writer.writerow(["time", "Ham", "Mom", "radius_bh", "mass_bh"])
    writer.writerow(
        [
            time,
            l2norm(eqs.C[0][nghost:-nghost]),
            l2norm(eqs.C[1][nghost:-nghost]),
            rbh,
            mbh,
        ]
    )

    filvar = None
    filter_frequency = -1
    if g.num_filters > 0:
        for fx in g.Filter:
            if fx.get_apply_filter == FilterApply.APPLY_VARS:
                filvar = fx
                filter_frequency = fx.get_frequency()
                break

    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        if filter_frequency > 0 and i % filter_frequency == 0:
            # Apply filter to the variables
            print("Applying filter to variables")
            for j in range(eqs.Nu):
                eqs.u[j][:] = filvar.filter(eqs.u[j])

        time += dt
        if i % print_interval == 0 or i % output_interval == 0:
            eqs.cal_constraints(eqs.u, g)
            rbh_guess = rbh
            rbh, mbh, rTheta = eqs.find_horizon(g, rbh_guess)
            if rbh == None:
                rbh = 0.0
                mbh = 0.0
            eqs.set_ah(rbh, mbh)
            bhpts = int(rbh * len(r) / (r[-1] - r[0]))
            print(
                f"Horizon(s) found at {rbh:.03f} with masses {mbh:03f} and bhpts={bhpts:d}."
            )
        if i % print_interval == 0:
            hamnorm = l2norm(eqs.C[0][nghost:-nghost])
            momnorm = l2norm(eqs.C[1][nghost:-nghost])
            print(
                f"Step {i:d}, t={time:.2e}, |chi|={l2norm(eqs.u[0]):.2e}, |grr|={l2norm(eqs.u[1]):.2e}, |gtt|={l2norm(eqs.u[2]):.2e}, |Arr|={l2norm(eqs.u[3]):.2e}, |K|={l2norm(eqs.u[4]):.2e}, |Gt|={l2norm(eqs.u[5]):.2e}, |alpha|={l2norm(eqs.u[6]):.2e}, |beta|={l2norm(eqs.u[7]):.2e}, |gB|={l2norm(eqs.u[8]):.2e}, |Ham|={hamnorm:.2e}, |Mom|={momnorm:.2e}"
            )
            rbh, mbh = eqs.get_ah()
            writer.writerow([time, hamnorm, momnorm, rbh, mbh])
        if i % output_interval == 0:
            fname = f"{output_dir}/bssn_{i:04d}.curve"
            butil.write_curve(fname, time, g.xi[0], eqs)
            fname = f"{output_dir}/rtheta_{i:04d}.curve"
            butil.write_curve_functions(fname, ["rTheta"], time, g.xi[0], [rTheta])

        if np.isnan(eqs.u[1]).any():
            print("Solution has a NaN.  Bye.")
            break

    confile.close()


if __name__ == "__main__":
    parfile = sys.argv[1]
    main(parfile)
