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


def verify_params(params):
    if params["Nx"] < 10:
        raise ValueError("Nx must be at least 10")
    if params["NGhost"] < 2:
        raise ValueError("NGhost must be at least 2")
    if params["NGhost"] > 4:
        raise ValueError("NGhost must be at most 4")
    if params["cfl"] <= 0.0:
        raise ValueError("cfl must be positive")
    if params["cfl"] >= 1.0:
        raise ValueError("cfl must be less than 1.0")
    if params["Nt"] <= 0:
        raise ValueError("Nt must be positive")
    if params["output_interval"] <= 0:
        raise ValueError("output_interval must be positive")
    if params["print_interval"] <= 0:
        raise ValueError("print_interval must be positive")
    if params["output_interval"] < params["print_interval"]:
        raise ValueError("output_interval must be greater than print_interval")
    if params["output_dir"] == "":
        raise ValueError("output_dir must be a non-empty string")
    if params["D1"] not in ["E4", "E6", "JP6", "KP4"]:
        raise ValueError("D1 must be one of {E4, E6, JP6, KP4}")
    if params["D2"] not in ["E4", "E6", "JP6"]:
        raise ValueError("D2 must be one of {E4, E6, JP6}")
    if params["Mass"] <= 0.0:
        raise ValueError("Mass must be positive")
    if params["eta"] < 1.0:
        raise ValueError("eta must be greater than or equal to 1.0")


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


def write_curve2(filename, time, x, eqs):
    nghosts = 4
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(eqs.u_names)):
            f.write(f"# {eqs.u_names[m]}\n")
            rr = np.log(x[nghosts:-nghosts])
            ff = eqs.u[m][nghosts:-nghosts]
            for xi, di in zip(rr, ff):
                f.write(f"{xi:.8e} {di:.8e}\n")
        for m in range(len(eqs.c_names)):
            f.write(f"# {eqs.c_names[m]}\n")
            rr = np.log(x[nghosts:-nghosts])
            ff = eqs.C[m][nghosts:-nghosts]
            for xi, di in zip(rr, ff):
                f.write(f"{xi:.8e} {di:.8e}\n")

def write_curve_file(filename, uname, time, x, u):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        f.write(f"# {uname}\n")
        for xi, ui in zip(x, u):
            f.write(f"{xi:.8e} {ui:.8e}\n")


def get_filter_type(filter_str):
    try:
        return filter_type_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter type string: '{filter_str}'")

def get_filter_apply(filter_str):
    try:
        return filter_apply_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter apply string: '{filter_str}'")

def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Verify parameters
    verify_params(params)

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
    if params["D1"] == "E4":
        D1 = ExplicitFirst44_1D(dr)
        g.set_D1(D1)
    elif params["D1"] == "E6":
        D1 = ExplicitFirst642_1D(dr)
        g.set_D1(D1)
    elif params["D1"] == "JP6":
        D1 = CompactFirst1D(r, DerivType.D1_JP6, CFDSolve.LUSOLVE)
        #D1 = CompactFirst1D(r, DerivType.D1_JP6, CFDSolve.D_LU)
        g.set_D1(D1)
    elif params["D1"] == "KP4":
        D1 = CompactFirst1D(r, DerivType.D1_KP4, CFDSolve.LUSOLVE)
        g.set_D1(D1)
    else:
        raise NotImplementedError("D1 = { E4, E6, JP6, KP4 }")

    if params["D2"] == "E4":
        D2 = ExplicitSecond44_1D(dr)
        g.set_D2(D2)
    elif params["D2"] == "E6":
        D2 = ExplicitSecond642_1D(dr)
        g.set_D2(D2)
    elif params["D2"] == "JP6":
        D2 = CompactSecond1D(r, DerivType.D2_JP6, CFDSolve.LUSOLVE)
        #D2 = CompactSecond1D(r, DerivType.D2_JP6, CFDSolve.D_LU)
        g.set_D2(D2)
    else:
        raise NotImplementedError("D2 = { E4, E6, JP6 }")

    filter_str = params.get("Filter", "None")
    ftype = get_filter_type(filter_str)
    if ftype == FilterType.KREISS_OLIGER_O6:
        sigma = params.get("FilterKOsigma", 0.1)
        fbounds = params.get("FilterBoundary", False)
        bssn_filter = KreissOligerFilterO6_1D(dr, sigma, filter_boundary=fbounds)
        g.set_filter(bssn_filter)
    elif ftype == FilterType.KREISS_OLIGER_O8:
        sigma = params.get("FilterKOsigma", 0.1)
        fbounds = params.get("FilterBoundary", False)
        bssn_filter = KreissOligerFilterO8_1D(dr, sigma, filter_boundary=fbounds)
        g.set_filter(bssn_filter)
    elif ftype in CompactFilterTypes:
        fapply = get_filter_apply(params.get("FilterApply", FilterApply.APPLY_VARS))
        fmethod = CFDSolve.SCIPY
        fbounds = params.get("FilterBoundary", False)
        ffreq = params.get("FilterFrequency", 1)
        alpha = params.get("FilterAlpha", 0.0)
        beta = params.get("FilterBeta", 0.0)
        bssn_filter = NCompactFilter.init_filter(r, ftype, fapply, fmethod, ffreq, fbounds, alpha, beta)
        g.set_filter(bssn_filter)
    elif ftype == FilterType.NONE:
        pass
    else:
        raise NotImplementedError("Filter = { KO6, KO8, JTT6, JTP6, JTT8, JTP8, KP4 }")

    if g.num_filters > 0:
        print(f"Filter type: {g.Filter[0].get_filter_type()}")
        print(f"Filter apply: {g.Filter[0].get_apply_filter()}")

    # GBSSN system: (sys, lapse advection, shift advection)
    #    sys = 0 (Eulerian), 1 (Lagrangian)
    sys = bssn.GBSSNSystem(1, 1, 1)
    eqs = bssn.BSSN(
        g, params["Mass"], params["eta"], extended_domain, BCType.FUNCTION, sys, have_d2=True
    )
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
    rbh_guess = 0.7
    rbh, mbh, rTheta = eqs.find_horizon(g, rbh_guess)
    if rbh == None:
        rbh  = 0.0
        mbh  = 0.0
    eqs.set_ah(rbh, mbh)
    print(f"Horizon(s) found at {rbh:03f} with masses {mbh:03f}")

    step = 0
    fname = f"{output_dir}/bssn_{step:04d}.curve"
    write_curve(fname, 0.0, g.xi[0], eqs)
    fname = f"{output_dir}/rtheta_{step:04d}.curve"
    write_curve_file(fname, "rTheta", time, g.xi[0], rTheta)

    # Create a CSV file for constraint norms
    conname = f"{output_dir}/bssn_constraints.dat"
    confile = open(conname, mode="w", newline="")
    writer = csv.writer(confile)
    writer.writerow(["time", "Ham", "Mom", "radius_bh", "mass_bh"])
    writer.writerow(
        [time, l2norm(eqs.C[0][nghost:-nghost]), l2norm(eqs.C[1][nghost:-nghost]), rbh, mbh]
    )

    filvar = None
    filter_frequency = -1
    if g.num_filters > 0:
        for fx in g.Filter:
            if fx.get_apply_filter == FilterApply.APPLY_VARS:
                filvar = fx
                filter_frequency = fx.get_frequency()

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
                rbh  = 0.0
                mbh  = 0.0
            eqs.set_ah(rbh, mbh)
            print(f"Horizon(s) found at {rbh:.03f} with masses {mbh:03f}.")
            fname = f"{output_dir}/rtheta_{i:04d}.curve"
            write_curve_file(fname, "rTheta", time, g.xi[0], rTheta)
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
            write_curve(fname, time, g.xi[0], eqs)
        if np.isnan(eqs.u[1]).any():
            print("Solution has a NaN.  Bye.")
            break

    confile.close()


if __name__ == "__main__":
    parfile = sys.argv[1]
    main(parfile)
