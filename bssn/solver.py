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

def get_d1_type(d_str):
    try:
        return d1_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D1 type string: '{d_str}'")

def get_d2_type(d_str):
    try:
        return d2_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D2 type string: '{d_str}'")

def get_cfd_solve(d_str):
    try:
        return cfd_solve_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown Deriv Solve type string: '{d_str}'")


def init_derivative_operators(r, params):
    d1_param = params["D1"]
    # Check if d1 is a string or a list of strings
    if isinstance(d1_param, str):
        d1_list = [d1_param]
    elif isinstance(d1_param, list) and all(isinstance(item, str) for item in d1_param):
        d1_list = d1_param
    else:
        raise TypeError("D1 must be a string or a list of strings")

    d2_param = params["D2"]
    # Check if d2 is a string or a list of strings
    if isinstance(d2_param, str):
        d2_list = [d2_param]
    elif isinstance(d2_param, list) and all(isinstance(item, str) for item in d2_param):
        d2_list = d2_param
    else:
        raise TypeError("D2 must be a string or a list of strings")

    method_str = params.get("DerivSolveMethod", "SCIPY")
    method = get_cfd_solve(method_str)
    mask_bh = params.get("BHMask", False)

    if mask_bh and len(d1_list) != 2:
        raise TypeError("D1 must be a list of two operators for BHMask")

    dr = r[1] - r[0]
    if len(d1_list) >= 1:
        d1type = get_d1_type(d1_list[0])
        if d1type  == DerivType.D1_E44:
            D1 = ExplicitFirst44_1D(dr)
        elif d1type == DerivType.D1_E642:
            D1 = ExplicitFirst642_1D(dr)
        elif d1type in CompactFirstDerivatives:
            D1 = NCompactDerivative.deriv(r, d1type, method)
        else:
            raise NotImplementedError("D1 Type = {d1type}")
        
        d2type = get_d2_type(d2_list[0])
        if d2type == DerivType.D2_E44:
            D2 = ExplicitSecond44_1D(dr)
        elif d2type == DerivType.D2_E642:
            D2 = ExplicitSecond642_1D(dr)
        elif d2type in CompactSecondDerivatives:
            D2 = NCompactDerivative.deriv(r, d2type, method)
        else:
            raise NotImplementedError("D2 Type = {d2type}")

    elif mask_bh:
        rbh = params.get("BHMaskPos", 0.0)
        rbh_width = params.get("BHMaskWidth", 0.1)
        d1_0_type = get_d1_type(d1_list[0])
        d1_bh_type = get_d1_type(d1_list[1])
        if d1_0_type and d1_bh_type in CompactFirstDerivatives:
            D1 = NCompactDerivative.bh_deriv(r, d1_0_type, d1_bh_type, method, rbh, rbh_width)
        else:
            raise TypeError("Invalid FIRST derivative types for masking")

        d2_0_type = get_d2_type(d2_list[0])
        d2_bh_type = get_d2_type(d2_list[1])
        if d2_0_type and d2_bh_type in CompactSecondDerivatives:
            D2 = NCompactDerivative.bh_deriv(r, d2_0_type, d2_bh_type, method, rbh, rbh_width)
        else:
            raise TypeError("Invalid SECOND derivative types for masking")
    else:
        raise RuntimeError("Failed to initialize derivative operators: D1 and D2 are unbound.")

    return D1, D2

def init_filter(r, params):
    f_param = params.get("Filter", "None")

    if isinstance(f_param, str):
        fparam_list = [f_param]
    elif isinstance(f_param, list) and all(isinstance(item, str) for item in f_param):
        fparam_list = f_param
    else:
        raise TypeError("Filter must be a string or a list of strings")

    filters = []
    mask_bh = params.get("BHMask", False)

    for fstr in fparam_list:
        ftype = get_filter_type(fstr)
        if ftype == FilterType.KREISS_OLIGER_O6:
            sigma = params.get("FilterKOsigma", 0.1)
            fbounds = params.get("FilterBoundary", False)
            dr = r[1] - r[0]
            bssn_filter = KreissOligerFilterO6_1D(dr, sigma, filter_boundary=fbounds)
            filters.append(bssn_filter)
        elif ftype == FilterType.KREISS_OLIGER_O8:
            sigma = params.get("FilterKOsigma", 0.1)
            fbounds = params.get("FilterBoundary", False)
            dr = r[1] - r[0]
            bssn_filter = KreissOligerFilterO8_1D(dr, sigma, filter_boundary=fbounds)
            filters.append(bssn_filter)
        elif ftype in CompactFilterTypes:
            fapply = get_filter_apply(params.get("FilterApply", FilterApply.APPLY_VARS))
            fmethod = CFDSolve.SCIPY
            fbounds = params.get("FilterBoundary", False)
            ffreq = params.get("FilterFrequency", 1)
            alpha = params.get("FilterAlpha", 0.0)
            beta = params.get("FilterBeta", 0.0)
            if mask_bh:
                mask_pos = params.get("BHMaskPos", 0.0)
                mask_width = params.get("BHMaskWidth", 0.1)
                bssn_filter = NCompactFilter.init_bh_filter(r, ftype, fapply, fmethod, ffreq, mask_pos, mask_width, fbounds, alpha, beta)
                filters.append(bssn_filter)
            else:
                bssn_filter = NCompactFilter.init_filter(r, ftype, fapply, fmethod, ffreq, fbounds, alpha, beta)
                filters.append(bssn_filter)
        elif ftype == FilterType.NONE:
            pass
        else:
            raise NotImplementedError("Filter = { KO6, KO8, JTT6, JTP6, JTT8, JTP8, KP4 }")
    return filters


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

    D1, D2 = init_derivative_operators(r, params)
    g.set_D1(D1)
    g.set_D2(D2)

    F1 = init_filter(r, params)
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
