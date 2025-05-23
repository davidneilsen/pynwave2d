import numpy as np
import sys
import os
import csv
#from pdb import set_trace as bp

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
    if params["Filter"] not in ["KO6", "KO8"]:
        raise ValueError("D2 must be one of {KO6, KO8}")
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


def main():
    # Read parameters
    with open("params.toml", "rb") as f:
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
        D1 = CompactFirst1D(r, "D1_JTP6", method="LUSOLVE")
        g.set_D1(D1)
    elif params["D1"] == "KP4":
        D1 = CompactFirst1D(r, "D1_KP4", method="LUSOLVE")
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
        D2 = CompactSecond1D(r, "D2_JTP6", method="LUSOLVE")
        g.set_D2(D2)
    else:
        raise NotImplementedError("D2 = { E4, E6, JP6 }")
    
    if "Filter" in params:
        sigma = params.get("KOsigma", 0.1)
        apply_diss_boundaries = params.get("ApplyDissBounds", False)
        if params["Filter"] == "KO6":
            bssn_filter = KreissOligerFilterO6_1D( dr, sigma, apply_diss_boundaries)
            g.set_filter(bssn_filter)
        elif params["Filter"] == "KO8":
            bssn_filter = KreissOligerFilterO8_1D( dr, sigma, apply_diss_boundaries=True)
            g.set_filter(bssn_filter)
        else:
            raise NotImplementedError("Filter = { KO6, KO8 }")

        print(f"Filter type: {g.Filter.get_filter_type()}")
        print(f"Filter apply: {g.Filter.get_apply_filter()}")
        print(f"Filter sigma: {bssn_filter.get_sigma()}")

    # GBSSN system: (sys, lapse advection, shift advection)
    #    sys = 0 (Eulerian), 1 (Lagrangian)
    sys = bssn.GBSSNSystem(1, 1, 1)
    eqs = bssn.BSSN(g, params["Mass"], params["eta"], extended_domain, "FUNCTION", sys)
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

    # Create a CSV file for constraint norms
    conname = f"{output_dir}/bssn_constraints.dat"
    confile = open(conname, mode='w', newline='')
    writer = csv.writer(confile)
    writer.writerow(["time", "Ham", "Mom"])
    writer.writerow([time, l2norm(eqs.C[0][nghost:-nghost]), l2norm(eqs.C[1][nghost:-nghost])])

    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        time += dt
        if i % print_interval == 0 or i % output_interval == 0:
            eqs.cal_constraints(eqs.u, g)
        if i % print_interval == 0:
            hamnorm = l2norm(eqs.C[0][nghost:-nghost])
            momnorm = l2norm(eqs.C[1][nghost:-nghost])
            print(
                f"Step {i:d}, t={time:.2e}, |chi|={l2norm(eqs.u[0]):.2e}, |grr|={l2norm(eqs.u[1]):.2e}, |gtt|={l2norm(eqs.u[2]):.2e}, |Arr|={l2norm(eqs.u[3]):.2e}, |K|={l2norm(eqs.u[4]):.2e}, |Gt|={l2norm(eqs.u[5]):.2e}, |alpha|={l2norm(eqs.u[6]):.2e}, |beta|={l2norm(eqs.u[7]):.2e}, |gB|={l2norm(eqs.u[8]):.2e}, |Ham|={hamnorm:.2e}, |Mom|={momnorm:.2e}"
            )
            writer.writerow([time, hamnorm, momnorm])
        if i % output_interval == 0:
            fname = f"{output_dir}/bssn_{i:04d}.curve"
            write_curve(fname, time, g.xi[0], eqs)

    confile.close()

if __name__ == "__main__":
    main()
