import numpy as np
import sys
import os
import csv
import argparse

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nwave import *


def func(x):
    k1 = 3.0
    k2 = 1.0
    f = np.cos(x) + np.sin(k1 * x)
    df = k1 * np.cos(k1 * x) - np.sin(x)
    ddf = -np.cos(x) - k1**2 * np.sin(k1 * x)
    return f, df, ddf


def func2(x):
    """Simpler test for testing 'symmetry' to make sure x and y give the same values"""
    freq = (1.0 / 2.0) * np.pi
    f = np.cos(freq * x)

    # first order analytical derivative
    df = -freq * np.sin(freq * x)

    # second order analytical derivative
    ddf = -freq * freq * np.cos(freq * x)

    return f, df, ddf

def func3(x):
    k = 8.0
    f = np.exp(-x**2) * np.cos(k * x)
    df = -2 * x * np.exp(-x**2) * np.cos(k * x) - k * np.exp(-x**2) * np.sin(k * x)
    ddf = (4 * x**2 - 2 - 4 * k * x) * np.exp(-x**2) * np.cos(k * x) + \
          (4 * k * x - 2 * k**2) * np.exp(-x**2) * np.sin(k * x)
    return f, df, ddf

def write_to_csv(filename, x, f, df_exact, ddf_exact, df_numeric, ddf_numeric):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "f", "df_exact", "df_numeric", "df_error",
                         "ddf_exact", "ddf_numeric", "ddf_error"])
        for xi, fi, dfa, dfn, ddfa, ddfn in zip(x, f, df_exact, df_numeric, ddf_exact, ddf_numeric):
            writer.writerow([xi, fi, dfa, dfn, abs(dfa - dfn), ddfa, ddfn, abs(ddfa - ddfn)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--xmin", type=float, default=0.0, help="xmin", metavar="DBL")
    parser.add_argument("-b", "--xmax", type=float, default=1.0, help="xmin", metavar="DBL")
    parser.add_argument("-l", "--level", type=int, default=0, help="Level", metavar="INT")
    parser.add_argument("--D1", type=str, default="E4", help="First Derivative (E4, E6, JTP6, AE4, AE6)", metavar="STR")
    parser.add_argument("--D2", type=str, default="E4", help="Second Derivative", metavar="STR")
    parser.add_argument('n0', type=int, help="Nx0", metavar="NX")

    args = parser.parse_args()

    n0 = args.n0
    if n0 % 2 == 1:
        n0 -= 1
    level = args.level
    nx = (2**level) * n0 + 1
 

    params = { 
        "Nx": nx,
        "Xmin": args.xmin,
        "Xmax": args.xmax,
    }

    g = Grid1D(params)
    dx = g.dx[0]
    x = g.xi[0]

    if args.D1 == "E4" or args.D1 == "AE4":
        D1 = ExplicitFirst44_1D(dx)
    elif args.D1 == "E6" or args.D1 == "AE6":
        D1 = ExplicitFirst642_1D(dx)
    elif args.D1 == "JTP6":
        D1 = CompactFirst1D(x, DerivType.D1_JTP6, CFDSolve.LUSOLVE)
    else:
        raise NotImplementedError

    if args.D2 == "E4":
        D2 = ExplicitSecond44_1D(dx)
    elif args.D2 == "E6":
        D2 = ExplicitSecond642_1D(dx)
    elif args.D2 == "JTP6":
        D2 = CompactSecond1D(x, DerivType.D2_JTP6, CFDSolve.LUSOLVE)
    else:
        raise NotImplementedError


    f, df_exact, ddf_exact = func3(x)

    if args.D1 == "AE4" or args.D1 == "AE6":
        # use func as the vel
        beta, db, ddb = func(x)
        df_numeric = D1.advec_grad(f, beta)
    else:
        df_numeric = D1.grad(f)


    ddf_numeric = D2.grad2(f)

    fname = f"ctest_{level}.csv"
    write_to_csv(fname, x, f, df_exact, ddf_exact, df_numeric, ddf_numeric)


if __name__ == "__main__":
    main()

