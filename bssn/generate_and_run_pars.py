#!/usr/bin/env python3
"""
generate_and_run_pars.py

Creates par files by replacing uppercase placeholders in a base template and runs:
    python solver.py <parfile>

Edit the lists in the CONFIG section below to control runs.
"""

import os
import argparse
import itertools
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -----------------------
# CONFIG: edit these lists
# -----------------------
BASE_PAR = """Nt = 12500
Nx = 3000
Xmin = -120.0
Xmax =  120.0
level = LEVEL

NGhost = 3
cfl = 0.10
D1 = D1LIST
D2 = D2LIST

BHMask = BHMASK
BHMaskPos = 0.0
BHMaskWidth = 0.15

# bssn outer bc types = [ "Sommerfeld", "Sponge" ]
bssn_outer_bc = OUTERBC

# initial data types = [ "Puncture", "EddingtonFinkelstein" ]
Mass = 1.0
collapsed_lapse = COLLAPSEDLAPSE
initial_data = "Puncture"

# output parameters
output_dir = OUTPUTDIR
output_interval = 50
print_interval = 5
bhmask_outer_radius = 50.0

[lapse]
type = "1+log"
k = 2.0

[shift_damping]
type = "constant"
eta = 2.0

# dissipation filter (interior-only; boundary terms off)
[[filters]]
type = "KO8"
apply = "Rhs"
sigma = 0.4
filter_boundary = false
"""

# Provide lists for each placeholder. Values may be strings or booleans.
# LEVEL: could be strings like "01" or integers like 1
LEVELS = [0, 1, 2, 3]

# D1/D2: if strings, they'll be quoted in the par file automatically.
D1SING_LIST= [ "E4", "E6", "JP6", "C4_1", "C6_4", "A4_3" ]
D2SING_LIST= [ "E4", "E6", "JP6", "2C4_1", "2C6_1", "2A_1" ]
D1MIX_LIST = [ "C4_1/ME4", "C6_4/ME4", "C6_4/ME6", "A4_3/ME6" ]
D2MIX_LIST = [ "2C4_1/ME4", "C6_4/ME4", "2C6_1/ME6", "2A_1/ME6" ]  # If empty, it will copy D1_LIST
D1_LIST = D1SING_LIST + D1MIX_LIST
D2_LIST = D2SING_LIST + D2MIX_LIST

if len(D2_LIST) == 0:
    D2_LIST = D1_LIST.deepcopy()

if len(D2_LIST) != len(D1_LIST):
    raise ValueError(f"D2_LIST needs to have same length as D1_LIST")

# OUTER_BC_LIST: Either "Sommerfeld" or "Sponge"
OUTERBC_LIST = ["Sommerfeld", "Sponge"]

# COLLAPSEDLAPSE: booleans or strings 'true'/'false'
COLLAPSED_LAPSE_LIST = [True]

# OUTPUT basename (we will append D1/D2/LEVEL etc.)
OUTPUT_BASENAME = "srun"

# Additional options
PAR_OUTPUT_DIR = Path("generated_pars")  # directory to write par files

# -----------------------
# End CONFIG
# -----------------------

def fmt_val_for_par(val):
    """
    Convert Python value into the textual form to put into the par file.
    - Strings: if not already quoted, quote them.
    - Booleans: lower-case 'true'/'false' (no quotes).
    - Numbers: str(value)
    """
    print(f"val is {val}")
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        s = val.strip()
        # if looks already quoted, leave as-is
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            if "/" in val:
                left, right = val.split("/",1)
                s = f'["{left}","{right}"]'
            print(f"returning {s}")
            return s
        # if it is a numeric-like string, return as-is (no quotes) for numeric fields
        # but for D1/D2 we prefer quoted string; decision below is caller's responsibility.
        if "/" in val:
            left, right = val.split("/",1)
            s = f'["{left}", "{right}"]'
            return s
        else:
            return f'"{s}"'
    # fallback
    return str(val)

def make_output_dirname(basename, level, d1, d2, collapsed, outerbc):
    # create a sensible directory name: basename_LEVEL_D1_D2_collapsedTrue/False
    lvl = str(level)
    d1s = str(d1).replace("/","_Mix_")
    d2s = str(d2).replace("/","_Mix_")
    print(f'd1 = {d1}, and d1s = {d1s}')
    coll = "collapsed" if collapsed else "normal"
    obc = "rad" if outerbc == "Sommerfeld" else "sponge"
    name = f"{basename}_L{lvl}_D1{d1s}_D2{d2s}_{coll}_B{obc}"
    # sanitize spaces or quotes
    return name.replace(" ", "_").replace('"', '').replace("'", "")

def build_par_text(template, replacements):
    """Replace placeholders exactly as in template. Expects replacements dict mapping token->string."""
    out = template
    for token, text in replacements.items():
        out = out.replace(token, text)
    return out

def write_par_file(text, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text)
    return target_path

def run_solver(parfile_path, dry_run=False, verbose=True):
    cmd = ["python", "solver.py", str(parfile_path)]
    if dry_run:
        if verbose:
            print("[DRY] Would run:", " ".join(shlex.quote(x) for x in cmd))
        return 0
    if verbose:
        print("Running:", " ".join(shlex.quote(x) for x in cmd))
    # run and return exit code
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except FileNotFoundError as e:
        print(f"Error: failed to start solver (FileNotFound): {e}")
        return 127
    except Exception as e:
        print(f"Error running solver: {e}")
        return 1

def generate_and_run(levels, d1_list, d2_list, collapsed_list,
                     outerbc_list,
                     basename, outdir: Path,
                     jobs=1, dry_run=False, verbose=True):
    # prepare cross product
    dindex = list(range(len(d1_list)))
    combos = list(itertools.product(levels, dindex, collapsed_list, outerbc_list))
    if verbose:
        print(f"Preparing {len(combos)} runs...")

    # prepare output dir
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for (level, dindex, collapsed, outerbc) in combos:
        # format replacements for par file
        # LEVEL: keep as the literal token the user expected. If user supplied '01' we keep '01' as text.
        level_text = str(level)
        # D1/D2 -> quoted strings in the par file
        d1 = d1_list[dindex]
        d2 = d2_list[dindex]
        BHMask = False
        if "/" in d1 or "/" in d2:
            BHMask = True
        
        d1_text = fmt_val_for_par(d1)
        d2_text = fmt_val_for_par(d2)
        collapsed_text = fmt_val_for_par(collapsed)
        bhmask_text = fmt_val_for_par(BHMask)
        outerbc_text = fmt_val_for_par(outerbc)

        # OUTPUTDIR: follow user's pattern: basename + D1LIST -> we'll produce a path string in quotes
        outdirname = make_output_dirname(basename, level_text, d1, d2, collapsed, outerbc)
        outputdir_text = f'"{outdirname}"'  # quoted string in par file

        replacements = {
            "LEVEL": level_text,
            "D1LIST": d1_text,
            "D2LIST": d2_text,
            "COLLAPSEDLAPSE": collapsed_text,
            "OUTERBC": outerbc_text,
            "BHMASK": bhmask_text,
            "OUTPUTDIR": outputdir_text,
        }

        par_text = build_par_text(BASE_PAR, replacements)

        # parfile name e.g. par_L01_D1E4_D2A1_collapsedTrue.par
        par_filename = f"par_L{level_text}_D1{str(d1).replace(' ', '')}_D2{str(d2).replace(' ', '')}_c{str(collapsed)}_b{str(outerbc)}.par"
        par_filename = par_filename.replace("/", "_Mix_")
        par_filename = par_filename.replace("Sommerfeld", "Rad")
        print(f'par_filename = {par_filename}')
        par_path = outdir / par_filename
        write_par_file(par_text, par_path)

        tasks.append(par_path)

    # run tasks (sequential or parallel)
    results = {}
    if jobs <= 1:
        for p in tasks:
            rc = run_solver(p, dry_run=dry_run, verbose=verbose)
            results[str(p)] = rc
    else:
        # Threaded parallelism (subprocesses are used to run solver)
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            future_to_path = {ex.submit(run_solver, p, dry_run, verbose): p for p in tasks}
            for fut in as_completed(future_to_path):
                p = future_to_path[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    rc = -1
                    print(f"Task for {p} raised: {e}")
                results[str(p)] = rc

    if verbose:
        print("\nSummary (parfile -> exitcode):")
        for p, rc in results.items():
            print(f"  {p} -> {rc}")
    return results

def parse_args():
    ap = argparse.ArgumentParser(description="Generate par files and run solver.py on each.")
    ap.add_argument("--jobs", "-j", type=int, default=1, help="Number of parallel jobs (default 1).")
    ap.add_argument("--outdir", "-o", default=str(PAR_OUTPUT_DIR), help="Directory to write generated par files.")
    ap.add_argument("--dry-run", action="store_true", help="Only write par files and print commands; don't execute solver.")
    ap.add_argument("--no-run", action="store_true", help="Alias for --dry-run (keeps backward-compatibility).")
    ap.add_argument("--quiet", action="store_true", help="Less verbose output.")
    return ap.parse_args()

def main():
    args = parse_args()
    jobs = max(1, args.jobs)
    dry_run = args.dry_run or args.no_run
    verbose = not args.quiet

    # Use the CONFIG values defined at top of file.
    res = generate_and_run(
        levels=LEVELS,
        d1_list=D1_LIST,
        d2_list=D2_LIST,
        collapsed_list=COLLAPSED_LAPSE_LIST,
        outerbc_list=OUTERBC_LIST,
        basename=OUTPUT_BASENAME,
        outdir=Path(args.outdir),
        jobs=jobs,
        dry_run=dry_run,
        verbose=verbose,
    )
    # exit status: non-zero if any run failed
    any_failed = any(rc != 0 for rc in res.values())
    if any_failed:
        print("One or more runs returned non-zero exit codes.")
        raise SystemExit(1)
    print("All runs completed (exit codes reported above).")

if __name__ == "__main__":
    main()

