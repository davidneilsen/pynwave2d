#!/usr/bin/env python3

"""
Example commandline: python plotlyfiles.py --xkey="time" --ykey="Ham" --semi --show --files srun_L3_D1E4_D2E4_collapsed_Brad_3/bssn_constraints.dat srun_L3_D1E6_D2E6_collapsed_Brad_3/bssn_constraints.dat srun_L3_D1C4_1_D22C4_1_collapsed_Brad_3/bssn_constraints.dat srun_L3_D1C6_4_D22C6_1_collapsed_Brad_3/bssn_constraints.dat --labels "E4" "E6" "C4" "C6"
"""

"""
plot_publication.py

Usage examples:

# Single file, single y column, show on screen:
python plot_publication.py --files data1.csv --labels "Experiment A" --xkey time --ykey value --show

# Multiple files, each with same x and y columns, save PDF:
python plot_publication.py --files a.csv b.csv --labels "A" "B" --xkey time --ykey value --pdf compare.pdf

# Multiple y columns (comma-separated) from a single file:
python plot_publication.py --files data.csv --labels "Dataset" --xkey time --ykey "v1,v2,v3" --show

# Per-file different y columns: provide the same number of --ykey entries as files
python plot_publication.py --files a.csv b.csv --labels "A" "B" --xkey time --ykey "y1,y2" "y3" --show

Notes:
- Requires plotly and pandas. For PDF export also install: pip install -U kaleido
- If you request PDF and kaleido isn't present the script will print an explanatory error.
"""

import argparse
import sys
import os
from typing import List
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Publication-style defaults
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 600
#DEFAULT_FONT = dict(family="Serif, Times New Roman, DejaVu Serif, Georgia", size=12, color="#000")
DEFAULT_FONT = dict(family="Arial, Helvetica, DejaVu Sans, sans-serif", size=20, color="#000")
LINE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Prepare a publication-quality Plotly plot from CSV files.")
    ap.add_argument("--files", "-f", required=True, nargs="+",
                    help="CSV file(s) to plot.")
    ap.add_argument("--labels", "-l", nargs="+",
                    help="Label(s) corresponding to the files. If omitted, filenames are used.")
    ap.add_argument("--xkey", required=True,
                    help="Column name in CSV to use for x axis.")
    ap.add_argument("--ykey", "-y", required=True, nargs="+",
                    help=("Column name(s) in CSV to use for y axis. "
                          "Supply one ykey per file, or a single ykey for all files. "
                          "Each ykey can be a comma-separated list of columns to plot multiple traces per file."))
    ap.add_argument("--semi", action="store_true",
                    help="Make semilog plot (log scale on Y axis).")
    ap.add_argument("--pdf", metavar="OUTFILE",
                    help="Write plot to a PDF file (requires kaleido).")
    ap.add_argument("--show", action="store_true",
                    help="Show the plot in the default browser (interactive).")
    ap.add_argument("--title", default="",
                    help="Plot title.")
    ap.add_argument("--xlabel", default="",
                    help="Override x-axis label (default: xkey).")
    ap.add_argument("--ylabel", default="",
                    help="Override y-axis label (default: ykey).")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help=f"Figure width in pixels (default {DEFAULT_WIDTH}).")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                    help=f"Figure height in pixels (default {DEFAULT_HEIGHT}).")
    return ap.parse_args()

def expand_ykeys_for_files(files: List[str], ykey_args: List[str]) -> List[List[str]]:
    """
    Return a list (per file) of list of y-column names.
    ykey_args may contain:
      - exactly 1 entry => apply to all files (may be comma-separated)
      - same length as files => each entry applies to corresponding file
    Each entry can be comma-separated to request multiple y columns for that file.
    """
    if len(ykey_args) == 1 and len(files) > 1:
        base = [s.strip() for s in ykey_args[0].split(",") if s.strip()]
        return [base for _ in files]
    if len(ykey_args) == len(files):
        out = []
        for s in ykey_args:
            cols = [c.strip() for c in s.split(",") if c.strip()]
            out.append(cols)
        return out
    # fallback: try using the first for all
    base = [s.strip() for s in ykey_args[0].split(",") if s.strip()]
    return [base for _ in files]

def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}")
    return df

def make_fig(files: List[str], labels: List[str], xkey: str, ykeys_per_file: List[List[str]], *,
             title: str = "", xlabel: str = "", ylabel: str = "", semi: bool = False,
             width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> go.Figure:
    fig = go.Figure()

    color_cycle = LINE_PALETTE
    color_idx = 0
    for idx, path in enumerate(files):
        label = labels[idx]
        df = load_csv(path)

        if xkey not in df.columns:
            raise KeyError(f"xkey '{xkey}' not found in columns of {path}. Available columns: {list(df.columns)}")

        x = df[xkey]

        for ycol in ykeys_per_file[idx]:
            if ycol not in df.columns:
                raise KeyError(f"y column '{ycol}' not found in {path}. Available columns: {list(df.columns)}")
            y = df[ycol]

            trace_name = f"{label} — {ycol}" if len(ykeys_per_file[idx]) > 1 else label
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=trace_name,
                marker=dict(symbol="circle-open", size=6),
                line=dict(width=2),
                hovertemplate=f"%{{x}}<br>%{{y}}<extra>{trace_name}</extra>",
                marker_color=color_cycle[color_idx % len(color_cycle)]
            ))
            color_idx += 1

    # Layout: white background, publication-friendly
    fig.update_layout(
        template=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14)),
        font=DEFAULT_FONT,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bordercolor="rgba(0,0,0,0)",
            bgcolor="rgba(255,255,255,0)"
        ),
        margin=dict(l=80, r=40, t=80, b=80),
        width=width,
        height=height
    )
    fig.update_xaxes(range=[0.0, 70.0])

    # Axes styling (thin axis lines, ticks outwards-like)
    fig.update_xaxes(
        title_text=(xlabel or xkey),
        title_font=dict(family="Arial, Helvetica, DejaVu Sans, sans-serif", size=20),
        tickfont=dict(family="Arial, Helvetica, DejaVu Sans, sans-serif", size=14),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#222",
        linewidth=1,
        ticks="outside",
        mirror=True,
    )
    fig.update_yaxes(
        title_text=(ylabel or "" if ylabel else ", ".join({yk for sub in ykeys_per_file for yk in sub})),
        title_font=dict(family="Arial, Helvetica, DejaVu Sans, sans-serif", size=20),
        tickfont=dict(family="Arial, Helvetica, DejaVu Sans, sans-serif", size=14),
        showgrid=True,
        gridcolor="#eaeaea",
        zeroline=False,
        showline=True,
        linecolor="#222",
        linewidth=1,
        ticks="outside",
        mirror=True,
    )

    if semi:
        # log y axis
        fig.update_yaxes(type="log", autorange=True)
        # If there are non-positive values, Plotly will fail to render them on log scale.
        # We leave it to the user to ensure the y data are positive; otherwise plotly will drop invalid points.

    return fig

def main():
    args = parse_args()

    files = args.files
    # verify file existence early:
    for f in files:
        if not os.path.exists(f):
            print(f"Error: file not found: {f}", file=sys.stderr)
            sys.exit(2)

    labels = []
    if args.labels:
        if len(args.labels) != len(files):
            if len(args.labels) == 1 and len(files) > 1:
                labels = [args.labels[0]] * len(files)
            else:
                print("Warning: number of labels does not match number of files; falling back to filenames for missing labels.", file=sys.stderr)
                # build labels using provided ones and filenames
                labels = list(args.labels) + [os.path.splitext(os.path.basename(f))[0] for f in files[len(args.labels):]]
        else:
            labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in files]

    # Prepare ykeys per file
    try:
        ykeys_per_file = expand_ykeys_for_files(files, args.ykey)
    except Exception as e:
        print(f"Error parsing ykey arguments: {e}", file=sys.stderr)
        sys.exit(2)

    # Build figure
    try:
        fig = make_fig(
            files=files,
            labels=labels,
            xkey=args.xkey,
            ykeys_per_file=ykeys_per_file,
            title=args.title,
            xlabel=args.xlabel or args.xkey,
            ylabel=args.ylabel,
            semi=args.semi,
            width=args.width,
            height=args.height
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)

    # Output
    if args.pdf:
        out = args.pdf
        # Ensure extension
        if not out.lower().endswith(".pdf"):
            out = out + ".pdf"
        try:
            # Use kaleido via plotly.io
            pio.write_image(fig, out, format="pdf", engine="kaleido")
            print(f"Wrote PDF to: {out}")
        except Exception as e:
            print("Failed to write PDF. Make sure 'kaleido' is installed (pip install -U kaleido).", file=sys.stderr)
            print(f"Detailed error: {e}", file=sys.stderr)
            sys.exit(4)

    if args.show or not args.pdf:
        # If user requested show OR didn't request PDF specifically, open the interactive html in browser
        # We'll create a temporary HTML viewer if they didn't request PDF only.
        html_out = os.path.splitext(os.path.basename(files[0]))[0] + "_plot.html"
        try:
            pio.write_html(fig, file=html_out, auto_open=True, include_plotlyjs="cdn")
            print(f"Interactive plot written to {html_out} (opened in browser).")
        except Exception as e:
            print(f"Failed to write/show interactive HTML: {e}", file=sys.stderr)
            sys.exit(5)

if __name__ == "__main__":
    main()
