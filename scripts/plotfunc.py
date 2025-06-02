import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from readcurve import *

def main():
    parser = argparse.ArgumentParser(description="Plot data from multiple files.")
    parser.add_argument("files", nargs="+", help="List of data files to plot.")
    parser.add_argument("--func", "-f", type=str, help="Function to plot")
    args = parser.parse_args()

    sns.set_theme(context="paper", style="ticks", palette="bright")
    funckey = args.func
    markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v', '<', '>']
    for i, file in enumerate(args.files):
        try:
            data = read_visit_curve_file(file)
            x = data["functions"][funckey][0]
            y = data["functions"][funckey][1]
            time = data["time"]
            lab = f"file: {i:d} time: {time:.2f}"
            plt.plot(x, y, marker=markers[i % len(markers)], label=lab)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
