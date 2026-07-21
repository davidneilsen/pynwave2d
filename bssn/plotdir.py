import argparse
from cycler import cycler
import numpy as np
import pandas as pd
import re
import seaborn as sns
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description="Plot data from multiple files.")
    parser.add_argument("files", nargs="+", help="List of data files to plot.")
    parser.add_argument("-x", "--xkey", type=str, required=True, help="key for x data")
    parser.add_argument("-y", "--ykey", type=str, required=True, help="key for y data")
    parser.add_argument("--labels", nargs="*", help="Optional labels for each dataset")
    parser.add_argument("--title", type=str, help="Optional title for the plot")
    parser.add_argument("--xlabel", type=str, help="Optional label for the x-axis")
    parser.add_argument("--ylabel", type=str, help="Optional label for the y-axis")
    parser.add_argument("--ylog", action="store_true", help="Semi-log plot")
    parser.add_argument("--noshow", action="store_false", help="do not show plot")
    parser.add_argument(
        "-c", "--conv", type=int, default=-1, help="Plot convergence [INT]"
    )
    parser.add_argument("-s", "--save", type=str, help="save file with filename")

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match the number of files.")
        print("Num of files: " + str(len(args.files)))
        print("Num of labels: " + str(len(args.labels)))
        return

    # sns.set_theme(context='paper', style='ticks', palette='colorblind')
    #sns.set_theme(context="paper", style="ticks", palette="bright")

    #    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
    #                  cycler(linestyle=['-', '--', ':', '-.']))
    #    plt.rc('lines', linewidth=4)
    #    plt.rc('axes', prop_cyle=default_cycler)

    #    mpl.rcParams['axes.prop_cycle'] = (cycler(color=['b', 'g', 'r', 'c']) + cycler(linestyle=['-', '--', ':', '-.']))

    #    mpl.rcParams['axes.prop_cycle'] = (cycler(linestyle=['-', '--', ':', '-.']) * cycler(color=['b', 'g', 'r', 'm', 'y']))

    fig_data = []
    for i, file in enumerate(args.files):
        try:
            fname = file + "/bssn_constraints.dat"
            data = pd.read_csv(fname)
            for key in data.keys():
                if re.match(r"^" + args.xkey, key):
                    xkey = key
            for key in data.keys():
                if re.match(r"^" + args.ykey, key):
                    ykey = key
            print("xkey = " + xkey)
            print("ykey = " + ykey)
            x = data[xkey]
            y = data[ykey]
            label = (
                args.labels[i]
                if args.labels
                else file.replace("/maxwell_constraints.csv", "")
            )
            #           if '4' in label:
            #               lalpha = 0.3
            #           else:
            #               lalpha = 1.0
            #           if '642' in label:
            #               lalpha = 1.0

            lalpha = 1.0
            lstyle = "-"
            if "L0" in label:
                lalpha = 0.3
            if "E" in label:
                lstyle = "--"

            fig_data.append([x, y, label])
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # our custom event handler
    def update_trace(trace, points, selector):
        # this list stores the points which were clicked on
        # in all but one trace they are empty
        if len(points.point_inds) == 0:
            return
        for i,_ in enumerate(f.data):
            fig.data[i]['line']['width'] = default_linewidth + highlighted_linewidth_delta * (i == points.trace_index)

    fig = go.FigureWidget()
    fig.layout.hovermode = 'closest'
    fig.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
    default_linewidth = 2
    highlighted_linewidth_delta = 2

    num_of_traces = len(fig_data)
    for dd in fig_data:
        trace = go.Scatter(
            x=dd[0],
            y=dd[1],
            mode="lines+markers",
            name=dd[2],
            line={'width': default_linewidth },
            hoverinfo='skip'
        )
        fig.add_trace(trace)

    # we need to add the on_click event to each trace separately       
    for i in range( len(fig.data) ):
        fig.data[i].on_click(update_trace)

    # Layout with log scale option
    fig.update_layout(
        title='Derivative Error Plots (dxf)',
        xaxis=dict(title='x', type='linear'),  # use 'log' for log scale
        yaxis=dict(title='y', type='log'),
        hovermode='x',
        template='plotly_white'
    )

    # let's show the figure 
    fig.show()
         

if __name__ == "__main__":
    main()
