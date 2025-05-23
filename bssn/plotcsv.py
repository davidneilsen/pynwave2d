import argparse
from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot data from multiple files.")
    parser.add_argument("files", nargs='+', help="List of data files to plot.")
    parser.add_argument("-x", "--xkey", type=str, required=True, help="key for x data")
    parser.add_argument("-y", "--ykey", type=str, required=True, help="key for y data")
    parser.add_argument("--labels", nargs='*', help="Optional labels for each dataset")
    parser.add_argument("--title", type=str, help="Optional title for the plot")
    parser.add_argument("--xlabel", type=str, help="Optional label for the x-axis")
    parser.add_argument("--ylabel", type=str, help="Optional label for the y-axis")
    parser.add_argument("--ylog", action="store_true", help="Semi-log plot")
    parser.add_argument("--noshow", action="store_false", help="do not show plot")
    parser.add_argument("-c", "--conv", type=int, default=-1, help="Plot convergence [INT]")
    parser.add_argument("-s", "--save", type=str, help="save file with filename")
    
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match the number of files.")
        return

    #sns.set_theme(context='paper', style='ticks', palette='colorblind')
    sns.set_theme(context='paper', style='ticks', palette='bright')

#    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
#                  cycler(linestyle=['-', '--', ':', '-.']))
#    plt.rc('lines', linewidth=4)
#    plt.rc('axes', prop_cyle=default_cycler)

#    mpl.rcParams['axes.prop_cycle'] = (cycler(color=['b', 'g', 'r', 'c']) + cycler(linestyle=['-', '--', ':', '-.']))

#    mpl.rcParams['axes.prop_cycle'] = (cycler(linestyle=['-', '--', ':', '-.']) * cycler(color=['b', 'g', 'r', 'm', 'y']))

    for i, file in enumerate(args.files):
        try:
            data = pd.read_csv(file)
            for key in data.keys():
                if re.match(r'^' + args.xkey, key):
                    xkey = key
            for key in data.keys():
                if re.match(r'^' + args.ykey, key):
                    ykey = key
            print("xkey = " + xkey)        
            print("ykey = " + ykey)        
            x = data[xkey]
            y = data[ykey]
            label = args.labels[i] if args.labels else file.replace('/maxwell_constraints.csv','')
#           if '4' in label:
#               lalpha = 0.3
#           else:
#               lalpha = 1.0
#           if '642' in label:
#               lalpha = 1.0
               
            lalpha = 1.0
            lstyle = '-'
            if 'L0' in label:
                lalpha = 0.3
            if 'E' in label:
                lstyle = '--'
            
            if args.ylog == True:
                plt.semilogy(x, y, label=label, linestyle=lstyle, alpha=lalpha)
            else:
                plt.plot(x, y, label=label)
            if args.conv > 1:
                if i == 0 or i == 1:
                    factor = 2**(args.conv)
                    temp = 1.0 / factor**(2-i)
                    y2 = [temp * e for e in y]
                    label2 = label + ' / 2^' + str(args.conv)
                    if args.ylog == True:
                        plt.semilogy(x, y2, linestyle='dashed', label=label2)
                    else:
                        plt.plot(x, y2, linestyle='dashed', label=label2)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if args.title:
        plt.title(args.title)
    
    plt.xlabel(args.xlabel if args.xlabel else f"{args.xkey}")
    plt.ylabel(args.ylabel if args.ylabel else f"{args.ykey}")
    plt.legend(ncol=3)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save)
    if args.noshow:
        plt.show()

if __name__ == "__main__":
    main()

