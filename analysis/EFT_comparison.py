'''
By default, compares BIT and LT results for given year
Can also pass in any two result files to compare
'''

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import glob
import os
import mplhep as hep
plt.style.use(hep.style.CMS)
from plots.helpers import finalizePlotDir

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--dir', action='store', default='./results', help='Specify directory with results')
    argParser.add_argument('--files', action='store', nargs=2, help="Specify files to plot")
    argParser.add_argument('--legend', action='store', nargs=2, default=['BIT','LT'], help="Specify names for the legend")
    argParser.add_argument('--uaf', action='store_true', default=False)
    args = argParser.parse_args()

    # dir to safe plot to
    if args.uaf:
        plot_dir = '/home/users/sjeon/public_html/tW_scattering/multidim/'
    else:
        plot_dir = './plots/'
    finalizePlotDir(plot_dir)

    # load data
    json_data = []
    if args.files:
        for i in range(2):
            with open(args.files[i], 'r') as openfile:
                json_data.append(json.load(openfile))
            print("Opening %s..." %args.files[i])
    else:
        method = ['bit', 'lt']
        for i in range(2):
            list_of_files = glob.glob(f'{args.dir}/results_{method[i]}_*')
            if not list_of_files: # if list is empty
                raise Exception('No files found! Check your arguments.')
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, 'r') as openfile:
                json_data.append(json.load(openfile))
            print("Opening %s..." %latest_file)

    # parse data
    results = {0:{}, 1:{}}
    results[0] = {'X':[], 'Y':[], 'Z':[]}
    results[1] = {'X':[], 'Y':[], 'Z':[]}
    for f in range(2):
        for point in json_data[f]['data']:
            parsed = point.split('_')
            results[f]['X'].append(parsed[1])
            results[f]['Y'].append(parsed[3])
            results[f]['Z'].append(json_data[f]['data'][point])
        N = int(np.sqrt(len(results[f]['X'])))
        for axis in ['X','Y','Z']:
            results[f][axis] = np.reshape(results[f][axis], (N,N)).tolist()

    fig, ax, = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60,
        loc=0,
        ax=ax,
       )

    ax.set_ylim(-8.1, 8.1)
    ax.set_xlim(-8.1, 8.1)

    CS_bit = ax.contour(results[0]['X'], results[0]['Y'], results[0]['Z'],
                 levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                 linestyles='dashed',linewidths=(4,))

    CS_lt = ax.contour(results[1]['X'], results[1]['Y'], results[1]['Z'],
                 levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                 linestyles='solid',linewidths=(4,))

    h1,l1 = CS_bit.legend_elements(args.legend[0])
    h2,l2 = CS_lt.legend_elements(args.legend[1])

    for l in [l1, l2]:
        for i in range(2):
            l[i] = l[i].replace('2.28','68\%')
            l[i] = l[i].replace('5.99', '95\%')
    ax.legend(h1+h2,l1+l2)
    plt.show()

    fig.savefig(plot_dir+'scan_comparison.png')
    fig.savefig(plot_dir+'scan_comparison.pdf')
