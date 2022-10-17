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

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--year', action='store', default='2016', help='Specify year to plot')
    argParser.add_argument('--dir', action='store', default='./results', help='Specify directory with results')
    argParser.add_argument('--files', action='store', nargs=2, help="Specify files to plot")
    args = argParser.parse_args()


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
            list_of_files = glob.glob(f'{args.dir}/results_{args.year}_{method[i]}_*')
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
        for point in json_data[f]:
            parsed = point.split('_')
            results[f]['X'].append(parsed[1])
            results[f]['Y'].append(parsed[3])
            results[f]['Z'].append(json_data[f][point])
        N = int(np.sqrt(len(results[f]['X'])))
        for axis in ['X','Y','Z']:
            results[f][axis] = np.reshape(results[f][axis], (N,N)).tolist()

    print(results)

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

    fmt_bit = {}
    strs_bit = ['BIT, 68%', 'BIT, 95%']
    for l, s in zip(CS_bit.levels, strs_bit):
        fmt_bit[l] = s

    fmt_lt = {}
    strs_lt = ['LT, 68%', 'LT, 95%']
    for l, s in zip(CS_lt.levels, strs_lt):
        fmt_lt[l] = s

    # Label every other level using strings
    ax.clabel(CS_bit, CS_bit.levels, inline=True, fmt=fmt_bit, fontsize=10)
    ax.clabel(CS_lt, CS_lt.levels, inline=True, fmt=fmt_lt, fontsize=10)

    plt.show()

    fig.savefig('/home/users/sjeon/public_html/tW_scattering/scan_comparison.png')
    fig.savefig('/home/users/sjeon/public_html/tW_scattering/scan_comparison.pdf')
