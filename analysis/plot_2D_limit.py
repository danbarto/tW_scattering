#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')

# data handling and numerical analysis
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import boost_histogram as bh

import scipy
import pickle

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from analysis.Tools.config_helpers import finalizePlotDir, loadConfig

if __name__ == '__main__':
    cfg = loadConfig()

    plot_dir = os.path.expandvars(f"{cfg['meta']['plots']}/BIT/toys/")
    finalizePlotDir(plot_dir)

    with open('results_with_fixed_template_5_-5_training_v50.pkl', 'rb') as f:
        results = pickle.load(f)

    x = np.array([-6,-3,0,3,6])
    y = np.array([-6,-3,0,3,6])
    X, Y = np.meshgrid(x, y)

    z = []
    n_toys = 0
    z_toys = {i:[] for i in range(n_toys)}

    for ix, iy in zip(X.flatten(), Y.flatten()):

        predicted = -2*(results[f'SM_{ix}_{iy}'] - results[f'BSM_{ix}_{iy}'])
        #predicted = -2*(results[f'BSM_{ix}_{iy}'] - results[f'BSM_0_0'])
        z.append(predicted)

        for i in range(n_toys):
            z_toys[i].append(-2*(results[f'SM_toy_{i}_{ix}_{iy}'] - results[f'BSM_toy_{i}_{ix}_{iy}']))
            #z_toys[i].append(-2*(results[f'SM_toy_{i}_{ix}_{iy}'] - results[f'BSM_toy_{i}_0_0']))


    fig, ax, = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(
        "Preliminary",
        data=True,
        #year=2018,
        lumi=138,
        loc=0,
        ax=ax,
        )

    strs = ['68%', '95%']
    levels = [2.28, 5.99]  # higher values for 2D limits!
    styles = ['dotted', 'dashed', 'dashdot', 'loosely dotted']

    Z_exp = np.array(z)
    Z_exp = np.reshape(Z_exp, X.shape)
    CS_exp = ax.contour(X, Y, Z_exp, levels = levels, colors=['blue', 'red'], # 68/95 % CL
                    linestyles='solid', linewidths=(3))

    CS_toys = []
    for i in range(n_toys):
        Z_tmp = np.reshape(np.array(z_toys[i]), X.shape)
        CS_toys.append(
            ax.contour(X, Y, Z_tmp, levels = levels, colors=['blue', 'red'],
                    linestyles='dotted', linewidths=(3))
        )

    fmt = {}
    for l, s in zip(CS_exp.levels, strs):
        fmt[l] = s

    # Label every other level using strings
    ax.clabel(CS_exp, CS_exp.levels, inline=True, fmt=fmt, fontsize=10)
    ax.set_ylim(-10,10)
    ax.set_xlim(-10,10)


    fig.savefig(f'{plot_dir}/scan_predicted_fixed_5_-5_v50.png')
    fig.savefig(f'{plot_dir}/scan_predicted_fixed_5_-5_v50.pdf')
