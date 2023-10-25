#!/usr/bin/env python3

import numpy as np
import pandas as pd
import uproot

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

if __name__ == '__main__':

    f_in = uproot.open("../Histos.Frozen.cpQMcpt.root")

    hist = f_in['hist_pxy;2']
    x_min = -15
    x_max = 20
    x_bins = 100
    y_min = -15
    y_max = 15
    y_bins = 100

    fig, ax, = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(
        "WIP",
        data=True,
        #year=2018,
        lumi=137,
        loc=0,
        ax=ax,
        )

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, x_bins),
        np.linspace(y_min, y_max, y_bins)
    )

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel(r'$C_{\varphi Q}^{-}/\Lambda^{2} (TeV^{-2})$')
    ax.set_ylabel(r'$C_{\varphi t}/\Lambda^{2} (TeV^{-2})$')

    Z = hist.values().transpose()
    Z = Z + (Z==0)*1000  # replace zeros with large values to remove additional contours

    CS = ax.contour(X, Y, Z, levels = [2.28, 5.99], colors=['#FF595E', '#5bc0de'], # 68/95 % CL
                    linestyles=('-',),linewidths=(4,))


    fmt = {}
    strs = ['68%', '95%']
    for l, s in zip(CS.levels, strs):
        fmt[l] = s

    # Label every other level using strings
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

    fig.savefig('TOPEFT_ver4.png')
    
    if False:
        # NOTE old attempt to use the fit output directly, but this needs smoothing
        # that's why the above histograms are used
        #
        # load combine multi dim fit result file
        f_in = uproot.open("../higgsCombine.021723.EFT.Frozen.anatest25v1.Data.cpQMcpt.MultiDimFit.root")
        tree = f_in['limit']

        # filter and sort cpt and cpqm values
        tmp = pd.DataFrame({'cpt': np.round(tree.arrays()['cpt'], 3), 'cpQM': np.round(tree.arrays()['cpQM'], 3), 'deltaNLL': tree.arrays()['deltaNLL']})
        tmp = tmp[(((tmp['cpt']%0.05>0.048)|(tmp['cpt']%0.05<0.002)))]  #

        cpQM = sorted(list(set(tmp['cpQM'])))
        cpt  = sorted(list(set(tmp['cpt'])))

        # some points are missing (failed fits?)
        missing_points = []
        for ix in cpQM:
            for iy in cpt:
                if len(tmp[((tmp['cpt']==iy) & (tmp['cpQM']==ix))])<1:
                    missing_points.append((ix,iy))

        vx = list(tmp['cpt'])
        vy = list(tmp['cpQM'])
        vz = list(tmp['deltaNLL'])

        for x, y in missing_points:
            vx.append(x)
            vy.append(y)
            vz.append(0)  # NOTE maybe this should be nan or whatever

        X, Y = np.meshgrid(cpQM, cpt)

        df = pd.DataFrame({
            'cpt': vx,
            'cpQM': vy,
            'deltaNLL': vz,
        })

        # now sort
        df = df.sort_values(by=['cpt', 'cpQM'])
        Z = np.reshape(df['deltaNLL'].values, X.shape)

        fig, ax, = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(
            "WIP",
            data=False,
            #year=2018,
            lumi=41.5,
            loc=0,
            ax=ax,
            )

        ax.set_ylim(-25, 20)
        ax.set_xlim(-15, 20)

        ax.set_xlabel(r'$C_{\varphi Q}^{-}/\Lambda^{2} (TeV^{-2})$')
        ax.set_ylabel(r'$C_{\varphi t}/\Lambda^{2} (TeV^{-2})$')

        # NOTE: X and Y are switched in this plot!
        CS = ax.contour(Y, X, Z, levels = [2.28, 5.99], colors=['#FF595E', '#5bc0de'], # 68/95 % CL
                        linestyles=('-',),linewidths=(4,))
        fmt = {}
        strs = ['68%', '95%']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

        fig.savefig('TOPEFT.png')