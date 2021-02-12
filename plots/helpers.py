import numpy as np
import os
import matplotlib.pyplot as plt
from Tools.helpers import finalizePlotDir
from coffea import hist

import re
bkgonly = re.compile('(?!(MuonEG))')

def makePlot(output, histo, axis, bins=None, mc_sel=bkgonly, data_sel='MuonEG', normalize=True, log=False, save=False, axis_label=None, ratio_range=None, upHists=[], downHists=[], shape=False):
    
    processes = [ p[0] for p in output[histo].values().keys() if not p[0]=='MuonEG' ]
    
    histogram = output[histo].copy()
    histogram = histogram.project(axis, 'dataset')
    if bins:
        histogram = histogram.rebin(axis, bins)

    y_max = histogram[mc_sel].sum("dataset").values(overflow='over')[()].max()

    MC_total = histogram[mc_sel].sum("dataset").values(overflow='over')[()].sum()
    Data_total = 0
    if data_sel:
        Data_total = histogram[data_sel].sum("dataset").values(overflow='over')[()].sum()
    
    print ("Data:", round(Data_total,0), "MC:", round(MC_total,2))
    
    if normalize and data_sel:
        scales = { process: Data_total/MC_total for process in processes }
        histogram.scale(scales, axis='dataset')
    else:
        scales = {}

    if shape:
        scales = { process: 1/histogram[process].sum("dataset").values(overflow='over')[()].sum() for process in processes }
        histogram.scale(scales, axis='dataset')
    
    if data_sel:
        fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    else:
        fig, ax  = plt.subplots(1,1,figsize=(10,10) )

    if shape:
        ax = hist.plot1d(histogram[mc_sel], overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None)
    else:
        ax = hist.plot1d(histogram[mc_sel], overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts)
    if data_sel:
        ax = hist.plot1d(histogram[data_sel], overlay="dataset", ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

        hist.plotratio(
                num=histogram[data_sel].sum("dataset"),
                denom=histogram[mc_sel].sum("dataset"),
                ax=rax,
                error_opts=data_err_opts,
                denom_fill_opts=None, # triggers this: https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L376
                guide_opts={},
                unc='num',
                #unc=None,
                overflow='over'
        )
    
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for handle, label in zip(handles, labels):
        try:
            new_labels.append(my_labels[label])
            if not label=='MuonEG':
                handle.set_color(colors[label])
        except:
            pass

    if data_sel:
        if ratio_range:
            rax.set_ylim(*ratio_range)
        else:
            rax.set_ylim(0.1,1.9)
        rax.set_ylabel('Obs./Pred.')
        if axis_label:
            rax.set_xlabel(axis_label)

    ax.set_xlabel(axis_label)
    ax.set_ylabel('Events')
    
    if not shape:
        addUncertainties(ax, axis, histogram, mc_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=False, scales=scales)

    if data_sel:
        addUncertainties(rax, axis, histogram, mc_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=True, scales=scales)
    
    if log:
        ax.set_yscale('log')
        
    y_mult = 1.3 if not log else 100
    ax.set_ylim(0.01,y_max*y_mult if not shape else 2)

    ax.legend(
        loc='upper right',
        ncol=2,
        borderaxespad=0.0,
        labels=new_labels,
        handles=handles,
    )
    plt.subplots_adjust(hspace=0)

    fig.text(0.0, 0.995, '$\\bf{CMS}$ Preliminary', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.8, 0.995, '13 TeV', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    if normalize:
        fig.text(0.55, 0.65, 'Data/MC = %s'%round(Data_total/MC_total,2), fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )


    if save:
        #finalizePlotDir(outdir)
        fig.savefig(save)
        print ("Figure saved in:", save)


def addUncertainties(ax, axis, h, selection, up_vars, down_vars, overflow='over', rebin=False, ratio=False, scales={}):
    
    if rebin:
        h = h.project(axis, 'dataset').rebin(axis, rebin)
    
    bins = h[selection].axis(axis).edges(overflow=overflow)
    
    values = h[selection].project(axis, 'dataset').sum('dataset').values(overflow=overflow, sumw2=True)[()]
    central = values[0]
    stats = values[1]
    
    up = np.zeros_like(central)
    down = np.zeros_like(central)
    
    for up_var in up_vars:
        if rebin:
            up_var = up_var.project(axis, 'dataset').rebin(axis, rebin)
            up_var.scale(scales, axis='dataset')
        up += (up_var[selection].sum('dataset').values(overflow=overflow, sumw2=False)[()] - central)**2
    
    for down_var in down_vars:
        if rebin:
            down_var = down_var.project(axis, 'dataset').rebin(axis, rebin)
            down_var.scale(scales, axis='dataset')
        down += (down_var[selection].sum('dataset').values(overflow=overflow, sumw2=False)[()] - central)**2
    
    up   += stats 
    down += stats
 
    if ratio:
        up = np.ones_like(central) + np.sqrt(up)/central
        down = np.ones_like(central) - np.sqrt(down)/central
    else:
        up = central + np.sqrt(up)
        down = central - np.sqrt(down)
    
    opts = {'step': 'post', 'label': 'uncertainty', 'hatch': '///',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0}
    
    ax.fill_between(x=bins, y1=np.r_[down, down[-1]], y2=np.r_[up, up[-1]], **opts)

colors = {
    'tW_scattering': '#FF595E',
    'topW_v2': '#FF595E',
    #'tW_scattering': '#000000', # this would be black
    'TTW': '#8AC926',
    'TTX': '#FFCA3A',
    'TTZ': '#FFCA3A',
    'TTH': '#34623F',
    'TTTT': '#0F7173',
    'ttbar': '#1982C4',
    'wjets': '#6A4C93',
    'diboson': '#525B76',
    'WZ': '#525B76',
    'WW': '#34623F',
    'DY': '#6A4C93',
    'MuonEG': '#000000',
}
'''
other colors (sets from coolers.com):
#525B76 (gray)
#34623F (hunter green)
#0F7173 (Skobeloff)
'''

my_labels = {
    'tW_scattering': 'top-W scat.',
    'topW_v2': 'top-W scat.',
    'TTW': r'$t\bar{t}$W+jets',
    'TTX': r'$t\bar{t}$Z/H',
    'TTH': r'$t\bar{t}$H',
    'TTZ': r'$t\bar{t}$Z',
    'TTTT': r'$t\bar{t}t\bar{t}$',
    'ttbar': r'$t\bar{t}$+jets',
    'wjets': 'W+jets',
    'DY': 'Drell-Yan',
    'diboson': 'VV/VVV',
    'WZ': 'WZ',
    'WW': 'WW',
    'MuonEG': 'Observed',
    'pseudodata': 'Pseudo-data',
    'uncertainty': 'Uncertainty',
}


data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

signal_err_opts = {
    'linestyle':'-',
    'color':'crimson',
    'elinewidth': 1,
}

#signal_err_opts = {
#    'linestyle': '-',
#    'marker': '.',
#    'markersize': 0.,
#    'color': 'k',
#    'elinewidth': 1,
#    'linewidth': 2,
#}


error_opts = {
    'label': 'uncertainty',
    'hatch': '///',
    'facecolor': 'none',
    'edgecolor': (0,0,0,.5),
    'linewidth': 0
}

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 1.0
}

line_opts = {
    'linestyle':'-',
#    'elinewidth': 1,
}

signal_fill_opts = {
    'linewidth': 2,
    'linecolor': 'k',
    'edgecolor': (1,1,1,0.0),
    'facecolor': 'none',
    'alpha': 0.1
}

