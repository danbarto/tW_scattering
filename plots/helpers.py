import numpy as np
import os
import matplotlib.pyplot as plt
from Tools.helpers import finalizePlotDir
from coffea import hist

import re
bkgonly = re.compile('(?!(MuonEG))')

def makePlot(histo, axis, bins=None, mc_sel=bkgonly, data_sel='MuonEG', normalize=True, log=False, save=False):
    
    processes = [ p[0] for p in histo.values().keys() if not p[0]=='MuonEG' ]
    
    histogram = histo.copy()
    histogram = histogram.project(axis, 'dataset')
    if bins:
        histogram = histogram.rebin(axis, bins)

    y_max = histogram[mc_sel].sum("dataset").values(overflow='over')[()].max()

    MC_total = histogram[mc_sel].sum("dataset").values(overflow='over')[()].sum()
    Data_total = histogram[data_sel].sum("dataset").values(overflow='over')[()].sum()
    
    print ("Data:", round(Data_total,0), "MC:", round(MC_total,2))
    
    if normalize:
        scales = { process: Data_total/MC_total for process in processes }
        histogram.scale(scales, axis='dataset')
    else:
        scales = {}
    
    fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    ax = hist.plot1d(histogram[mc_sel], overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts)
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

    
    rax.set_ylim(0.1,1.9)
    rax.set_ylabel('Obs./Pred.')
    ax.set_ylabel('Events')
    
    addUncertainties(ax, axis, histogram, mc_sel, [], [], overflow='over', rebin=bins, ratio=False, scales=scales)
    addUncertainties(rax, axis, histogram, mc_sel, [], [], overflow='over', rebin=bins, ratio=True, scales=scales)
    
    if log:
        ax.set_yscale('log')
        
    y_mult = 1.3 if not log else 100
    ax.set_ylim(0.01,y_max*y_mult)

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

    if save:
        fig.savefig(save)
        print ("Figure saved in:", save)


def saveFig( fig, ax, rax, path, name, scale='linear', shape=False, y_max=-1 ):
    # this should get retired
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    ax.set_ylabel('Events')

    if scale == 'linear':
        if y_max<0 or True:
            pass
        else:
            ax.set_ylim(0, 1 if shape else 1.2*y_max)
    else:
        if y_max<0 and not shape:
            pass
        else:
            ax.set_ylim(0.000005 if shape else 0.05, 3 if shape else 300*y_max)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    WHcount = 0
    for handle, label in zip(handles, labels):
        try:
            new_labels.append(my_labels[label])
            if not label=='pseudodata':
                handle.set_color(colors[label])
        except:
            pass

    if rax:
        plt.subplots_adjust(hspace=0)
        rax.set_ylabel('Obs./Pred.')
        rax.set_ylim(0.5,1.5)

    ax.legend(title='',ncol=2,handles=handles, labels=new_labels, frameon=False)

    fig.text(0.0, 0.995, '$\\bf{CMS}$', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.15, 1., '$\\it{Simulation}$', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.8, 1., '13 TeV', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    fig.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    fig.savefig(os.path.join(outdir, "{}.png".format(name)))
    #ax.clear()

def addUncertainties(ax, axis, h, selection, up_vars, down_vars, overflow='over', rebin=False, ratio=False, scales={}):
    
    if rebin:
        h = h.rebin(axis, rebin)
    
    bins = h[selection].axis(axis).edges(overflow=overflow)
    
    values = h[selection].sum('dataset').values(overflow=overflow, sumw2=True)[()]
    central = values[0]
    stats = values[1]
    
    up = np.zeros_like(central)
    down = np.zeros_like(central)
    
    for up_var in up_vars:
        if rebin:
            up_var = up_var.rebin(axis, rebin)
            up_var.scale(scales, axis='dataset')
        up += (up_var[selection].sum('dataset').values(overflow=overflow, sumw2=False)[()] - central)**2
    
    for down_var in down_vars:
        if rebin:
            down_var = down_var.rebin(axis, rebin)
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

signal_fill_opts = {
    'linewidth': 2,
    'linecolor': 'k',
    'edgecolor': (1,1,1,0.0),
    'facecolor': 'none',
    'alpha': 0.1
}

