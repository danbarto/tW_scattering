import numpy as np
import os
import matplotlib.pyplot as plt
from Tools.helpers import finalizePlotDir
from coffea import hist

import re


colors = {
    'tW_scattering': '#FF595E',
    'topW_v2': '#FF595E',
    'topW_v3': '#FF595E',
    #'tW_scattering': '#000000', # this would be black
    'TTW': '#8AC926',
    'TTX': '#FFCA3A',
    'TTZ': '#FFCA3A',
    'TTH': '#34623F',
    'TTTT': '#0F7173',
    'ttbar': '#1982C4',
    'wjets': '#6A4C93',
    'diboson': '#525B76',
    'rare': '#6A4C93',
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
    'topW_v3': 'top-W scat.',
    'TTW': r'$t\bar{t}$W+jets',
    'TTX': r'$t\bar{t}$Z/H',
    'TTH': r'$t\bar{t}$H',
    'TTZ': r'$t\bar{t}$Z',
    'TTTT': r'$t\bar{t}t\bar{t}$',
    'ttbar': r'$t\bar{t}$+jets',
    'wjets': 'W+jets',
    'DY': 'Drell-Yan',
    'diboson': 'VV/VVV',
    'rare': 'Rare',
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
    'linewidth': 3,
}

signal_fill_opts = {
    'linewidth': 2,
    'linecolor': 'k',
    'edgecolor': (1,1,1,0.0),
    'facecolor': 'none',
    'alpha': 0.1
}


def makePlot(output, histo, axis, bins=None, data=[], normalize=True, log=False, save=False, axis_label=None, ratio_range=None, upHists=[], downHists=[], shape=False, ymax=False, new_colors=colors, new_labels=my_labels, order=None, signals=[], omit=[]):
    
    if save:
        finalizePlotDir( '/'.join(save.split('/')[:-1]) )
    
    mc_sel   = re.compile('(?!(%s))'%('|'.join(data+omit))) if len(data+omit)>0 else re.compile('')
    data_sel = re.compile('|'.join(data))
    bkg_sel  = re.compile('(?!(%s))'%('|'.join(data+signals+omit))) if len(data+signals+omit)>0 else re.compile('')

    if histo is None:
        processes = [ p[0] for p in output.values().keys() if not p[0] in data ]
        histogram = output.copy()
    else:
        processes = [ p[0] for p in output[histo].values().keys() if not p[0] in data ]
        histogram = output[histo].copy()

    histogram = histogram.project(axis, 'dataset')
    if bins:
        histogram = histogram.rebin(axis, bins)

    y_max = histogram[bkg_sel].sum("dataset").values(overflow='over')[()].max()

    MC_total = histogram[bkg_sel].sum("dataset").values(overflow='over')[()].sum()
    Data_total = 0
    if data:
        Data_total = histogram[data_sel].sum("dataset").values(overflow='over')[()].sum()
        #observation = histogram[data[0]].sum('dataset').copy()
        #first = True
        #for d in data:
        #    print (d)
        #    if not first:
        #        observation.add(histogram[d].sum('dataset'))
        #        print ("adding")
        #    first = False
    
    print ("Data:", round(Data_total,0), "MC:", round(MC_total,2))
    
    if normalize and data_sel:
        scales = { process: Data_total/MC_total for process in processes }
        histogram.scale(scales, axis='dataset')
    else:
        scales = {}

    if shape:
        scales = { process: 1/histogram[process].sum("dataset").values(overflow='over')[()].sum() for process in processes }
        histogram.scale(scales, axis='dataset')
    
    if data:
        fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    else:
        fig, ax  = plt.subplots(1,1,figsize=(10,10) )

    if signals:
        for sig in signals:
            ax = hist.plot1d(histogram[sig], overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None)

    if shape:
        ax = hist.plot1d(histogram[bkg_sel], overlay="dataset", ax=ax, stack=False, overflow='over', clear=False, line_opts=line_opts, fill_opts=None)
    else:
        ax = hist.plot1d(histogram[bkg_sel], overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, order=(order if order else processes))
    if data:
        ax = hist.plot1d(histogram[data_sel].sum("dataset"), ax=ax, overflow='over', error_opts=data_err_opts, clear=False)
        #ax = hist.plot1d(observation, ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

        hist.plotratio(
                num=histogram[data_sel].sum("dataset"),
                denom=histogram[bkg_sel].sum("dataset"),
                ax=rax,
                error_opts=data_err_opts,
                denom_fill_opts=None, # triggers this: https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L376
                guide_opts={},
                unc='num',
                #unc=None,
                overflow='over'
        )
    
    
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = []
    for handle, label in zip(handles, labels):
        try:
            if label is None or label=='None':
                updated_labels.append("Observation")
                handle.set_color('#000000')
            else:
                updated_labels.append(new_labels[label])
                handle.set_color(new_colors[label])
        except:
            pass

    if data:
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
        addUncertainties(ax, axis, histogram, bkg_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=False, scales=scales)

    if data:
        addUncertainties(rax, axis, histogram, bkg_sel, [output[histo+'_'+x] for x in upHists], [output[histo+'_'+x] for x in downHists], overflow='over', rebin=bins, ratio=True, scales=scales)
    
    if log:
        ax.set_yscale('log')
        
    y_mult = 1.3 if not log else 100
    if ymax:
        ax.set_ylim(0.01, ymax)
    else:
        ax.set_ylim(0.01,y_max*y_mult if not shape else 2)

    ax.legend(
        loc='upper right',
        ncol=2,
        borderaxespad=0.0,
        labels=updated_labels,
        handles=handles,
    )
    plt.subplots_adjust(hspace=0)

    fig.text(0.0, 0.995, '$\\bf{CMS}$ Preliminary', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.8, 0.995, '13 TeV', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    if normalize:
        fig.text(0.55, 0.65, 'Data/MC = %s'%round(Data_total/MC_total,2), fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )


    if save:
        #finalizePlotDir(outdir)
        fig.savefig("{}.pdf".format(save))
        fig.savefig("{}.png".format(save))
        #fig.savefig(save)
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


def scale_and_merge(histogram, samples, fileset, nano_mapping, lumi=60):
    """
    Scale NanoAOD samples to a physical cross section.
    Merge NanoAOD samples into categories, e.g. several ttZ samples into one ttZ category.
    
    histogram -- coffea histogram
    samples -- samples dictionary that contains the x-sec and sumWeight
    fileset -- fileset dictionary used in the coffea processor
    nano_mapping -- dictionary to map NanoAOD samples into categories
    lumi -- integrated luminosity in 1/fb
    """
    histogram = histogram.copy()
    
    scales = {sample: lumi*1000*samples[sample]['xsec']/samples[sample]['sumWeight'] for sample in samples if sample in fileset}
    histogram.scale(scales, axis='dataset')
    for cat in nano_mapping:
        # print (cat)
        if len(nano_mapping[cat])>1:
            for sample in nano_mapping[cat][1:]:
                histogram[nano_mapping[cat][0]].add(histogram[sample])
                histogram = histogram.remove([sample], 'dataset')
                
    return histogram
