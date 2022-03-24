import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist

import numpy as np

from Tools.config_helpers import loadConfig, make_small

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot

from klepto.archives import dir_archive


if __name__ == '__main__':


    small = False
    cfg = loadConfig()

    plot_dir = os.path.expandvars(cfg['meta']['plots']) + '/N_b/' + 'dilep'
    
    cacheName = 'forward_dilep_2018_Nb'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)

    cache.load()

    output = cache.get('simple_output')
    
    # defining some new axes for rebinning.
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    N_bins_red_central = hist.Bin('multiplicity', r'$N$', 8, -0.5, 7.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    min_mass_bins = hist.Bin('mass', r'$M\ (GeV)$', [0,20,40,60,80,82,84,86,88,90,92,94,96,98])
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    eta_bins = hist.Bin('eta', r'$\eta $', 20, -5.0, 5.0)
    ht_bins =  hist.Bin("ht",        r"$H_{T}$ (GeV)", 35, 0, 1400)
    lt_bins =  hist.Bin("ht",        r"$H_{T}$ (GeV)", 50, 0, 1000)
    m3l_bins = hist.Bin('mass', r'$M\ (GeV)$', 30, 100, 700)
    mll_bins = hist.Bin('mass', r'$M\ (GeV)$', 10, 80, 100)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
    st_bins =  hist.Bin("ht",        r"$H_{T}$ (GeV)", 44, 240, 2000)
    lep_pt_bins =  hist.Bin('pt', r'$p_{T}\ (GeV)$', 45, 0, 450)
    lead_lep_pt_bins =  hist.Bin('pt', r'$p_{T}\ (GeV)$', 60, 0, 600)
    onZ_pt_bins =  hist.Bin('pt', r'$p_{T}\ (GeV)$', 40, 0, 600)
    jet_pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 40, 0, 600)
    phi_bins = hist.Bin('phi', r'$p_{T}\ (GeV)$', 12, -3, 3)
    
    my_labels = {
        'topW_v2': 'top-W scat.',
        'topW_v3': 'top-W scat.',
        'TTZ': r'$t\bar{t}Z$',
        'TTXnoW': r'$t\bar{t}Z/H$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'ttbar': r'$t\bar{t}$',
        'DY': 'Drell-Yan',
        'WW': 'WW',
        'WZ': 'WZ',
    }
    
    my_colors = {
        'topW_v2': '#FF595E',
        'topW_v3': '#FF595E',
        'TTZ': '#FFCA3A',
        'TTXnoW': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'ttbar': '#1982C4',
        'DY': '#6A4C93',
        'WW': '#34623F',
        'WZ': '#525B76',
    }
    TFnormalize = True
    version_dir = '/2018_Nb_weights/'


    makePlot(output, 'N_b', 'multiplicity',
        data=['DoubleMuon', 'MuonEG', 'EGamma'],
        bins=N_bins_red, log=False, normalize=TFnormalize, axis_label=r'$N_{b-tag}$',
        new_colors=my_colors, new_labels=my_labels,
        order=['topW_v3', 'diboson', 'TTW', 'TTXnoW', 'DY', 'ttbar'],
        upHists=['centralUp', 'upCentral'], downHists=['centralDown', 'downCentral'],
        shape=False,
        save=os.path.expandvars(plot_dir+version_dir+'N_b'),
        )

    
