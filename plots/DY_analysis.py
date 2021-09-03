import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
import copy
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

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--version', action='store', default='v21', help="Version of the NN training. Just changes subdir.")
    args = argParser.parse_args()

    small       = args.small
    verysmall   = args.verysmall
    if verysmall:
        small = True
    year        = args.year
    cfg         = loadConfig()

    if year == '2019':
        # load the results
        lumi = 35.9+41.5+60.0
        first = True
        for y in ['2016', '2016APV', '2017', '2018']:
            cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), 'DY_analysis_%s'%y), serialized=True)
            cache.load()
            tmp_output = cache.get('simple_output')
            if first:
                output = copy.deepcopy(tmp_output)
            else:
                for key in tmp_output:
                    if type(tmp_output[key]) == hist.hist_tools.Hist:
                        try:
                            output[key].add(tmp_output[key])
                        except KeyError:
                            print ("Key %s not present in all years. Skipping."%key)
            first = False
            del cache

    else:
        cacheName = 'DY_analysis_%s'%year
        if small: cacheName += '_small'
        cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)

        cache.load()
        output = cache.get('simple_output')

        lumi        = cfg['lumi'][(int(year) if year != '2016APV' else year)]

    plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'DY/%s_v21/'%cfg['meta']['version'])
    
    NN = len(output['node0_score_incl'].values().keys())>0
    
    # defining some new axes for rebinning.
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    pt_bins_coarse_red = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 100)
    pt_bins_ext = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 1000)
    ht_bins = hist.Bin('ht', r'$H_{T}\ (GeV)$', 10, 0, 1000)
    ht_bins_red = hist.Bin('ht', r'$p_{T}\ (GeV)$', 7,100,800)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 8, 0, 1)  # FIXME update to 8
 

    my_labels = {
        'topW_v3': 'top-W scat.',
        'topW_EFT_cp8': 'EFT, cp8',
        'topW_EFT_mix': 'EFT mix',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'rare': 'rare',
        'ttbar': r'$t\bar{t}$',
        'XG': 'XG',  # this is bare XG
        'conv_mc': 'conversion',
        'np_obs_mc': 'nonprompt (MC true)',
        'np_est_mc': 'nonprompt (MC est)',
        'cf_obs_mc': 'charge flip (MC true)',
        'cf_est_mc': 'charge flip (MC est)',
        'np_est_data': 'nonprompt (est)',
        'cf_est_data': 'charge flip (est)',
        'DY': 'Drell-Yan',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'rare': '#EE82EE',
        'ttbar': '#1982C4',
        'XG': '#5bc0de',
        'conv_mc': '#5bc0de',
        'np_obs_mc': '#1982C4',
        'np_est_mc': '#1982C4',
        'np_est_data': '#1982C4',
        'cf_obs_mc': '#0F7173',
        'cf_est_mc': '#0F7173',
        'cf_est_data': '#0F7173',
        'DY': '#6A4C93',
    }


    #for k in my_labels.keys():

    ## DATA DRIVEN BKG ESTIMATES


    all_processes = [ x[0] for x in output['N_ele'].values().keys() ]

    if year == '2018':
        data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
    else:
        data_all = ['DoubleMuon', 'MuonEG', 'DoubleEG', 'SingleElectron', 'SingleMuon']



    sub_dir = '/'

    data    = data_all
    #order   = ['np_est_data', 'XG', 'cf_est_data', 'TTW', 'TTH', 'TTZ','rare', 'diboson', 'topW_v3']
    order   = ['topW_v3', 'ttbar', 'XG', 'TTW', 'TTH', 'TTZ','rare', 'diboson', 'DY']
    #order   = ['np_est_data', 'conv_mc', 'cf_est_data', 'TTW', 'TTH', 'TTZ','rare', 'diboson', 'topW_v3']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    makePlot(output, 'MET', 'pt',
         data=data,
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'MET_pt'),
        )

    makePlot(output, 'fwd_jet', 'pt',
         data=data,
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'fwd_jet_pt'),
        )

    makePlot(output, 'N_ele', 'multiplicity',
         data=data,
         bins=N_bins_red, log=False, normalize=True, axis_label=r'$N_{e}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'N_ele'),
        )

    makePlot(output, 'N_central', 'multiplicity',
         data=data,
         bins=N_bins_red, log=False, normalize=True, axis_label=r'$N_{jet,\ central}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'N_central'),
        )

    makePlot(output, 'N_b', 'multiplicity',
         data=data,
         bins=N_bins_red, log=False, normalize=True, axis_label=r'$N_{b-tag}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'N_b'),
        )

    makePlot(output, 'N_jet', 'multiplicity',
         data=data,
         bins=N_bins_red, log=False, normalize=True, axis_label=r'$N_{jet,\ all}$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'N_jet'),
        )

    makePlot(output, 'j1', 'pt',
         data=data,
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_T(j_1)\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'j1_pt'),
        )

    makePlot(output, 'j1', 'eta',
         data=data,
         bins=eta_bins, log=False, normalize=True, axis_label=r'$\eta(j_1)$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'j1_eta'),
        )

    makePlot(output, 'j2', 'pt',
         data=data,
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_T(j_2)\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'j2_pt'),
        )

    makePlot(output, 'j2', 'eta',
         data=data,
         bins=eta_bins, log=False, normalize=True, axis_label=r'$\eta(j_2)$',
         new_colors=my_colors, new_labels=my_labels, lumi=lumi,
         order=order,
         signals=signals,
         omit=omit,
         save=os.path.expandvars(plot_dir+sub_dir+'j2_eta'),
        )
