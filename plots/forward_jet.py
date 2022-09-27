
import os
import awkward as ak

from coffea import processor, hist, util
from coffea.processor import accumulate
import copy
import numpy as np

from Tools.config_helpers import loadConfig, load_yaml, data_path, get_latest_output
from Tools.config_helpers import get_merged_output
from Tools.helpers import get_samples

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, scale_and_merge

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--normalize', action='store_true', default=None, help="Normalize?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--version', action='store', default='v21', help="Version of the NN training. Just changes subdir.")
    argParser.add_argument('--DY', action='store_true', help="DY")
    argParser.add_argument('--postfix', action='store', default='', help="postfix for plot directory")
    args = argParser.parse_args()

    small       = args.small
    verysmall   = args.verysmall
    if verysmall:
        small = True
    TFnormalize = args.normalize
    year        = args.year
    cfg         = loadConfig()

    #year = int(args.year)
    era = ''
    ul = f"UL{str(year)[2:]}{era}"
    #cfg = loadConfig()

    samples = get_samples(f"samples_{ul}.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    renorm   = {}

    if year == '2018':
        data = ['SingleMuon', 'DoubleMuon', 'EGamma', 'MuonEG']
    else:
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron']
    order = ['topW_lep', 'diboson', 'rare', 'TTW', 'TTH', 'TTZ', 'DY', 'top', 'XG']

    datasets = data + order

    try:
        lumi_year = int(year)
    except:
        lumi_year = year
    lumi = cfg['lumi'][lumi_year]

    postfix = '_DY' if args.DY else None
    output = get_merged_output("OS_analysis", year=year, postfix=postfix)

    #plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'OS', args.version)
    plot_dir    = os.path.join("/home/daniel/TTW/tW_scattering/plots/images/", str(year), 'OS', args.version)

    if args.DY: plot_dir = plot_dir.replace('OS', 'DY')
    if args.postfix:
        plot_dir += '_%s'%args.postfix
    if TFnormalize:
        plot_dir += '/normalized/'

    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    N_bins_ele = hist.Bin('n_ele', r'$N$', 4, -0.5, 3.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    delta_phi_bins = hist.Bin('delta_phi', r'$\Delta\phi$', 32, 0, 3.2)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
    mjf_bins = hist.Bin('mass', r'$M\ (GeV)$', 50, 0, 2000)
    deltaEta_bins = hist.Bin('eta', r'$\eta $', 20, 0, 10)
    jet_pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 50, 0, 500)
    
    my_labels = {
        'topW_lep': 'top-W scat.',
        'TTZ': r'$t\bar{t}Z$',
        'TTXnoW': r'$t\bar{t}X\ (no\ W)$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'rare': 'rare',
        'top': r'$t\bar{t}$',
        'XG': 'XG',  # this is bare XG
        'DY': 'Drell-Yan',  # this is bare XG
        'conv_mc': 'conversion',
        'np_obs_mc': 'nonprompt (MC true)',
        'np_est_mc': 'nonprompt (MC est)',
        'cf_obs_mc': 'charge flip (MC true)',
        'cf_est_mc': 'charge flip (MC est)',
        'np_est_data': 'nonprompt (est)',
        'cf_est_data': 'charge flip (est)',
    }

    my_colors = {
        'topW_lep': '#FF595E',
        'TTZ': '#FFCA3A',
        'TTXnoW': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'rare': '#EE82EE',
        'top': '#1982C4',
        'XG': '#5bc0de',
        'conv_mc': '#5bc0de',
        'DY': '#6A4C93',
        'np_obs_mc': '#1982C4',
        'np_est_mc': '#1982C4',
        'np_est_data': '#1982C4',
        'cf_obs_mc': '#0F7173',
        'cf_est_mc': '#0F7173',
        'cf_est_data': '#0F7173',
    }

    all_processes = [ x[0] for x in output['N_jet'].values().keys() ]
    #data = ['MuonEG','DoubleMuon','EGamma','SingleElectron','SingleMuon', 'DoubleEG']
    #order = ['topW_v3', 'diboson', 'TTW', 'TTXnoW', 'DY', 'ttbar', 'XG']
    #order = ['topW_v3', 'diboson', 'TTXnoW', 'DY', 'ttbar', 'XG']
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    scale = {'DY': 1.5}

    for log in True, False:

        plot_dir_temp = plot_dir + "/log/" if log else plot_dir + "/lin/"

        makePlot(output, 'lead_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_pt')
                 )

        makePlot(output, 'lead_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='mm',
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_pt_mm')
                 )

        makePlot(output, 'lead_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='em',
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_pt_em')
                 )

        makePlot(output, 'lead_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='ee',
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_pt_ee')
                 )

        makePlot(output, 'lead_lep', 'eta',
                 data=data,
                 bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ lead \ lep$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_eta')
                 )

        #makePlot(output, 'lead_lep', 'phi',
        #         data=data,
        #         bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi\ lead \ lep$',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         save=os.path.expandvars(plot_dir_temp+'lead_lep_phi'),
        #         )

        makePlot(output, 'trail_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ trail \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'trail_lep_pt'),
                 )

        makePlot(output, 'trail_lep', 'eta',
                 data=data,
                 bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ trail \ lep$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'trail_lep_eta'),
                 )

        #makePlot(output, 'trail_lep', 'phi',
        #         data=data,
        #         bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi\ trail \ lep$',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         save=os.path.expandvars(plot_dir_temp+'trail_lep_phi'),
        #         )


        makePlot(output, 'PV_npvsGood', 'multiplicity',
                 data=data,
                 bins=None, log=log, normalize=TFnormalize, axis_label=r'$N_{PV}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'PV_npvsGood'),
                 )

        makePlot(output, 'N_fwd', 'multiplicity',
                 data=data,
                 bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{fwd\ jets}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 save=os.path.expandvars(plot_dir_temp+'N_fwd'),
                 )

        makePlot(output, 'N_fwd', 'multiplicity',
                 data=data,
                 bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{fwd\ jets}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'N_fwd_stat'),
                 )

        if not args.DY:
            makePlot(output, 'fwd_jet', 'pt',
                     data=data,
                     bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ selected\ fwd\ jet$ (GeV)',
                     new_colors=my_colors, new_labels=my_labels,
                     order=order,
                     omit=omit,
                     signals=signals,
                     lumi=lumi,
                     upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                     #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                     save=os.path.expandvars(plot_dir_temp+'fwd_jet_pt'),
                     )

            makePlot(output, 'fwd_jet', 'eta',
                     data=data,
                     bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ selected\ fwd\ jet$ (GeV)',
                     new_colors=my_colors, new_labels=my_labels,
                     order=order,
                     omit=omit,
                     signals=signals,
                     lumi=lumi,
                     upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                     #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                     save=os.path.expandvars(plot_dir_temp+'fwd_jet_eta'),
                     )

            #makePlot(output, 'fwd_jet', 'phi',
            #         data=data,
            #         bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi\ selected\ fwd\ jet$ (GeV)',
            #         new_colors=my_colors, new_labels=my_labels,
            #         order=order,
            #         omit=omit,
            #         signals=signals,
            #         lumi=lumi,
            #         #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
            #         save=os.path.expandvars(plot_dir_temp+'fwd_jet_phi'),
            #         )

        makePlot(output, 'N_jet', 'multiplicity',
                 data=data,
                 bins=N_bins, log=log, normalize=TFnormalize, axis_label=r'$N_{jet}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 save=os.path.expandvars(plot_dir_temp+'N_jet'),
                 )

        makePlot(output, 'lead_jet', 'pt',
                 data=data,
                 bins=jet_pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ jet$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'lead_jet_pt'),
                 )

        makePlot(output, 'lead_jet', 'eta',
                 data=data,
                 bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ jet$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'lead_jet_eta'),
                 )

        #makePlot(output, 'j1', 'phi',
        #         data=data,
        #         bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ jet$ (GeV)',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        #         save=os.path.expandvars(plot_dir_temp+'j1_phi'),
        #         )

        makePlot(output, 'sublead_jet', 'pt',
                 data=data,
                 bins=jet_pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ subleading\ jet$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'sublead_jet_pt'),
                 )

        makePlot(output, 'sublead_jet', 'eta',
                 data=data,
                 bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ subleading\ jet$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 save=os.path.expandvars(plot_dir_temp+'sublead_jet_eta'),
                 )

        makePlot(output, 'N_b', 'multiplicity',
                 data=data,
                 bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{b-tag}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 save=os.path.expandvars(plot_dir_temp+'N_b'),
                 )

        #makePlot(output, 'N_central', 'multiplicity',
        #         data=data,
        #         bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{central\ jet}$',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        #         save=os.path.expandvars(plot_dir_temp+'N_central'),
        #         )

        makePlot(output, 'N_jet', 'n_ele',
                 data=data,
                 bins=N_bins_ele, log=log, normalize=TFnormalize, axis_label=r'$N_{ele}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'N_ele'),
                 )


        '''
        makePlot(output, 'b1', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,
        omit=omit,
        signals=signals,
        lumi=lumi,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+'b1_pt'),
        )
        makePlot(output, 'b1', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,
        omit=omit,
        signals=signals,
        lumi=lumi,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+'b1_eta'),
        )
        makePlot(output, 'b1', 'phi',
        data=data,
        bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,
        omit=omit,
        signals=signals,
        lumi=lumi,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+'b1_phi'),
        )
        '''

        makePlot(output, 'MET', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['jes_up'], downHists=['jes_down'],
                 channel='ee',
                 rescale=scale,
                 save=os.path.expandvars(plot_dir_temp+'MET_pt_ee'),
                 )

        makePlot(output, 'MET', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['jes_up'], downHists=['jes_down'],
                 channel='em',
                 rescale=scale,
                 save=os.path.expandvars(plot_dir_temp+'MET_pt_em'),
                 )

        makePlot(output, 'MET', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['jes_up'], downHists=['jes_down'],
                 channel='mm',
                 rescale=scale,
                 save=os.path.expandvars(plot_dir_temp+'MET_pt_mm'),
                 )

        makePlot(output, 'MET', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 rescale=scale,
                 upHists=['jes_up', 'PU_up', 'b_up', 'l_up'], downHists=['jes_down', 'PU_down', 'b_down', 'l_down'],
                 #upHists=['jes_up'], downHists=['jes_down'],
                 save=os.path.expandvars(plot_dir_temp+'MET_pt'),
                 )

        makePlot(output, 'MET', 'phi',
                 data=data,
                 bins=None, log=log, normalize=TFnormalize, axis_label=r'$\phi(p_{T}^{miss})$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'MET_phi'),
                 )

        makePlot(output, 'N_tau', 'multiplicity',
                 data=data,
                 bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{tau\ }$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'N_tau'),
                 )

        #makePlot(output, 'N_track', 'multiplicity',
        #         data=data,
        #         bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{track\ }$',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        #         save=os.path.expandvars(plot_dir_temp+'N_track'),
        #         )

        makePlot(output, 'dilep_pt', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ dilep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'dilep_pt'),
                 )


        makePlot(output, 'dilep_mass', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='ee',
                 save=os.path.expandvars(plot_dir_temp+'dilep_mass_ee'),
                 )

        makePlot(output, 'dilep_mass', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='mm',
                 save=os.path.expandvars(plot_dir_temp+'dilep_mass_mm'),
                 )

        makePlot(output, 'dilep_mass', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='em',
                 save=os.path.expandvars(plot_dir_temp+'dilep_mass_em'),
                 )

        makePlot(output, 'dilep_mass', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'dilep_mass'),
                 )

        if not args.DY:
            makePlot(output, 'deltaEta', 'eta',
                     data=data,
                     bins=deltaEta_bins, log=log, normalize=TFnormalize, axis_label=r'$\Delta \eta $(GeV)',
                     new_colors=my_colors, new_labels=my_labels,
                     order=order,
                     omit=omit,
                     signals=signals,
                     lumi=lumi,
                     #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                     save=os.path.expandvars(plot_dir_temp+'deltaEta'),
                     )
        makePlot(output, 'dilep_mass_high_mt', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'dilep_mass_high_mt'),
                 )
        for ch in ['ee','mm','em']:
            makePlot(output, 'dilep_mass_high_mt', 'mass',
                     data=data,
                     bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                     new_colors=my_colors, new_labels=my_labels,
                     order=order,
                     omit=omit,
                     signals=signals,
                     lumi=lumi,
                     channel=ch,
                     save=os.path.expandvars(plot_dir_temp+'dilep_mass_high_mt_'+ch),
                     )
            makePlot(output, 'min_mt_lep_met', 'pt',
                     data=data,
                     bins=pt_bins, log=log, normalize=TFnormalize, axis_label='min_mt_lep_met (GeV)',
                     new_colors=my_colors, new_labels=my_labels,
                     order=order,
                     omit=omit,
                     signals=signals,
                     lumi=lumi,
                     channel=ch,
                     #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                     save=os.path.expandvars(plot_dir_temp+'min_mt_lep_met_'+ch),
                     )


        #makePlot(output, 'mjf_max', 'mass',
        #         data=data,
        #         bins=mjf_bins, log=log, normalize=TFnormalize, axis_label='mjf_max (GeV)',
        #         new_colors=my_colors, new_labels=my_labels,
        #         order=order,
        #         omit=omit,
        #         signals=signals,
        #         lumi=lumi,
        #         save=os.path.expandvars(plot_dir_temp+'mjf_max'),
        #         )

        makePlot(output, 'min_bl_dR', 'eta',
                 data=data,
                 bins=eta_bins, log=log, normalize=TFnormalize, axis_label='min_bl_dR (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'min_bl_dR'),
                 )

        makePlot(output, 'min_mt_lep_met', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=TFnormalize, axis_label='min_mt_lep_met (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'min_mt_lep_met'),
                 )

        makePlot(output, 'delta_phi_lep_met', 'delta_phi',
                 data=data,
                 bins=delta_phi_bins, log=log, normalize=TFnormalize, axis_label='delta_phi_lep_met',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'delta_phi_lep_met'),
                 )
