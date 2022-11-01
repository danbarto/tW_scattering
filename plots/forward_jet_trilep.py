import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist, util
from coffea.processor import accumulate

import numpy as np
import copy

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

    order = ['topW_lep', 'diboson', 'TTW', 'TTH', 'TTZ', 'DY', 'top', 'XG', 'rare']
    #order = ['topW', 'diboson', 'TTW', 'TTH', 'TTZ', 'top', 'XG']
    #order = ['topW', 'DY']

    datasets = data + order

    outputs = []

    try:
        lumi_year = int(year)
    except:
        lumi_year = year
    lumi = cfg['lumi'][lumi_year]

    output = get_merged_output("trilep_analysis", year=year)


    #if year == '2019':
    #    # load the results
    #    lumi = 35.9+41.5+60.0
    #    first = True
    #    for y in ['2016', '2016APV', '2017', '2018']:
    #        cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), 'forward_OS_%s'%y), serialized=True)
    #        cache.load()
    #        tmp_output = cache.get('simple_output')
    #        if first:
    #            output = copy.deepcopy(tmp_output)
    #        else:
    #            for key in tmp_output:
    #                if type(tmp_output[key]) == hist.hist_tools.Hist:
    #                    try:
    #                        output[key].add(tmp_output[key])
    #                    except KeyError:
    #                        print ("Key %s not present in all years. Skipping."%key)
    #        first = False
    #        del cache

    #else:
    #    cacheName = 'forward_OS_%s'%year
    #    if small: cacheName += '_small'
    #    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)

    #    cache.load()
    #    output = cache.get('simple_output')

    #    lumi        = cfg['lumi'][(int(year) if year != '2016APV' else year)]

    #plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'trilep/v0.7.0_v1/')
    plot_dir    = os.path.join("/home/daniel/TTW/tW_scattering/plots/images/", str(year), 'trilep', args.version)
    if args.postfix:
        plot_dir += '_%s'%args.postfix
    if TFnormalize:
        plot_dir += '/normalized/'

    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    N_bins_ele = hist.Bin('n_ele', r'$N$', 4, -0.5, 3.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    mass_bins_fine = hist.Bin('mass', r'$M\ (GeV)$', 20, 80, 100)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
    mjf_bins = hist.Bin('mass', r'$M\ (GeV)$', 50, 0, 2000)
    #mjf_bins = None
    deltaEta_bins = hist.Bin('eta', r'$\eta $', 20, 0, 10)
    #deltaEta_bins = None
    jet_pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 50, 0, 500)
    m3l_bins = hist.Bin('mass', r'$M\ (GeV)$', 15, 0, 300)
    m3l_bins_fine = hist.Bin('mass', r'$M\ (GeV)$', 20, 70, 110)
#ext_mass_axis           = hist.Bin("mass",          r"M (GeV)",         100, 0, 2000)  # for any other mass

    my_labels = {
        'topW_lep': 'top-W scat.',
        'topW_EFT_cp8': 'EFT, cp8',
        'topW_EFT_mix': 'EFT mix',
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
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
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

    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    TFnormalize = False
    for log in True, False:

        plot_dir_temp = plot_dir + "/log/" if log else plot_dir + "/lin/"

        makePlot(output, 'N_jet', 'multiplicity',
                data=data,
                bins=N_bins, log=log, normalize=TFnormalize, axis_label=r'$N_{jet}$',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                omit=omit,
                signals=signals,
                lumi=lumi,
                save=os.path.expandvars(plot_dir_temp+'N_jet'),
                )

        makePlot(output, 'N_b', 'multiplicity',
                data=data,
                bins=N_bins, log=log, normalize=TFnormalize, axis_label=r'$N_{b}$',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                omit=omit,
                signals=signals,
                lumi=lumi,
                save=os.path.expandvars(plot_dir_temp+'N_b'),
                )

        makePlot(output, 'N_fwd', 'multiplicity',
                data=data,
                bins=N_bins, log=log, normalize=TFnormalize, axis_label=r'$N_{fwd}$',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                omit=omit,
                signals=signals,
                lumi=lumi,
                save=os.path.expandvars(plot_dir_temp+'N_fwd'),
                )

        makePlot(output, 'M3l', 'mass',
                data=data,
                bins=m3l_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell\ell}$ (GeV)',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                lumi=lumi_year,
                signals=signals,
                save=os.path.expandvars(plot_dir_temp+'M3l'),
            )

        #for ch in ['mm', 'em', 'ee']:
        #    makePlot(output, 'M3l_offZ', 'mass',
        #             data=data,
        #             bins=m3l_bins_fine, log=log, normalize=TFnormalize, axis_label=r'$M3l$ (GeV)',
        #             new_colors=my_colors, new_labels=my_labels,
        #             order=order,
        #             lumi=lumi_year,
        #             signals=signals,
        #             overflowclip=True,
        #             channel=ch,
        #             save=os.path.expandvars(plot_dir_temp+'M3l_offZ_'+ch),
        #        )


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

        makePlot(output, 'dilep_pt', 'pt',
                data=data,
                bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}(\ell\ell)$ (GeV)',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                signals=signals,
                lumi=lumi_year,
                save=os.path.expandvars(plot_dir_temp+'dilep_pt'),
            )

        makePlot(output, 'dilep_eta', 'eta',
                data=data,
                bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta(\ell\ell)$',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                signals=signals,
                lumi=lumi_year,
                save=os.path.expandvars(plot_dir_temp+'dilep_eta'),
            )

        makePlot(output, 'dilep_mass', 'mass',
                data=data,
                bins=mass_bins_fine, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
                new_colors=my_colors, new_labels=my_labels,
                order=order,
                signals=signals,
                lumi=lumi_year,
                save=os.path.expandvars(plot_dir_temp+'dilep_mass'),
            )

        makePlot(output, 'lead_lep', 'pt',
            data=data,
            bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,
            save=os.path.expandvars(plot_dir_temp+'lead_lep_pt'),
            )

        makePlot(output, 'lead_lep', 'eta',
            data=data,
            bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ lead \ lep$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'lead_lep_eta'),
            )

        makePlot(output, 'trail_lep', 'pt',
            data=data,
            bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ trail \ lep\ (GeV)$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'trail_lep_pt'),
            )

        makePlot(output, 'trail_lep', 'eta',
            data=data,
            bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ trail \ lep$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'trail_lep_eta'),
            )

        makePlot(output, 'second_lep', 'pt',
            data=data,
            bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ second \ lep\ (GeV)$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'second_lep_pt'),
            )

        makePlot(output, 'second_lep', 'eta',
            data=data,
            bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ second \ lep$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'second_lep_eta'),
            )

        makePlot(output, 'MET', 'pt',
            data=data,
            bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}\ (GeV)$',
            new_colors=my_colors, new_labels=my_labels,
            order=order,lumi=lumi_year,
            save=os.path.expandvars(plot_dir_temp+'MET_pt'),
            )

    raise NotImplementedError

    makePlot(output, 'PV_npvsGood', 'multiplicity',
        data=data,
        bins=None, log=log, normalize=TFnormalize, axis_label=r'$N_{PV}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        save=os.path.expandvars(plot_dir_temp+version_dir+'PV_npvsGood'),
        )

    makePlot(output, 'N_fwd', 'multiplicity',
        data=data,
        bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{fwd\ jets}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_fwd'),
        )

    makePlot(output, 'N_fwd', 'multiplicity',
        data=data,
        bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{fwd\ jets}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_fwd_stat'),
        )

    makePlot(output, 'fwd_jet', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ selected\ fwd\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'fwd_jet_pt'),
        )

    makePlot(output, 'fwd_jet', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ selected\ fwd\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'fwd_jet_eta'),
        )

    makePlot(output, 'fwd_jet', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi\ selected\ fwd\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'fwd_jet_phi'),
        )

    makePlot(output, 'N_jet', 'multiplicity',
        data=data,
        bins=N_bins, log=log, normalize=TFnormalize, axis_label=r'$N_{jet}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_jet'),
        )

    makePlot(output, 'j1', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j1_pt'),
        )

    makePlot(output, 'j1', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j1_eta'),
        )

    makePlot(output, 'j1', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j1_phi'),
        )
   
    makePlot(output, 'j2', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j2_pt'),
        )

    makePlot(output, 'j2', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j2_eta'),
        )

    makePlot(output, 'j2', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j2_phi'),
        )
    
    makePlot(output, 'j3', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j3_pt'),
        )

    makePlot(output, 'j3', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j3_eta'),
        )

    makePlot(output, 'j3', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'j3_phi'),
        )
        

    makePlot(output, 'N_b', 'multiplicity',
        data=data,
        bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{b-tag}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_b'),
        )

    makePlot(output, 'N_central', 'multiplicity',
        data=data,
        bins=N_bins_red_central, log=log, normalize=TFnormalize, axis_label=r'$N_{central\ jet}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order, lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'], 
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_central'),
        )
    

    makePlot(output, 'b1', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'b1_pt'),
        )

    makePlot(output, 'b1', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'b1_eta'),
        )

    makePlot(output, 'b1', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi\ leading\ b-jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'b1_phi'),
        )
        

    makePlot(output, 'MET', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}^{miss}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'MET_pt'),
        )

    makePlot(output, 'MET', 'phi',
        data=data,
        bins=phi_bins, log=log, normalize=TFnormalize, axis_label=r'$\phi(p_{T}^{miss})$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        save=os.path.expandvars(plot_dir_temp+version_dir+'MET_phi'),
        )
    
    makePlot(output, 'onZ_pt', 'pt',
        data=data,
        bins=onZ_pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ onZ$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'onZ_pair_pt'),
        )
    
    makePlot(output, 'HT', 'ht',
        data=data,
        bins=ht_bins, log=log, normalize=TFnormalize, axis_label=r'$H_{T}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'HT'),
        )
    
    makePlot(output, 'ST', 'ht',
        data=data,
        bins=st_bins, log=log, normalize=TFnormalize, axis_label=r'$H_{T}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'ST'),
        )
    
    makePlot(output, 'LT', 'ht',
        data=data,
        bins=lt_bins, log=log, normalize=TFnormalize, axis_label=r'$L_{T}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'LT'),
        )

    makePlot(output, 'M3l', 'mass',
        data=data,
        bins=m3l_bins, log=log, normalize=TFnormalize, axis_label=r'$M3l$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order, lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'M3l'),
        )
    
    makePlot(output, 'M_ll', 'mass',
        data=data,
        bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'M_ll'),
        )
    '''makePlot(output, 'M_ll_all', 'mass',
        data=data,
        bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'M_ll_all'),
        )
    makePlot(output, 'M_ll_worst', 'mass',
        data=data,
        bins=mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M_{\ell\ell}$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'M_ll_worst'),
        )'''
    
    makePlot(output, 'min_mass_SFOS', 'mass',
        data=data,
        bins=min_mass_bins, log=log, normalize=TFnormalize, axis_label=r'$M\ (GeV)$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'min_mass_SFOS'),
        )
    
    makePlot(output, 'deltaEta', 'eta',
        data=data,
        bins=deltaEta_bins, log=log, normalize=TFnormalize, axis_label=r'$\delta \eta $(GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'deltaEta'),
        )
    
    makePlot(output, 'mjf_max', 'mass',
        data=data,
        bins=mjf_bins, log=True, normalize=TFnormalize, axis_label='mjf_max (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        save=os.path.expandvars(plot_dir_temp+version_dir+'mjf_max'),
        )
    
    makePlot(output, 'min_bl_dR', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label='min_bl_dR (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'min_bl_dR'),
        )
    
    makePlot(output, 'min_mt_lep_met', 'pt',
        data=data,
        bins=pt_bins, log=True, normalize=TFnormalize, axis_label='min_mt_lep_met (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'min_mt_lep_met'),
        )
    
    makePlot(output, 'leading_jet_pt', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'leading_jet_pt'),
        )
    
    makePlot(output, 'leading_jet_eta', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading \ btag$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'leading_jet_eta'),
        )
    
    makePlot(output, 'subleading_jet_pt', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ subleading\ jet$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'subleading_jet_pt'),
        )
    
    makePlot(output, 'subleading_jet_eta', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ subleading \ jet$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'subleading_jet_eta'),
        )
    
    makePlot(output, 'leading_btag_pt', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ leading\ btag$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'leading_btag_pt'),
        )
    
    makePlot(output, 'subleading_btag_pt', 'pt',
        data=data,
        bins=pt_bins, log=log, normalize=TFnormalize, axis_label=r'$p_{T}\ subleading\ btag$ (GeV)',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'subleading_btag_pt'),
        )
    
    makePlot(output, 'leading_btag_eta', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ leading \ btag$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'leading_btag_eta'),
        )
    
    makePlot(output, 'subleading_btag_eta', 'eta',
        data=data,
        bins=eta_bins, log=log, normalize=TFnormalize, axis_label=r'$\eta\ subleading \ btag$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'subleading_btag_eta'),
        )
        

    makePlot(output, 'N_ele', 'multiplicity',
        data=data,
        bins=N_bins_red, log=log, normalize=TFnormalize, axis_label=r'$N_{ele}$',
        new_colors=my_colors, new_labels=my_labels,
        order=order,lumi=lumi_year,
        #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
        save=os.path.expandvars(plot_dir_temp+version_dir+'N_ele'),
        )
