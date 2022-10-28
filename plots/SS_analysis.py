import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
import copy
import numpy as np
import re

from Tools.config_helpers import loadConfig, make_small

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, make_plot_from_dict

from Tools.config_helpers import get_merged_output

def get_histograms(output, histograms, total=False):
    '''
    output - e.g. output['MET'] selected histogram
    histograms - list of histograms to get, e.g. ['TTW', 'np_est_data']

    keep systematic axes and kinematic axes, integrate/project away all others
    '''
    wildcard = re.compile('.')
    out = {}
    # remove EFT axes that's not always there
    if 'EFT' in output.axes():
        tmp = output.integrate('EFT', 'central').copy()
    else:
        tmp = output.copy()

    all_axes = tmp.axes()
    for hist in histograms:
        # find whether the histogram is in predicitions or datasets
        for i, ax in enumerate(all_axes):
            ax_name = ax.name
            if hist in [x.name for x in ax.identifiers()]:
                break

        if ax_name == 'prediction':
            out[hist] = tmp[(wildcard, hist)].sum('dataset', 'prediction')
        elif ax_name == 'dataset':
            out[hist] = tmp[(hist, 'central')].sum('dataset', 'prediction')

    if total:
        tmp = out[list(histograms)[0]].copy()
        tmp.clear()
        for hist in histograms:
            tmp.add(out[hist])
        return tmp
    else:
        return out

def get_standard_plot(
        output,
        hist,
        axis,
        name,
        log=False,
        lumi=1,
        blind=False,
        overflow = 'over',
        systematics = True,
):
    mc = get_histograms(output[hist], ['topW_lep', 'rare', 'diboson', 'TTW', 'conv_mc', 'TTZ', 'TTH', 'cf_est_data', 'np_est_data'])
    data_options = ['DoubleMuon', 'SingleMuon', 'MuonEG', 'EGamma', 'DoubleEG', 'SingleElectron']
    identifiers = [x.name for x in output[hist].axes()[0].identifiers()]
    datasets = []
    for d in data_options:
        if d in identifiers:
            datasets.append(d)

    data = get_histograms(output[hist], datasets, total=True)

    make_plot_from_dict(
        mc, axis = axis,
        data = data.integrate('systematic', 'central') if not blind else None,
        save = plot_dir+sub_dir+name,
        overflow = overflow,
        log = log,
        lumi = lumi,
        systematics = systematics,
    )

def get_nonprompt_plot(output, hist, axis, name, log=False, overflow='over'):
    mc = get_histograms(output[hist], ['np_est_mc'])
    data = get_histograms(output[hist], ['np_obs_mc'], total=True)

    make_plot_from_dict(
        mc, axis = axis,
        data = data.integrate('systematic', 'central'),
        save = plot_dir+sub_dir+name,
        overflow = overflow,
        log = log,
        data_label = 'Nonprompt observed',
#        systematics = False,
    )

def get_chargeflip_plot(output, hist, axis, name, log=False, overflow='over'):
    mc = get_histograms(output[hist], ['cf_est_mc'])
    data = get_histograms(output[hist], ['cf_obs_mc'], total=True)

    make_plot_from_dict(
        mc, axis = axis,
        data = data.integrate('systematic', 'central'),
        save = plot_dir+sub_dir+name,
        overflow = overflow,
        log = log,
        data_label = 'Charge flip obs',
#        systematics = False,
    )


if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--normalize', action='store_true', default=None, help="Normalize?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--version', action='store', default='v21', help="Version of the NN training. Just changes subdir.")
    argParser.add_argument('--sample', action='store', default=None, help="Plot single sample")
    argParser.add_argument('--postfix', action='store', default='', help="postfix for plot directory")
    args = argParser.parse_args()

    small       = args.small
    verysmall   = args.verysmall
    if verysmall:
        small = True
    TFnormalize = args.normalize
    year        = args.year
    cfg         = loadConfig()

    if year == '2018':
        data = ['SingleMuon', 'DoubleMuon', 'EGamma', 'MuonEG']
    elif year == '2019':
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron', 'EGamma']
    else:
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron']
    order = ['topW_lep', 'diboson', 'rare', 'TTW', 'TTH', 'TTZ', 'top', 'XG']

    datasets = data + order

    try:
        lumi_year = int(year)
    except:
        lumi_year = year
    if year == '2019':
        lumi = sum([cfg['lumi'][y] for y in [2016, '2016APV', 2017, 2018]])
    else:
        lumi = cfg['lumi'][lumi_year]

    if year == '2019':
        outputs = []
        for y in ['2016', '2016APV', '2017', '2018']:
            outputs.append(get_merged_output("SS_analysis", year=y))
        output = accumulate(outputs)
        del outputs
    else:
        output = get_merged_output("SS_analysis", year=year, postfix='cpt_0_cpqm_0')

    #plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'OS', args.version)
    plot_dir    = os.path.join("/home/daniel/TTW/tW_scattering/plots/images/", str(year), 'SS', args.version)

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


    sub_dir = '/dd/'

    blind = True

    # Object multiplicities
    axis = hist.Bin('multiplicity', r'$N_{fwd\ jet}$', 5, -0.5, 4.5)
    get_standard_plot(output, 'N_fwd', axis, name='N_fwd', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'N_fwd', axis, name='np_N_fwd', log=False)
    get_chargeflip_plot(output, 'N_fwd', axis, name='cf_N_fwd', log=False)

    axis = hist.Bin('multiplicity', r'$N_{jet}$', 6, 3.5, 9.5)
    get_standard_plot(output, 'N_jet', axis, name='N_jet', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'N_jet', axis, name='np_N_jet', log=False)
    get_chargeflip_plot(output, 'N_jet', axis, name='cf_N_jet', log=False)

    axis = hist.Bin('multiplicity', r'$N_{central}$', 6, 2.5, 8.5)
    get_standard_plot(output, 'N_central', axis, name='N_central', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'N_central', axis, name='np_N_central', log=False)
    get_chargeflip_plot(output, 'N_central', axis, name='cf_N_central', log=False)

    axis = hist.Bin('multiplicity', r'$N_{b}$', 6, -0.5, 5.5)
    get_standard_plot(output, 'N_b', axis, name='N_b', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'N_b', axis, name='np_N_b', log=False)
    get_chargeflip_plot(output, 'N_b', axis, name='cf_N_b', log=False)

    axis = hist.Bin('multiplicity', r'$N_{e}$', 3, -0.5, 2.5)
    get_standard_plot(output, 'N_ele', axis, name='N_ele', log=False, lumi=lumi, blind=False, overflow='none')
    get_nonprompt_plot(output, 'N_ele', axis, name='np_N_ele', log=False, overflow='none')
    get_chargeflip_plot(output, 'N_ele', axis, name='cf_N_ele', log=False, overflow='none')

    axis = hist.Bin('multiplicity', r'$N_{\tau}$', 3, -0.5, 2.5)
    get_standard_plot(output, 'N_tau', axis, name='N_tau', log=False, lumi=lumi, blind=False, overflow='none')
    get_nonprompt_plot(output, 'N_tau', axis, name='np_N_tau', log=False, overflow='none')
    get_chargeflip_plot(output, 'N_tau', axis, name='cf_N_tau', log=False, overflow='none')

    # Event kinematics
    axis = hist.Bin('pt', r'$p_{T}^{miss}\ (GeV)$', 10, 0, 300)
    get_standard_plot(output, 'MET', axis, name='MET_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'MET', axis, name='np_MET_pt', log=False)
    get_chargeflip_plot(output, 'MET', axis, name='cf_MET_pt', log=False)

    axis = hist.Bin("ht", r"$L_{T}$ (GeV)", 25, 0, 1000)
    get_standard_plot(output, 'LT', axis, name='LT', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'LT', axis, name='np_LT', log=False)
    get_chargeflip_plot(output, 'LT', axis, name='cf_LT', log=False)

    axis = hist.Bin("ht", r"$H_{T}$ (GeV)", 25, 0, 1000)
    get_standard_plot(output, 'HT', axis, name='HT', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'HT', axis, name='np_HT', log=False)
    get_chargeflip_plot(output, 'HT', axis, name='cf_HT', log=False)

    axis = hist.Bin("ht", r"$S_{T}$ (GeV)", 25, 0, 1000)
    get_standard_plot(output, 'ST', axis, name='ST', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'ST', axis, name='np_ST', log=False)
    get_chargeflip_plot(output, 'ST', axis, name='cf_ST', log=False)

    # Forward jet
    axis = hist.Bin('pt', r'$p_{T}(fwd\ jet)\ (GeV)$', 10, 0, 300)
    get_standard_plot(output, 'fwd_jet', axis, name='fwd_jet_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'fwd_jet', axis, name='np_fwd_jet_pt', log=False)
    get_chargeflip_plot(output, 'fwd_jet', axis, name='cf_fwd_jet_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(fwd\ jet)$', 25, -5, 5)
    get_standard_plot(output, 'fwd_jet', axis, name='fwd_jet_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'fwd_jet', axis, name='np_fwd_jet_eta', log=False)
    get_chargeflip_plot(output, 'fwd_jet', axis, name='cf_fwd_jet_eta', log=False)

    axis = hist.Bin("p", r"$p$ (GeV)", 20, 0, 2000)
    get_standard_plot(output, 'fwd_jet', axis, name='fwd_jet_p', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'fwd_jet', axis, name='np_fwd_jet_p', log=False)
    get_chargeflip_plot(output, 'fwd_jet', axis, name='cf_fwd_jet_p', log=False)

    # leptons
    axis = hist.Bin('pt', r'$p_{T}(leading\ lep)\ (GeV)$', 10, 0, 300)
    get_standard_plot(output, 'lead_lep', axis, name='lead_lep_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'lead_lep', axis, name='np_lead_lep_pt', log=False)
    get_chargeflip_plot(output, 'lead_lep', axis, name='cf_lead_lep_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(leading\ lep)$', 25, -5, 5)
    get_standard_plot(output, 'lead_lep', axis, name='lead_lep_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'lead_lep', axis, name='np_lead_lep_eta', log=False)
    get_chargeflip_plot(output, 'lead_lep', axis, name='cf_lead_lep_eta', log=False)

    axis = hist.Bin('pt', r'$p_{T}(trailing\ lep)\ (GeV)$', 10, 0, 200)
    get_standard_plot(output, 'trail_lep', axis, name='trail_lep_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'trail_lep', axis, name='np_trail_lep_pt', log=False)
    get_chargeflip_plot(output, 'trail_lep', axis, name='cf_trail_lep_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(trailing\ lep)$', 25, -5, 5)
    get_standard_plot(output, 'trail_lep', axis, name='trail_lep_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'trail_lep', axis, name='np_trail_lep_eta', log=False)
    get_chargeflip_plot(output, 'trail_lep', axis, name='cf_trail_lep_eta', log=False)

    # jets
    axis = hist.Bin('pt', r'$p_{T}(leading\ jet)\ (GeV)$', 24, 20, 500)
    get_standard_plot(output, 'lead_jet', axis, name='lead_jet_pt', log=False, lumi=lumi, blind=blind, systematics=False)  # FIXME some syst currently broken
    get_nonprompt_plot(output, 'lead_jet', axis, name='np_lead_jet_pt', log=False)
    get_chargeflip_plot(output, 'lead_jet', axis, name='cf_lead_jet_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(leading\ jet)$', 25, -5, 5)
    get_standard_plot(output, 'lead_jet', axis, name='lead_jet_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'lead_jet', axis, name='np_lead_jet_eta', log=False)
    get_chargeflip_plot(output, 'lead_jet', axis, name='cf_lead_jet_eta', log=False)

    axis = hist.Bin('pt', r'$p_{T}(subleading\ jet)\ (GeV)$', 23, 20, 250)
    get_standard_plot(output, 'sublead_jet', axis, name='sublead_jet_pt', log=False, lumi=lumi, blind=blind, systematics=False)  # FIXME some syst currently broken
    get_nonprompt_plot(output, 'sublead_jet', axis, name='np_sublead_jet_pt', log=False)
    get_chargeflip_plot(output, 'sublead_jet', axis, name='cf_sublead_jet_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(subleading\ jet)$', 25, -5, 5)
    get_standard_plot(output, 'sublead_jet', axis, name='sublead_jet_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'sublead_jet', axis, name='np_sublead_jet_eta', log=False)
    get_chargeflip_plot(output, 'sublead_jet', axis, name='cf_sublead_jet_eta', log=False)

    axis = hist.Bin('pt', r'$p_{T}(leading\ bjet)\ (GeV)$', 24, 20, 500)
    get_standard_plot(output, 'lead_bjet', axis, name='lead_bjet_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'lead_bjet', axis, name='np_lead_bjet_pt', log=False)
    get_chargeflip_plot(output, 'lead_bjet', axis, name='cf_lead_bjet_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(leading\ bjet)$', 25, -5, 5)
    get_standard_plot(output, 'lead_bjet', axis, name='lead_bjet_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'lead_bjet', axis, name='np_lead_bjet_eta', log=False)
    get_chargeflip_plot(output, 'lead_bjet', axis, name='cf_lead_bjet_eta', log=False)

    axis = hist.Bin('pt', r'$p_{T}(subleading\ bjet)\ (GeV)$', 23, 20, 250)
    get_standard_plot(output, 'sublead_bjet', axis, name='sublead_bjet_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'sublead_bjet', axis, name='np_sublead_bjet_pt', log=False)
    get_chargeflip_plot(output, 'sublead_bjet', axis, name='cf_sublead_bjet_pt', log=False)

    axis = hist.Bin('eta', r'$\eta(subleading\ bjet)$', 25, -5, 5)
    get_standard_plot(output, 'sublead_bjet', axis, name='sublead_bjet_eta', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'sublead_bjet', axis, name='np_sublead_bjet_eta', log=False)
    get_chargeflip_plot(output, 'sublead_bjet', axis, name='cf_sublead_bjet_eta', log=False)

    # more event variables
    axis = hist.Bin('mass', r'$M(jj)\ (GeV)$', 20, 0, 2000)
    get_standard_plot(output, 'mjj_max', axis, name='mjj_max', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'mjj_max', axis, name='np_mjj_max', log=False)
    get_chargeflip_plot(output, 'mjj_max', axis, name='cf_mjj_max', log=False)

    axis = hist.Bin('mass', r'$M(\ell\ell)\ (GeV)$', 20, 0, 200)
    get_standard_plot(output, 'dilepton_mass', axis, name='dilepton_mass', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'dilepton_mass', axis, name='np_dilepton_mass', log=False)
    get_chargeflip_plot(output, 'dilepton_mass', axis, name='cf_dilepton_mass', log=False)

    axis = hist.Bin('mass', r'$min\ M_{T}(lep, p_{T}^{miss})\ (GeV)$', 20, 0, 200)
    get_standard_plot(output, 'min_mt_lep_met', axis, name='min_mt_lep_met', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'min_mt_lep_met', axis, name='np_min_mt_lep_met', log=False)
    get_chargeflip_plot(output, 'min_mt_lep_met', axis, name='cf_min_mt_lep_met', log=False)

    axis = hist.Bin("delta", r"$\Delta\eta(jj)$", 25, 0, 10)
    get_standard_plot(output, 'delta_eta_jj', axis, name='delta_eta_jj', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'delta_eta_jj', axis, name='np_delta_eta_jj', log=False)
    get_chargeflip_plot(output, 'delta_eta_jj', axis, name='cf_delta_eta_jj', log=False)

    axis = hist.Bin("delta", r"$min\ \Delta R(bjet, \ell)$", 25, 0, 10)
    get_standard_plot(output, 'min_bl_dR', axis, name='min_bl_dR', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'min_bl_dR', axis, name='np_min_bl_dR', log=False)
    get_chargeflip_plot(output, 'min_bl_dR', axis, name='cf_min_bl_dR', log=False)

    axis = hist.Bin('pt', r'$p_{T}(\ell\ell)\ (GeV)$', 15, 0, 300)
    get_standard_plot(output, 'dilepton_pt', axis, name='dilepton_pt', log=False, lumi=lumi, blind=blind)
    get_nonprompt_plot(output, 'dilepton_pt', axis, name='np_dilepton_pt', log=False)
    get_chargeflip_plot(output, 'dilepton_pt', axis, name='cf_dilepton_pt', log=False)

    if False:

        '''
        Look if we can reintegrate this plot? Do we need it?
        '''
        fig, ax  = plt.subplots(1,1,figsize=(10,10) )
        ax = hist.plot2d(
            output['chargeFlip_vs_nonprompt']['ttbar'].sum('n_ele').sum('dataset'),
            xaxis='n1',
            ax=ax,
            text_opts={'format': '%.3g'},
            patch_opts={},
        )
        ax.set_xlabel(r'$N_{charge flips}$')
        ax.set_ylabel(r'$N_{nonprompt}$')
        fig.savefig(plot_dir+'/SS/nChargeFlip_vs_nNonprompt_ttbar')

