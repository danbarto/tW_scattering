#!/usr/bin/env python3

import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.processor import accumulate
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
        normalize = False,
):
    mc = get_histograms(output[hist], ['topW_lep', 'rare', 'diboson', 'TTW', 'conv_mc', 'TTZ', 'TTH', 'np_est_data'])
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
        normalize = normalize,
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


if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--normalize', action='store_true', default=None, help="Normalize?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--version', action='store', default='v21', help="Version of the NN training. Just changes subdir.")
    argParser.add_argument('--sample', action='store', default=None, help="Plot single sample")
    argParser.add_argument('--postfix', action='store', default='', help="postfix for plot directory")
    args = argParser.parse_args()

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
            outputs.append(get_merged_output("trilep_analysis", year=y, postfix='cpt_0_cpqm_0'))
        output = accumulate(outputs)
        del outputs
    else:
        output = get_merged_output("trilep_analysis", year=year, postfix='cpt_0_cpqm_0')

    #plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'OS', args.version)
    plot_dir    = os.path.join("/home/daniel/TTW/tW_scattering/plots/images/", str(year), 'trilep', args.version)

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

    blind = False

    # Object multiplicities
    axis = hist.Bin('mass', r'$m_{\ell\ell} (GeV)$', 20, 0, 200)
    get_standard_plot(output, 'dilepton_mass', axis, name='dilepton_mass', log=False, lumi=lumi, blind=blind)
    #get_nonprompt_plot(output, 'N_fwd', axis, name='np_N_fwd', log=False)
    #get_chargeflip_plot(output, 'N_fwd', axis, name='cf_N_fwd', log=False)

    get_standard_plot(output, 'dilepton_mass_WZ', axis, name='dilepton_mass_WZ', log=False, lumi=lumi, blind=blind)
    get_standard_plot(output, 'dilepton_mass_XG', axis, name='dilepton_mass_XG', log=False, lumi=lumi, blind=blind)
    get_standard_plot(output, 'dilepton_mass_ttZ', axis, name='dilepton_mass_ttZ', log=False, lumi=lumi, blind=blind)
    get_standard_plot(output, 'dilepton_mass_topW', axis, name='dilepton_mass_topW', log=False, lumi=lumi, blind=blind)

    axis = hist.Bin('pt', r'$p_{T} (GeV)$', 25, 0, 500)
    get_standard_plot(output, 'fwd_jet', axis, name='fwd_jet_pt', log=False, lumi=lumi, blind=blind, normalize=TFnormalize)

    axis = hist.Bin('pt', r'$p_{T}^{miss} (GeV)$', 25, 0, 500)
    get_standard_plot(output, 'MET', axis, name='MET_pt', log=False, lumi=lumi, blind=blind, normalize=TFnormalize)
