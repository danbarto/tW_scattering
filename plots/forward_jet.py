
import os
import awkward as ak

from coffea import processor, hist, util
from coffea.processor import accumulate
import copy
import numpy as np
import re

from analysis.Tools.config_helpers import loadConfig, make_small, load_yaml, get_merged_output
from analysis.Tools.config_helpers import lumi as lumis
from analysis.Tools.samples import Samples

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, make_plot_from_dict

def get_histograms(output, histograms, total=False):
    '''
    output - e.g. output['MET'] selected histogram
    histograms - list of histograms to get, e.g. ['TTW', 'np_est_data']

    keep systematic axes and kinematic axes, integrate/project away all others
    '''
    wildcard = re.compile('.')
    out = {}
    tmp = output.copy()

    for hist in histograms:
        out[hist] = tmp[hist].sum('dataset')

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
    mc = get_histograms(output[hist], ['topW_lep', 'rare', 'diboson', 'TTW', 'TTZ', 'TTH', 'top'])
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

    if year == '2018':
        data = ['SingleMuon', 'DoubleMuon', 'EGamma', 'MuonEG']
    elif year == '2019':
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron', 'EGamma']
    else:
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron']
    order = ['topW_lep', 'diboson', 'rare', 'TTW', 'TTH', 'TTZ', 'top', 'XG']

    datasets = data + order

    years = ['2016', '2016APV', '2017', '2018']
    select_histograms = [
        #'bit_score_pp', 'bit_score_mm',
        'N_fwd', 'N_jet', 'N_central', 'N_b', 'N_ele',
        'N_tau', 'MET',
    ]

    if year == '2019':
        lumi = sum([lumis[y] for y in years])
    else:
        lumi = lumis[year]

    samples = Samples.from_yaml(f'../analysis/Tools/data/samples_v0_8_0_SS.yaml')
    mapping = load_yaml('../analysis/Tools/data/nano_mapping.yaml')

    if year == '2019':
        outputs = []
        for y in years:
            print (y)
            outputs.append(
                get_merged_output(
                    "forward_jet", y,
                    '../outputs/',
                    samples, mapping,
                    lumi=lumis[y],
                    variations = ['central', 'fake'],  # NOTE: modify here to include more systematic variations
                    select_histograms=select_histograms,
                ))
        output = accumulate(outputs)
        del outputs
    else:
        output = get_merged_output(
            "forward_jet", year,
            '../outputs/',
            samples, mapping,
            lumi=lumi,
            variations = ['central', 'fake'],  # NOTE: modify here to include more systematic variations
            select_histograms=select_histograms,
        )

    plot_dir    = os.path.join("../images/", str(year), 'OS', args.version)
    sub_dir = '/mc/'


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

    blind = False

    axis = N_bins
    get_standard_plot(output, 'N_b', axis, name='N_b', log=False, lumi=lumi, overflow='none', blind=blind)
