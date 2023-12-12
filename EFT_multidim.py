'''
This should run the final analyis:
- pick up cached histograms
- rebin distributions
- create inputs for data card
- run fits
'''
from __future__ import division
import os
import re
import time
import pickle
import gzip
import json
from datetime import datetime

import numpy as np
import pandas as pd
import socket

from coffea import hist, util
from coffea.processor import accumulate

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, colors, finalizePlotDir
from analysis.Tools.helpers import make_bh, get_samples
from analysis.Tools.config_helpers import get_cache, loadConfig, data_pattern, load_yaml, data_path
from analysis.Tools.config_helpers import lumi as lumis
from analysis.Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
from analysis.Tools.dataCard import dataCard
from analysis.Tools.samples import Samples
from analysis.Tools.HyperPoly import *

from analysis.Tools.limits import get_systematics, add_signal_systematics

from scipy import interpolate

# Taken from the NanoAOD-tools module
def get_points(points):
    points = points.replace('LHEWeight_','').replace('_nlo','')
    ops = points.split('_')[::2]
    vals = [ float(x.replace('p','.')) for x in points.split('_')[1::2] ]
    return dict(zip(ops, vals))

def get_coordinates(points):
    points = points.replace('LHEWeight_','').replace('_nlo','')
    vals = [ float(x.replace('p','.')) for x in points.split('_')[1::2] ]
    return tuple(vals)

def histo_values(histo, weight):
    return histo[weight].sum('dataset').values(overflow='all')[()]

def get_NLL(card_name):
    # card_name: string
    res = card.calcNLL(card_name)  # NOTE: calcNLL makes a unique dir
    try:
        nll = res['nll0'][0] + res['nll'][0]
    except (KeyError,ValueError):
        print (f"Couldn't calculate NLL for card {card_name}. Probably the fit failed")
        nll = -99999

    return nll

data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

wildcard = re.compile('.')

def write_card_wrapper(arguments):
    write_card(*arguments)

def combine_cards_wrapper(arguments):
    return combine_cards(*arguments)

def combine_cards(c, cards, out):
    print("\n\nCOMBINING CARDS\n\n")
    print(cards, out)
    print('\n\n')
    return card.combineCards(cards, name=out)

def write_trilep_card(histogram, year, region, axis, cpt, cpqm,
                      plot_dir='./',
                      systematics=False,
                      bsm_scales={},
                      histogram_incl = None,
                      signal_histogram = None,
                      ):

    print(f"Making trilep card for {year=}, {region=}, {cpt=}, {cpqm=}")
    x = cpt
    y = cpqm
    histo_name = region
    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
    plot_name = plot_name_short + f'_{region}_{year}'
    SM_plot_name = f'{region}_{year}'

    sm_point = 'central'
    if region == 'trilep_ttZ':
        bsm_point = 'central'
    else:
        if cpt == 0 and cpqm == 0:
            bsm_point = 'central'
        else:
            bsm_point = f"eft_cpt_{cpt}_cpqm_{cpqm}"
    ul = str(year)[2:]

    print ("Filling background histogram")
    backgrounds = {
        'signal':    histogram[('topW_lep', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'TTW':       histogram[('TTW', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'TTH':       histogram[('TTH', sm_point,'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'TTZ':       histogram[('TTZ', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'TZQ':       histogram[('TZQ', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'rare':      histogram[('rare', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'diboson':   histogram[('diboson', sm_point, 'central', 'central')].sum('EFT', 'systematic','prediction').copy(),
        'conv':      histogram[(wildcard, sm_point, 'conv_mc', 'central')].sum('EFT', 'systematic', 'prediction').copy(),
        'nonprompt': histogram[(wildcard, sm_point, 'np_est_data', 'central')].sum('EFT', 'systematic', 'prediction').copy(),
        }

    for p in backgrounds.keys():
        backgrounds[p] = backgrounds[p].rebin(axis.name, axis)

    total = backgrounds['signal'].copy()
    total.clear()
    for k in backgrounds.keys():
        nan_check = np.any(np.isnan(backgrounds[k].sum('dataset').values()[()]))
        print(f"NaN check for {p=}: {nan_check=}")
        if nan_check: print ("\n\n    !!!!!!!!\n\n\n\n")
        if not k == 'signal':
            total.add(backgrounds[k])

    print ("Filling data histogram. I can still stay blind!")
    observation = histogram[(data_pattern, 'central', 'central', 'central')].sum('dataset', 'EFT', 'systematic', 'prediction').copy()
    observation = observation.rebin(axis.name, axis)
    print (observation.values())
    # unblind the first 8 bins. this is hacky.
    unblind = observation._sumw[()][:0]
    blind   = np.zeros_like(observation._sumw[()][0:])
    observation._sumw[()] = np.concatenate([unblind, blind])

    print ("Filling signal histogram")
    if signal_histogram:
        pass
    else:
        signal_histogram = histogram
    signal = signal_histogram[('topW_lep', bsm_point, 'central', 'central')].sum('systematic', 'EFT', 'prediction').copy()
    signal = signal.rebin(axis.name, axis)
    #print()
    #signal = histogram[('topW_lep', bsm_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()  # FIXME this will eventually need the EFT axis?
    #print("!!!", signal.values())
    #print(f"{axis.name=}")
    #print(f"{region=}")
    #signal = signal.rebin(axis.name, axis)
    #print("!!!", signal.values())

    systematics= [
        ('signal_norm', 1.1, 'signal'),
        ('TTW_norm', 1.15, 'TTW'),
        ('TTZ_norm', 1.10, 'TTZ'),
        ('TZQ_norm', 1.10, 'TZQ'),
        ('TTH_norm', 1.15, 'TTH'),
        ('conv_norm', 1.20, 'conv'),
        ('diboson_norm', 1.20, 'diboson'),
        ('nonprompt_norm', 1.30, 'nonprompt'),
    ]

    print ("Making first plots")
    print ("...prepping the plots")
    hist_list = [
        backgrounds['signal'],
        backgrounds['rare'],
        backgrounds['diboson'],
        backgrounds['conv'],
        backgrounds['nonprompt'],
        backgrounds['TTZ'],
        backgrounds['TZQ'],
        backgrounds['TTH'],
        backgrounds['TTW'],
        ]
    edges = backgrounds['signal'].sum('dataset').axes()[0].edges()

    labels = [
        'SM scat.',
        'Rare',
        r'$VV/VVV$',
        r'$X\gamma$',
        'nonprompt',
        r'$t\bar{t}Z$',
        r'$tZq$',
        r'$t\bar{t}H$',
        r'$t\bar{t}W$',
        ]

    hist_colors = [
        colors['signal'],
        colors['rare'],
        colors['diboson'],
        colors['XG'],
        colors['non prompt'],
        colors['TTZ'],
        colors['TZQ'],
        colors['TTH'],
        colors['TTW'],
        ]

    fig, (ax, rax) = plt.subplots(2,1,figsize=(12,10), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)

    hep.cms.label(
        "Work in Progress",
        data=True,
        lumi=lumi,
        com=13,
        loc=0,
        ax=ax,
        )

    print ("...building histogram")

    hep.histplot(
        [ x.sum('dataset').values()[()] for x in hist_list],
        edges,
        histtype="fill",
        stack=True,
        label=labels,
        color=hist_colors,
        ax=ax)

    hep.histplot(
        [ signal.sum('dataset').values()[()]],
        edges,
        histtype="step",
        label=[r'$C_{\varphi t}=%s, C_{\varphi Q}^{-}=%s$'%(x,y)],
        color=['black'],
        ax=ax)

    hep.histplot(
        [ observation.values()[()]],
        edges,
        histtype="errorbar",
        label=[r'Observation'],
        color=['black'],
        ax=ax)


    hist.plotratio(
        num=observation,
        denom=total.sum("dataset"),
        ax=rax,
        error_opts=data_err_opts,
        denom_fill_opts=None, # triggers this: https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L376
        guide_opts={},
        unc='num',
        #unc=None,
        #overflow='over'
    )

    ax.legend(ncol=3)
    # labels
    rax.set_xlabel(backgrounds['signal'].sum('dataset').axes()[0].label)
    ax.set_xlim(edges[0],edges[-1])
    rax.set_ylabel(r'rel. unc.')
    ax.set_ylabel(r'Events')

    print ("...storing plots")

    fig.savefig(f'{plot_dir}/{plot_name}.png')
    fig.savefig(f'{plot_dir}/{plot_name}.pdf')

    plt.close(fig)
    del fig, ax, rax



    if systematics:
        # NOTE this loads background systematics.
        # Not fully complete, but most important systematics are here
        print ("Getting Background systematics")
        systematics = get_systematics(histogram, year, sm_point,
                                        correlated=False,
                                        signal=False,
                                        overflow='none',
                                        samples=samples,
                                        mapping=mapping[f'UL{ul}'],
                                        rebin=axis,
                                        trilep=True,
                                        )
        if year.count('2016'):
            print ("lumi uncertainties for 2016")
            systematics += lumi_systematics_2016
        elif year.count('2017'):
            print ("lumi uncertainties for 2017")
            systematics += lumi_systematics_2017
        elif year.count('2018'):
            print ("lumi uncertainties for 2018")
            systematics += lumi_systematics_2018
        else:
            print ("No lumi systematic assigned.")

    #    print ("Getting signal systematics")
    #    systematics = add_signal_systematics(histogram, year, sm_point,
    #                                            systematics=systematics,
    #                                            correlated=False,
    #                                            proc='topW_lep',
    #                                            overflow='none',
    #                                            samples=samples,
    #                                            mapping=mapping[f'UL{ul}'],
    #                                            rebin=axis,
    #                                            )

    if cpt==0 and cpqm==0:
        sm_card = makeCardFromHist(
            backgrounds,
            ext=f'SM_{SM_plot_name}',
            systematics = systematics,
            data = observation,
            blind = True,
            )
    else:
        sm_card = None

    bsm_card = makeCardFromHist(
        backgrounds,
        ext=f'BSM_{plot_name}',
        bsm_hist = signal,
        bsm_scales = bsm_scales,
        systematics = systematics,
        data = observation,
        blind = True,
        )

    bsm_cards[f'BSM_{plot_name}'] = bsm_card
    sm_cards[f'SM_{plot_name}'] = sm_card

    return plot_name, bsm_card, sm_card

def write_card(histogram, year, region, axis, cpt, cpqm,
               plot_dir='./',
               systematics=True,
               bsm_scales={},
               histogram_incl = None,
               signal_histogram = None,
               ):

    if region.count('trilep'):  # == 'dilepton_mass_ttZ' or region == 'signal_region_topW':
        return write_trilep_card(histogram, year, region, axis, cpt, cpqm, plot_dir, systematics, bsm_scales, histogram_incl, signal_histogram)

    print(f"Making dilep card for {year=}, {region=}, {cpt=}, {cpqm=}")

    x = cpt
    y = cpqm
    #sm_point  = f"cpt_{x}_cpqm_{y}"
    sm_point  = f"cpt_0_cpqm_0"
    bsm_point = f"bsm_cpt_{x}_cpqm_{y}"

    histo_name = region
    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
    plot_name = plot_name_short + f'_{region}_{year}'
    SM_plot_name = f'{region}_{year}'

    ul = str(year)[2:]

    print ("Filling background histogram")
    backgrounds = {
        'signal':    histogram[('topW_lep', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTW':       histogram[('TTW', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTH':       histogram[('TTH', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTZ':       histogram[('TTZ', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TZQ':       histogram[('TZQ', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'rare':      histogram[('rare', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'diboson':   histogram[('diboson', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'conv':      histogram[(wildcard, sm_point, 'conv_mc', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
        'nonprompt': histogram[(wildcard, sm_point, 'np_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
        }

    # NOTE some complication for charge flip.
    if region.count('_pp') or region.count('_mm'):
        backgrounds['chargeflip'] = histogram_incl[(wildcard, sm_point, 'cf_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy()
        backgrounds['chargeflip'].scale(0.5)
    else:
        backgrounds['chargeflip'] = histogram[(wildcard, sm_point, 'cf_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy()

    for p in backgrounds.keys():
        nan_check = np.any(np.isnan(backgrounds[p].sum('dataset').values()[()]))
        print(f"NaN check for {p=}: {nan_check=}")
        backgrounds[p] = backgrounds[p].rebin(axis.name, axis)

    total = backgrounds['signal'].copy()
    total.clear()
    for k in backgrounds.keys():
        if not k == 'signal':
            total.add(backgrounds[k])

    print ("Filling data histogram. I can still stay blind!")
    observation = histogram[(data_pattern, sm_point, 'central', 'central')].sum('dataset', 'EFT', 'systematic', 'prediction').copy()
    observation = observation.rebin(axis.name, axis)
    #print(sm_point)
    #print(observation.values())
    # unblind the first 8 bins. this is hacky.
    unblind = observation._sumw[()][:0]
    blind   = np.zeros_like(observation._sumw[()][0:])
    observation._sumw[()] = np.concatenate([unblind, blind])

    print ("Filling signal histogram")
    if signal_histogram:
        pass
    else:
        signal_histogram = histogram
    signal = signal_histogram[('topW_lep', bsm_point, 'central', 'central')].sum('systematic', 'EFT', 'prediction').copy()
    signal = signal.rebin(axis.name, axis)

    if systematics:
        # NOTE this loads background systematics.
        # Not fully complete, but most important systematics are here
        #print ("Getting Background systematics")
        systematics = []
        systematics += get_systematics(histogram, year, sm_point,
                                        correlated=False,
                                        signal=False,
                                        overflow='none',
                                        samples=samples,
                                        mapping=mapping[f'UL{ul}'],
                                        )
        if year.count('2016'):
            print ("lumi uncertainties for 2016")
            systematics += lumi_systematics_2016
        elif year.count('2017'):
            print ("lumi uncertainties for 2017")
            systematics += lumi_systematics_2017
        elif year.count('2018'):
            print ("lumi uncertainties for 2018")
            systematics += lumi_systematics_2018
        else:
            print ("No lumi systematic assigned.")

        print ("Getting signal systematics")
        systematics = add_signal_systematics(signal_histogram, year, sm_point,
                                                systematics=systematics,
                                                correlated=False,
                                                proc='topW_lep',
                                                overflow='none',
                                                samples=samples,
                                                mapping=mapping[f'UL{ul}'],
                                                )

    print ("Making first plots")
    print ("...prepping the plots")
    hist_list = [
        backgrounds['signal'],
        backgrounds['rare'],
        backgrounds['diboson'],
        backgrounds['conv'],
        backgrounds['nonprompt'],
        backgrounds['chargeflip'],
        backgrounds['TTZ'],
        backgrounds['TZQ'],
        backgrounds['TTH'],
        backgrounds['TTW'],
        ]
    edges = backgrounds['signal'].sum('dataset').axes()[0].edges()

    labels = [
        'SM scat.',
        'Rare',
        r'$VV/VVV$',
        r'$X\gamma$',
        'nonprompt',
        'charge mis-ID',
        r'$t\bar{t}Z$',
        r'$tZq$',
        r'$t\bar{t}H$',
        r'$t\bar{t}W$',
        ]

    hist_colors = [
        colors['signal'],
        colors['rare'],
        colors['diboson'],
        colors['XG'],
        colors['non prompt'],
        colors['chargeflip'],
        colors['TTZ'],
        colors['TZQ'],
        colors['TTH'],
        colors['TTW'],
        ]

    fig, (ax, rax) = plt.subplots(2,1,figsize=(12,10), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)

    hep.cms.label(
        "Work in Progress",
        data=True,
        lumi=lumi,
        com=13,
        loc=0,
        ax=ax,
        )

    print ("...building histogram")

    hep.histplot(
        [ x.sum('dataset').values()[()] for x in hist_list],
        edges,
        histtype="fill",
        stack=True,
        label=labels,
        color=hist_colors,
        ax=ax)

    hep.histplot(
        [ signal.sum('dataset').values()[()]],
        edges,
        histtype="step",
        label=[r'$C_{\varphi t}=%s, C_{\varphi Q}^{-}=%s$'%(x,y)],
        color=['black'],
        ax=ax)

    hep.histplot(
        [ observation.values()[()]],
        edges,
        histtype="errorbar",
        label=[r'Observation'],
        color=['black'],
        ax=ax)


    hist.plotratio(
        num=observation,
        denom=total.sum("dataset"),
        ax=rax,
        error_opts=data_err_opts,
        denom_fill_opts=None, # triggers this: https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L376
        guide_opts={},
        unc='num',
        #unc=None,
        #overflow='over'
    )

    ax.legend(ncol=3)
    # labels
    rax.set_xlabel(backgrounds['signal'].sum('dataset').axes()[0].label)
    ax.set_xlim(edges[0],edges[-1])
    rax.set_ylabel(r'rel. unc.')
    ax.set_ylabel(r'Events')

    print ("...storing plots")

    fig.savefig(f'{plot_dir}/{plot_name}.png')
    fig.savefig(f'{plot_dir}/{plot_name}.pdf')

    plt.close(fig)
    del fig, ax, rax

    # FIXME this needs to be parametrized again IF we decide to run significances!!
    if cpt == 0 and cpqm == 0:
        sm_card = makeCardFromHist(
            backgrounds,
            ext=f'SM_{SM_plot_name}',
            #scales = scales,
            #bsm_scales = bsm_scales,
            systematics = systematics,
            data = observation,
            blind = True,
            )
    else:
        sm_card = None  # background distributions are not parametrized anymore, so no need for a different card everytime


    bsm_card = makeCardFromHist(
        backgrounds,
        ext=f'BSM_{plot_name}',
        bsm_hist = signal,
        #scales = scales,
        bsm_scales = bsm_scales,
        systematics = systematics,
        data = observation,
        blind = True,
        )

    bsm_cards[f'BSM_{plot_name}'] = bsm_card
    sm_cards[f'SM_{plot_name}'] = sm_card

    return plot_name, bsm_card, sm_card

    ### needs to be fixed
    #else:
    #    bsm_cards[f'BSM_{plot_name}'] = f'/{card_dir}/BSM_{plot_name}_card.txt'
    #    sm_cards[f'SM_{plot_name}'] = f'/{card_dir}/SM_{plot_name}_card.txt'


if __name__ == '__main__':


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_scan', action='store_true', help="Run the full 2D scan")
    argParser.add_argument('--extended', action='store_true', help="Run extended 2D scan")
    argParser.add_argument('--regions', action='store', default='all', help="Which regions to include in the fit")
    argParser.add_argument('--comparison', action='store_true', help="Plot a comparison")
    argParser.add_argument('--overwrite', action='store_true', help="Overwrite existing cards")
    argParser.add_argument('--bit', action='store_true', help="Use boosted information tree (LT otherwise)")
    argParser.add_argument('--fit', action='store_true', help="Run combine (otherwise just plotting)")
    argParser.add_argument('--systematics', action='store_true', help="Run with realistic systematics (slower)")
    argParser.add_argument('--workers', action='store', default=5, type=int, help="Define how many cores/workers can be used for fitting")
    argParser.add_argument('--year', action='store', default=2016, type=str, help="Select years, comma separated")
    argParser.add_argument('--cpt', action='store', default=0, type=int, help="If run_scan is used, this is the cpt value that's being evaluated")
    argParser.add_argument('--cpqm', action='store', default=0, type=int, help="If run_scan is used, this is the cpqm value that's being evaluated")
    argParser.add_argument('--uaf', action='store_true', help="Store in different directory if on uaf.")
    argParser.add_argument('--scaling', action='store', choices=['LO','NLO'], help="run with scaling : LO or NLO?")
    argParser.add_argument('--physicsmodel', action='store_true', help="Use the higgs combine physics model")

    args = argParser.parse_args()

    run_scan = args.run_scan
    #inclusive = args.inclusive
    bit = args.bit
    fit = args.fit
    comparison = args.comparison

    import concurrent.futures
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from analysis.Tools.config_helpers import get_merged_output, load_yaml

    #ref_values = 'cpt_0p_cpqm_0p_nlo'
    ref_values = [0,0]


    #cfg = loadConfig()

    # set directories to save to
    if not args.uaf:
        base_dir = './plots/'
    else:
        base_dir = '/home/users/sjeon/public_html/tW_scattering/multidim/'
    finalizePlotDir(base_dir)
    dump_dir = './results/' # where to save json files
    finalizePlotDir(dump_dir)

    # NOTE placeholder systematics if run without --systematics
    mc_process_names = ['signal', 'TTW', 'TZQ', 'TTZ', 'TTH', 'conv', 'diboson', 'rare']
    systematics= [
        ('signal_norm',     1.10, 'signal'),
        ('TTW_norm',        1.15, 'TTW'),
        ('TTZ_norm',        1.10, 'TTZ'),
        ('TTH_norm',        1.15, 'TTH'),
        ('conv_norm',       1.20, 'conv'),
        ('diboson_norm',    1.20, 'diboson'),
        ('nonprompt_norm',  1.30, 'nonprompt'),
        ('chargeflip_norm', 1.20, 'chargeflip'),
    ]

    # lumi systematics following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Luminosity
    lumi_systematics_2016  = [ ('lumi16',     1.01,  p) for p in mc_process_names ]
    lumi_systematics_2016 += [ ('lumi161718', 1.006, p) for p in mc_process_names ]
    lumi_systematics_2017  = [ ('lumi17',     1.02,  p) for p in mc_process_names ]
    lumi_systematics_2017 += [ ('lumi161718', 1.009, p) for p in mc_process_names ]
    lumi_systematics_2017 += [ ('lumi1718',   1.006, p) for p in mc_process_names ]
    lumi_systematics_2018  = [ ('lumi18',     1.015, p) for p in mc_process_names ]
    lumi_systematics_2018 += [ ('lumi161718', 1.02,  p) for p in mc_process_names ]
    lumi_systematics_2018 += [ ('lumi1718',   1.002, p) for p in mc_process_names ]

    lt_axis      = hist.Bin("ht",   r"$L_{T}$ (GeV)", [100,200,300,400,500,600,700,2000])
    lt_red_axis  = hist.Bin("lt",   r"$L_{T}$ (GeV)", [0,400,1000])
    #bit_axis     = hist.Bin("bit",  r"N",             10, 0, 1)
    #bit_axis     = hist.Bin("bit",  r"N",             [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    bit_axis     = hist.Bin("bit",  r"BIT score",      20,0,1)
    mass_axis    = hist.Bin("mass", r"dilepton mass",  1,0,200)  # make this completely inclusive


    all_cards = []
    #card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    if socket.gethostname().count("uaf"):
        #release_location = os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_9/src/tW_scattering/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/')
        release_location = os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_9/src/tW_scattering/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/')
    else:
        #release_location = os.path.expandvars('./CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/')
        release_location = os.path.expandvars('./CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/')
    card = dataCard(releaseLocation=release_location)
    card_dir = os.path.abspath(os.path.expandvars('./data/cards/')) + '/'
    #card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    #card_dir = os.path.expandvars('$TWHOME/data/cards/')

    trilep_regions = [
        "trilep_ttZ",
        "trilep_WZ",
        "trilep_XG",
        "trilep_topW_qm_0Z",
        "trilep_topW_qp_0Z",
        "trilep_topW_qm_1Z",
        "trilep_topW_qp_1Z",
    ]

    # Define a scan
    if args.run_scan:
        xr = np.arange(-7,8,1)
        yr = np.arange(-7,8,1)
        if args.extended:
            xr = np.arange(-20,21,2)  # this is Y on the plots!
            yr = np.arange(-20,21,2)  # this is X on the plots!
    else:
        xr = np.array([int(args.cpt)])
        yr = np.array([int(args.cpqm)])

    X, Y = np.meshgrid(xr, yr)
    scan = list(zip(X.flatten(), Y.flatten()))
    if (0,0) not in scan:  # always add SM point to scan
        scan = [(0,0)] + scan

    # Define Scaling Polynomial
    def scalePolyNLO(xt, xQM):
        # ttZ SMEFT@NLO scale poly
        return 1 + 0.072813*xt - 0.098492*xQM + 0.005049*xt**2 - 0.002042*xt*xQM + 0.003988*xQM**2

    def scalePolyLO(xt, xQM):
        return 1 + 0.068485*xt - 0.104991*xQM + 0.003982*xt**2 - 0.002534*xt*xQM + 0.004144*xQM**2

    def ttH_scalePolyLO(xt, xQM):
        # ttH SMEFT@LO poly
        return 1 - 0.000363*xt + 0.000218*xQM + 0.00054*xt**2 - 0.000409*xt*xQM + 0.00061*xQM**2

    def tZq_scalePolyLO(xt, xQM):
        # tZq scaling
        # FIXME this needs a different sample splitting
        return 1 + 0.005554*xt + 0.021142*xQM + 0.000452*xt**2 - 0.000913*xt*xQM + 0.00114*xQM**2

    
    years = args.year.split(',')

    if args.overwrite:
        # load outputs (coffea histograms)
        # histograms are created per sample,
        # x-secs and lumi scales are applied on the fly below
        outputs = {}
        signal_outputs = {}
        outputs_tri = {}
        signal_outputs_tri = {}
        samples = Samples.from_yaml(f'analysis/Tools/data/samples_v0_8_0_SS.yaml')
        mapping = load_yaml('analysis/Tools/data/nano_mapping.yaml')

        for year in years:
            # SS_analysis_MCall_central_2018_cpt_-5_cpqm_-5_20230327_223314.coffea
            outputs[year] = util.load(f'./outputs/{year}_fixed_merged.coffea')
            signal_outputs[year] = util.load(f'./outputs/{year}_signal_merged.coffea')
            if args.regions in ['all', 'trilep', 'ttZ']:
                outputs_tri[year] = util.load(f'./outputs/{year}_trilep_merged.coffea')
                signal_outputs_tri[year] = util.load(f'./outputs/{year}_signal_trilep_merged.coffea')
            #if args.regions in ['inclusive', 'all']:
            #    outputs[year] = get_merged_output(
            #        'SS_analysis',
            #        year,
            #        './outputs/',
            #        samples, mapping,
            #        lumi=lumi,
            #        select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'] if args.bit else ['LT', 'LT_SR_pp', 'LT_SR_mm'],
            #    )#, date='20220624')
            #outputs_tri[year] = get_merged_output(
            #    'trilep_analysis',
            #    year,
            #    './outputs/',
            #    samples, mapping,
            #    lumi=lumis[year],
            #    select_histograms = ['dilepton_mass_ttZ', 'signal_region_topW'],
            #)#, date='20220624')

    results = {}

    sm_cards = {}
    bsm_cards = {}

    cards_to_write = []

    if args.regions == 'inclusive':
        regions = [
            ("bit_score_incl", bit_axis, lambda x: x["bit_score_incl"]),
        ]

    elif args.regions == 'SS':
        regions = [
            ("bit_score_pp", bit_axis, lambda x: x["bit_score_pp"]),
            ("bit_score_mm", bit_axis, lambda x: x["bit_score_mm"]),
        ]
    elif args.regions == 'SR':
        regions = [
            ("bit_score_pp", bit_axis, lambda x: x["bit_score_pp"]),
            ("bit_score_mm", bit_axis, lambda x: x["bit_score_mm"]),
            ("trilep_topW_qm_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qp_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qm_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(0.5,2.5))),
            ("trilep_topW_qp_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(0.5,2.5))),
        ]
    elif args.regions == 'ttZ':
        regions = [
            ("trilep_ttZ", mass_axis, lambda x: x['dilepton_mass_ttZ']),
        ]
    elif args.regions == 'trilep':
        regions = [
            ("trilep_ttZ", mass_axis, lambda x: x['dilepton_mass_ttZ']),
            ("trilep_WZ", lt_red_axis, lambda x: x["LT_WZ"]),
            ("trilep_XG", lt_red_axis, lambda x: x["LT_XG"]),
            ("trilep_topW_qm_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qp_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice( 0.5,  1.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qm_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice( 0.5,2.5))),
            ("trilep_topW_qp_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice( 0.5,  1.5)).integrate('N', slice( 0.5,2.5))),
        ]
    elif args.regions == 'all':
        regions = [
            ("bit_score_pp", bit_axis, lambda x: x["bit_score_pp"]),
            ("bit_score_mm", bit_axis, lambda x: x["bit_score_mm"]),
            ("trilep_ttZ",  mass_axis, lambda x: x['dilepton_mass_ttZ']),
            ("trilep_WZ", lt_red_axis, lambda x: x["LT_WZ"]),
            ("trilep_XG", lt_red_axis, lambda x: x["LT_XG"]),
            ("trilep_topW_qm_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qp_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(-0.5,0.5))),
            ("trilep_topW_qm_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(0.5,2.5))),
            ("trilep_topW_qp_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(0.5,2.5))),
        ]


    if args.physicsmodel:

        scaling_pkl = []
        cards = {}
        for year in years:
            suffix = "_scaled" if args.scaling else ""
            plot_dir = base_dir + year + suffix + '/'
            finalizePlotDir(plot_dir)

            ul = str(year)[2:]
            lumi = lumis[year]

            # channels in combine will be named after the `region` here
            for region, axis, get_histo in regions:

                channel = f'{region}_{year}'

                if args.overwrite:
                    if region in trilep_regions:
                        output = outputs_tri[year]
                        signal_output = signal_outputs_tri[year]
                    else:
                        output = outputs[year]
                        signal_output = signal_outputs[year]

                # NOTE so here we need to extract the scaling dictionary
                '''
                {'channel': 'SR', 'process': 'ttbar', 'parameters': ['cSM[1]', 'cpt[0,-20,20]', 'cpQM[0,-20,20]',
                '''

                coordinates = [(0, 0), (3, 0), (0, 3), (6, 0), (3, 3), (0, 6)]
                ref_coordinates = [0,0]

                sm = get_histo(signal_output)[('topW_lep', f'bsm_cpt_0_cpqm_0', 'central', 'central')].values()[('topW_lep', f'bsm_cpt_0_cpqm_0', 'central', 'central')]
                predictions = []
                for cpt, cpqm in coordinates:
                    #predictions[(cpt, cpqm)] = get_histo(signal_output).sum('dataset').integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', f'bsm_cpt_{cpt}_cpqm_{cpqm}').values()[()]
                    #predictions[(cpt, cpqm)] = get_histo(signal_output)[('topW_lep', f'bsm_cpt_{cpt}_cpqm_{cpqm}', 'central', 'central')].values()[('topW_lep', f'bsm_cpt_{cpt}_cpqm_{cpqm}', 'central', 'central')]
                    predictions.append(get_histo(signal_output)[('topW_lep', f'bsm_cpt_{cpt}_cpqm_{cpqm}', 'central', 'central')].values()[('topW_lep', f'bsm_cpt_{cpt}_cpqm_{cpqm}', 'central', 'central')]/sm)

                hp = HyperPoly(2)
                hp.initialize(coordinates, ref_coordinates)
                coeffs = hp.get_parametrization(np.array(predictions))

                print("SM values")
                print(sm)
                print("prediction for 6,6")
                print(hp.eval(coeffs, [6,6]))

                # from 'sm + x0 + x1 + x0**2 + x0*x1 + x1**2'
                # to sm + x0 + x0**2 + x1 + x1*x0 + x1**2
                # so columns 0, 1, 3, 2, 4, 5
                scaling = coeffs[[0,1,3,2,4,5]].T

                scaling_pkl.append(
                    {
                        'channel': f'dc_{channel}',
                        'process': 'signal',
                        'parameters': ['cSM[1]', 'cpt[0,-20,20]', 'cpQM[0,-20,20]'],
                        'scaling': scaling
                    }
                )


                # Now write the SM data card like before and be done??
                if region.count('pp') or region.count('mm'):
                    histogram_incl = output['bit_score_incl']  # NOTE not nice, but need this for now
                else:
                    histogram_incl = None

                # FIXME these scales need to go into the scaling file as well
                bsm_scales = {'TTZ': 1, 'TTH': 1, 'TZQ': 1}  # NOTE: 1 is default, just make it explicit

                sm_card = write_card(
                    get_histo(output),
                    year,
                    region,
                    axis,
                    0,
                    0,
                    plot_dir,
                    args.systematics,
                    bsm_scales,
                    histogram_incl,
                    get_histo(signal_output),
                )

                cards[channel] = sm_card[2]

                print(sm_card)

                #raise NotImplementedError

        print(scaling_pkl)

        combined_card = combine_cards(card, cards, 'combined_card_physics_model.txt')

        with gzip.open(f"{card_dir}/scaling.pkl.gz", "wb") as fout:
            pickle.dump(scaling_pkl, fout)

        res = card.interference_scan(combined_card, f"{card_dir}/scaling.pkl.gz")

        raise NotImplementedError



    else:

        for x,y in scan:
            if args.scaling == 'LO':
                bsm_scales = {'TTZ': scalePolyLO(x,y), 'TTH': ttH_scalePolyLO(x,y), 'TZQ': tZq_scalePolyLO(x,y)}
            elif args.scaling == 'NLO':
                bsm_scales = {'TTZ': scalePolyNLO(x,y), 'TTH': ttH_scalePolyLO(x,y), 'TZQ': tZq_scalePolyLO(x,y)}
            else:
                bsm_scales = {'TTZ': 1, 'TTH': 1, 'TZQ': 1}  # NOTE: 1 is default, just make it explicit

            print (f"Working on point {x}, {y}")

            sm_cards[(x,y)] = {}
            bsm_cards[(x,y)] = {}


            for year in years:
                suffix = "_scaled" if args.scaling else ""
                plot_dir = base_dir + year + suffix + '/'
                finalizePlotDir(plot_dir)

                ul = str(year)[2:]
                lumi = lumis[year]

                for region, axis, get_histo in regions:
                    if args.overwrite:
                        if region in trilep_regions:
                            output = outputs_tri[year]
                            signal_output = signal_outputs_tri[year]
                        else:
                            output = outputs[year]
                            signal_output = signal_outputs[year]

                    if args.overwrite:
                        #histogram = output[region]
                        if region.count('pp') or region.count('mm'):
                            histogram_incl = output['bit_score_incl']  # NOTE not nice, but need this for now
                        else:
                            histogram_incl = None
                        cards_to_write.append((get_histo(output), year, region, axis, x, y, plot_dir, args.systematics, bsm_scales, histogram_incl, get_histo(signal_output)))
                    #bsm_card, sm_card = write_card(output, year, region, axis, x, y,
                    #                               plot_dir='./',
                    #                               systematics=True,
                    #                               )

                    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
                    plot_name = plot_name_short + f'_{region}_{year}'
                    SM_plot_name = f'{region}_{year}'
                    bsm_cards[(x,y)][f'BSM_{plot_name}'] = f'/{card_dir}/BSM_{plot_name}_card.txt'
                    sm_cards[(x,y)][f'SM_{SM_plot_name}']   = f'/{card_dir}/SM_{SM_plot_name}_card.txt'

        workers = args.workers
        if args.overwrite:
            if workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    for card_name, result in zip(cards_to_write, executor.map(write_card_wrapper, cards_to_write)):
                        print (f"Done with {card_name}")
            else:
                #print (cards_to_write)
                result = [write_card_wrapper(c) for c in cards_to_write]  # fuck map

        if fit:
            all_combined = []
            print ("Starting with combining cards")
            for x,y in scan:

                plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
                plot_name = plot_name_short + f'_{region}_{year}'
                SM_plot_name = f'_{region}_{year}'  # FIXME we should really carry these names through
                # NOTE running years individually and then just combining
                # this avoids having to load all the histograms at once
                print ("Combining cards:")
                print (sm_cards[(x,y)])
                # FIXME: run this step in parallel too!
                if x==0 and y==0:
                    all_combined.append((card, sm_cards[(x,y)], f'SM_combined.txt'))
                    #sm_card_combined = card.combineCards(sm_cards[(x,y)], name=f'SM_combined.txt')
                    #all_cards.append(sm_card_combined)
                #sm_card_combined = card.combineCards(sm_cards[(x,y)], name=f'SM_{plot_name_short}.txt')  # NOTE old parametrized version
                all_combined.append((card, bsm_cards[(x,y)], f'BSM_{plot_name_short}.txt'))
                #bsm_card_combined = card.combineCards(bsm_cards[(x,y)], name=f'BSM_{plot_name_short}.txt')

                #all_cards.append(sm_card_combined)
                #all_cards.append(bsm_card_combined)

            print ("Done with the pre-processing and data card making, running fits now.")
            print (f"Using {workers} workers")


            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(combine_cards_wrapper, all_combined):
                    print ("Combined card:", result)
                    all_cards.append(result)
                    #print (val, result)
                    #all_nll[card_name.split('/')[-1].strip('.txt')] = result

            all_nll = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for card_name, result in zip(all_cards, executor.map(get_NLL, all_cards)):

                    #print (val, result)
                    all_nll[card_name.split('/')[-1].strip('.txt')] = result

            for x,y in scan:
                plot_name_short = f"BIT_cpt_{x}_cpqm_{y}" if bit else f"LT_cpt_{x}_cpqm_{y}"
                print (plot_name_short)
                #results[(x,y)] = -2*(all_nll[f'SM_{plot_name_short}'] - all_nll[f'BSM_{plot_name_short}'])
                results[(x,y)] = -2*(all_nll[f'SM_combined'] - all_nll[f'BSM_{plot_name_short}'])

    if fit and len(xr)>4:


        nodes = np.array( [
            [13, 9],
            [12, 10],
            [9, 10],
            [5, 8],
            [0, 4],
            [-5, -3],
            [-6, -5],
            [-7.5, -10],
            [-7.5, -14],
            [-7, -16],
            [-6, -18],
            [-5, -19],
            [-2.5, -20],
            [0, -19],
            [2, -18],
            [1, -16],
            [0, -14],
            [-1, -10],
            [0, -4],
            [2.5, 0],
            [5, 3],
            [10, 6],
            [12, 7.5],
            [12.7, 8],
            [13, 9],
        ] )

        ttz_x = nodes[:,0]
        ttz_y = nodes[:,1]

        tck,u     = interpolate.splprep( [ttz_x,ttz_y] ,s = 0 )
        ttz_xnew, ttz_ynew = interpolate.splev( np.linspace( 0, 1, 100 ), tck,der = 0)


        ## FIXME!!!!
        #results[(2, -4)] = 3
        #results[(-3, 7)] = 13

        Z = np.array(list(results.values()))
        Z = np.reshape(Z, X.shape)


        Z[np.where(Z<0)] = 0

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        from matplotlib.colors import LogNorm
        im = ax.matshow(Z)
        #ax.set_xticks(range(df.select_dtypes(['number']).shape[1]))
        #ax.set_xticklabels(df.select_dtypes(['number']).columns, rotation=90, fontdict={'fontsize':12})
        #ax.set_yticks(range(df.select_dtypes(['number']).shape[1]))
        #ax.set_yticklabels(df.select_dtypes(['number']).columns, fontdict={'fontsize':12})
        cbar = ax.figure.colorbar(im, norm=LogNorm(vmin=0.01, vmax=100))
        #cbar.ax.tick_params(labelsize=12)

        ax.set_title('Limits', fontsize=16)
        fig.savefig('./plots/limits_2D_test.png')


        fig, ax, = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(
            "WIP",
            data=False,
            #year=2018,
            lumi=41.5,
            loc=0,
            ax=ax,
            )

        ax.set_ylim(-25, 20)
        ax.set_xlim(-15, 20)

        ax.set_xlabel(r'$C_{\varphi Q}^{-}/\Lambda^{2} (TeV^{-2})$')
        ax.set_ylabel(r'$C_{\varphi t}/\Lambda^{2} (TeV^{-2})$')

        # NOTE: X and Y are switched in this plot!
        CS = ax.contour(Y, X, Z, levels = [2.28, 5.99], colors=['#FF595E', '#5bc0de'], # 68/95 % CL
                        linestyles=('-',),linewidths=(4,))
        fmt = {}
        strs = ['68%', '95%']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

        ax.plot( ttz_xnew, ttz_ynew, linewidth=4, color='#525B76')

        # make a manual legend
        import matplotlib.patches as mpatches
        patch1 = mpatches.Patch(color='#FF595E', label='68% CL expected')
        patch2 = mpatches.Patch(color='#5bc0de', label='95% CL expected')
        patch3 = mpatches.Patch(color='#525B76', label='95% CL observed, TOP-21-001')

        ax.legend(handles=[patch1, patch2, patch3])

        plt.show()

        year_str = 'all' if len(years)>1 else str(years[0])

        fig.savefig(base_dir+f'scan_bit_{args.regions}_{args.scaling}_{year_str}.png')
        fig.savefig(base_dir+f'scan_bit_{args.regions}_{args.scaling}_{year_str}.pdf')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if bit:
            out_file = f'results_bit_{args.regions}_{args.scaling}_{year_str}_{timestamp}.json'
        else:
            out_file = f'results_lt_{args.regions}_{args.scaling}_{year_str}_{timestamp}.json'

        results_dump = {}
        results_dump['data'] = {}

        for x,y in results:
            results_dump['data'][f'cpt_{x}_cpqm_{y}'] = results[(x,y)]

        results_dump['years'] = years

        with open(dump_dir + out_file, 'w') as f:
            json.dump(results_dump, f)
            print (f"Stored results in {dump_dir+out_file}")


        # also do 1D plots

        fig, ax = plt.subplots()
        hep.cms.label(
                "Work in progress",
                data=True,
                #year=2018,
                lumi=60,
                loc=0,
                ax=ax,
               )
        midpoint = int(len(xr)/2)
        plt.plot(xr, Z[midpoint,:], label=r'cpt', c='green')
        plt.plot(yr, Z[:,midpoint], label=r'cpqm', c='darkviolet')
        plt.legend()
        plt.show()

        fig.savefig(base_dir+f'1D_scaling_test_{args.regions}.png')
        fig.savefig(base_dir+f'1D_scaling_test_{args.regions}.pdf')



    # NOTE re-init dataCard here just so that we always clean up the right dir...
    #card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    #card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    card = dataCard(releaseLocation=release_location)
    card.cleanUp()

    if False:
        # NOTE this does something completely unrelated and should be retired...

        rx = np.arange(-7,8,1)
        ry = np.arange(-7,8,1)
        X, Y = np.meshgrid(rx, ry)
        scan = zip(X.flatten(), Y.flatten())

        ## I can make a 2D matrix of the inclusive x-sec changes, and one for high LT
        output = get_merged_output('SS_analysis', '2018', select_datasets=['topW_lep'])

        axis = lt_axis
        values = {}
        for x,y in scan:
            bsm_point = f'eft_cpt_{x}_cpqm_{y}'

            values[(x,y)] = output['LT']['topW_lep'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', bsm_point).sum('dataset').rebin(axis.name, axis).values()[()]

        inclusive_vals = np.zeros((len(rx),len(ry)))
        high_e_vals = np.zeros((len(rx),len(ry)))
        for i,x in enumerate(rx):
            for j, y in enumerate(ry):
                inclusive_vals[i,j] = sum(values[(x,y)])/sum(values[(0,0)])
                high_e_vals[i,j] = values[(x,y)][-1]/values[(0,0)][-1]


        fig, ax = plt.subplots()

        hep.cms.label(
            "WIP",
            data=True,
            #year=2018,
            lumi=60.0+41.5+35.9,
            loc=0,
            ax=ax,
        )
        cax = ax.matshow(
            inclusive_vals,
            cmap='RdYlGn_r',
            vmin=0, vmax=5,
        )
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

        fig.colorbar(cax)

        fig.savefig(f'{base_dir}/inclusive_scaling.png')
        fig.savefig(f'{base_dir}/inclusive_scaling.pdf')

        fig, ax = plt.subplots()

        hep.cms.label(
            "WIP",
            data=True,
            lumi=60.0+41.5+35.9,
            loc=0,
            ax=ax,
        )
        cax = ax.matshow(
            high_e_vals,
            cmap='RdYlGn_r',
            vmin=0, vmax=50,
        )
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

        fig.colorbar(cax)

        fig.savefig(f'{base_dir}/high_E_scaling.png')
        fig.savefig(f'{base_dir}/high_E_scaling.pdf')
