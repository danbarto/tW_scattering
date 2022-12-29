from __future__ import division
import os
import re
import time
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd

from coffea import hist
from coffea.processor import accumulate

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, colors, finalizePlotDir
from Tools.helpers import make_bh, get_samples
from Tools.config_helpers import get_cache, loadConfig, data_pattern, load_yaml, data_path
from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
from Tools.yahist_to_root import yahist_to_root
from Tools.dataCard import dataCard

from Tools.HyperPoly import HyperPoly 
from Tools.limits import regroup_and_rebin, get_systematics, add_signal_systematics
from Tools.EFT_tools import make_scan

from scipy import interpolate


def histo_values(histo, weight):
    return histo[weight].sum('dataset').values(overflow='all')[()]

data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

wildcard = re.compile('.')

def make_plot(histogram, year, region, axis, cpt, cpqm,
              plot_dir='./',
              systematics=False,
              bsm_scales={'TTZ':1},
              scaling=None,
              ):

    x = cpt
    y = cpqm
    histo_name = region
    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
    plot_name = plot_name_short + f'_{region}_{year}_{scaling}'

    # scale
    histogram = histogram.copy()
    histogram.scale(bsm_scales, axis='dataset')

    sm_point = 'central'
    if region == 'trilep_ttZ':
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
    signal = histogram[('topW_lep', bsm_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()  # FIXME this will eventually need the EFT axis?
    signal = signal.rebin(axis.name, axis)

    if systematics:
        # NOTE this loads background systematics.
        # Not fully complete, but most important systematics are here
        print ("Getting Background systematics")
        systematics = get_systematics(histogram, year, sm_point,
                                        correlated=False,
                                        signal=False,
                                        overflow='none',
                                        samples=samples[year],
                                        mapping=mapping[f'UL{ul}'],
                                        rebin=axis,
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
        systematics = add_signal_systematics(histogram, year, sm_point,
                                                systematics=systematics,
                                                correlated=False,
                                                proc='topW_lep',
                                                overflow='none',
                                                samples=samples[year],
                                                mapping=mapping[f'UL{ul}'],
                                                rebin=axis,
                                                )


    print ("Making first plots")
    print ("...prepping the plots")
    global hist_list
    hist_list = [
        backgrounds['signal'],
        backgrounds['rare'],
        backgrounds['diboson'],
        backgrounds['conv'],
        backgrounds['nonprompt'],
        #backgrounds['chargeflip'],
        backgrounds['TTZ'],
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
        #'charge mis-ID',
        r'$t\bar{t}Z$',
        r'$t\bar{t}H$',
        r'$t\bar{t}W$',
        ]

    hist_colors = [
        colors['signal'],
        colors['rare'],
        colors['diboson'],
        colors['XG'],
        colors['non prompt'],
        #colors['chargeflip'],
        colors['TTZ'],
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

    return signal, backgrounds



if __name__ == '__main__':


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--systematics', action='store_true', help="Run with realistic systematics (slower)")
    argParser.add_argument('--year', action='store', default=2016, type=str, help="Select years, comma separated")
    argParser.add_argument('--cpt', action='store', default=0, type=int, help="If run_scan is used, this is the cpt value that's being evaluated")
    argParser.add_argument('--cpqm', action='store', default=0, type=int, help="If run_scan is used, this is the cpqm value that's being evaluated")
    argParser.add_argument('--uaf', action='store_true', help="Store in different directory if on uaf.")
    argParser.add_argument('--scaling', action='store', choices=['LO','NLO'], help="run with scaling : LO or NLO?")

    args = argParser.parse_args()

    import concurrent.futures
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from Tools.config_helpers import get_merged_output, load_yaml

    ref_values = [0,0]

    cfg = loadConfig()

    # set directories to save to
    if not args.uaf:
        base_dir = './plots_LT/'
    else:
        base_dir = '/home/users/sjeon/public_html/tW_scattering/multidim/LTplots/'
    finalizePlotDir(base_dir)

    # NOTE placeholder systematics if run without --systematics
    mc_process_names = ['signal', 'TTW', 'TTZ', 'TTH', 'conv', 'diboson', 'rare']
    systematics= [
        ('signal_norm', 1.1, 'signal'),
        ('TTW_norm', 1.15, 'TTW'),
        ('TTZ_norm', 1.10, 'TTZ'),
        ('TTH_norm', 1.15, 'TTH'),
        ('conv_norm', 1.20, 'conv'),
        ('diboson_norm', 1.20, 'diboson'),
        ('nonprompt_norm', 1.30, 'nonprompt'),
        ('chargeflip_norm', 1.20, 'chargeflip'),
    ]

    # lumi systematics following https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Luminosity
    lumi_systematics_2016  = [ ('lumi16', 1.01, p) for p in mc_process_names ]
    lumi_systematics_2016 += [ ('lumi161718', 1.006, p) for p in mc_process_names ]
    lumi_systematics_2017  = [ ('lumi17', 1.02, p) for p in mc_process_names ]
    lumi_systematics_2017 += [ ('lumi161718', 1.009, p) for p in mc_process_names ]
    lumi_systematics_2017 += [ ('lumi1718', 1.006, p) for p in mc_process_names ]
    lumi_systematics_2018  = [ ('lumi18', 1.015, p) for p in mc_process_names ]
    lumi_systematics_2018 += [ ('lumi161718', 1.02, p) for p in mc_process_names ]
    lumi_systematics_2018 += [ ('lumi1718', 1.002, p) for p in mc_process_names ]

    lt_axis      = hist.Bin("ht",      r"$L_{T}$ (GeV)",   [100,200,300,400,500,600,700,2000])
    lt_red_axis  = hist.Bin("lt",      r"$L_{T}$ (GeV)",   [0,100,200,300,400,500,600,700,800,900,1000])
    bit_axis     = hist.Bin("bit",           r"BIT score",         20,0,1)
    mass_axis     = hist.Bin("mass",           r"dilepton mass",   1,0,200)  # make this completely inclusive


    all_cards = []
    card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    card_dir = os.path.expandvars('$TWHOME/data/cards/')

    x = args.cpt
    y = args.cpqm

    # Define Scaling Polynomial
    def scalePolyNLO(xt, xQM):
        return 1 + 0.072813*xt - 0.098492*xQM + 0.005049*xt**2 - 0.002042*xt*xQM + 0.003988*xQM**2

    def scalePolyLO(xt, xQM):
        return 1 + 0.068485*xt - 0.104991*xQM + 0.003982*xt**2 - 0.002534*xt*xQM + 0.004144*xQM**2

    if args.year == 'all':
        years = ['2016','2016APV','2017','2018']
    else: 
        years = args.year.split(',')

    # load outputs (coffea histograms)
    # histograms are created per sample,
    # x-secs and lumi scales are applied on the fly below
    outputs = {}
    samples = {}
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    for year in years:
        ul = str(year)[2:]
        samples[year] = get_samples(f"samples_UL{ul}.yaml")
        outputs_tri[year] = get_merged_output(
            'trilep_analysis',
            year,
            select_histograms = ['dilepton_mass_ttZ', 'signal_region_topW'],
        )#, date='20220624')

    results = {}

    print (f"Working on point {x}, {y}")

    regions = [
        ("trilep_topW_qm_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(-0.5,0.5))),
        ("trilep_topW_qp_0Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(-0.5,0.5))),
        ("trilep_topW_qm_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(-1.5, -0.5)).integrate('N', slice(0.5,2.5))),
        ("trilep_topW_qp_1Z", lt_red_axis, lambda x: x["signal_region_topW"].integrate('charge', slice(0.5, 1.5)).integrate('N', slice(0.5,2.5))),
        ("trilep_topW", lt_red_axis, lambda x: x["signal_region_topW"].sum('charge').integrate('N', slice(0.5,2.5))),
    ]

    if args.scaling == 'LO':
        bsm_scales = {'TTZ': scalePolyLO(x,y)}
    elif args.scaling == 'NLO':
        bsm_scales = {'TTZ': scalePolyNLO(x,y)}
    else:
        bsm_scales = {'TTZ': 1}

    for year in years:
        print(f'=============={year}==============')
        results[year] = {}
        ul = str(year)[2:]
        if year == '2016APV':
            lumi = cfg['lumi'][year]
        else:
            lumi = cfg['lumi'][int(year)]

        for region, axis, get_histo in regions:
            print(f' *  Region: {region}')
            output = outputs[year]

            histogram_incl = None

            signal, backgrounds = make_plot(get_histo(output), year, region, axis, x, y, base_dir, args.systematics, bsm_scales, args.scaling)
            results[year][region] = {'signal':signal, 'backgrounds':backgrounds}

    # also plot for all years combined
    if args.year == 'all':
        print('==============all==============')
        lumi = 0
        for l in cfg['lumi']:
            lumi += cfg['lumi'][l]
        bg_list = ['signal','rare','diboson','conv','nonprompt','TTZ','TTH','TTW']
        for region, axis, get_histo in regions:
            print(f' * Region: {region}')
            signals = []
            for year in years:
                signals.append(results[year][region]['signal'])
            signal = accumulate(signals)
            del signals

            backgrounds = {}
            for bg in bg_list:
                bgs = []
                for year in years:
                    bgs.append(results[year][region]['backgrounds'][bg])
                backgrounds[bg] = accumulate(bgs)
                del bgs
                 
            print ("Making plot")
            print ("...prepping the plots")
            global hist_list
            hist_list = [
                backgrounds['signal'],
                backgrounds['rare'],
                backgrounds['diboson'],
                backgrounds['conv'],
                backgrounds['nonprompt'],
                backgrounds['TTZ'],
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

            ax.legend(ncol=3)
            # labels
            rax.set_xlabel(backgrounds['signal'].sum('dataset').axes()[0].label)
            ax.set_xlim(edges[0],edges[-1])
            rax.set_ylabel(r'rel. unc.')
            ax.set_ylabel(r'Events')

            print ("...storing plots")

            plot_name = f'BIT_cpt_{x}_cpqm_{y}_{region}_ALLYEARS_{args.scaling}'
            fig.savefig(f'{base_dir}/{plot_name}.png')
            fig.savefig(f'{base_dir}/{plot_name}.pdf')

            plt.close(fig)
            del fig, ax, rax

    print('Done!')
