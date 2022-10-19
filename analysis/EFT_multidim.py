'''
This should run the final analyis:
- pick up cached histograms
- rebin distributions
- create inputs for data card
- run fits
'''

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

def write_card(output, year, region, axis, cpt, cpqm,
               plot_dir='./',
               systematics=True,
               bsm_scales={},
               ):

    x = cpt
    y = cpqm
    sm_point  = f"cpt_{x}_cpqm_{y}"
    bsm_point = f"bsm_cpt_{x}_cpqm_{y}"

    histo_name = region
    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
    plot_name = plot_name_short + f'_{region}_{year}'

    ul = str(year)[2:]

    print ("Filling background histogram")
    backgrounds = {
        'signal':    output[histo_name][('topW_lep', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTW':       output[histo_name][('TTW', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTH':       output[histo_name][('TTH', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'TTZ':       output[histo_name][('TTZ', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'rare':      output[histo_name][('rare', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'diboson':   output[histo_name][('diboson', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
        'conv':      output[histo_name][(wildcard, sm_point, 'conv_mc', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
        'nonprompt': output[histo_name][(wildcard, sm_point, 'np_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
        }

    # NOTE some complication for charge flip.
    if region.count('_pp') or region.count('_mm'):
        backgrounds['chargeflip'] = output[histo_name.replace('pp', 'incl').replace('mm', 'incl')][(wildcard, sm_point, 'cf_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy()
        backgrounds['chargeflip'].scale(0.5)
    else:
        backgrounds['chargeflip'] = output[histo_name][(wildcard, sm_point, 'cf_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy()

    for p in backgrounds.keys():
        backgrounds[p] = backgrounds[p].rebin(axis.name, axis)

    total = backgrounds['signal'].copy()
    total.clear()
    for k in backgrounds.keys():
        if not k == 'signal':
            total.add(backgrounds[k])

    print ("Filling data histogram. I can still stay blind!")
    observation = output[histo_name][(data_pattern, sm_point, 'central', 'central')].sum('dataset', 'EFT', 'systematic', 'prediction').copy()
    observation = observation.rebin(axis.name, axis)
    # unblind the first 8 bins. this is hacky.
    unblind = observation._sumw[()][:8]
    blind   = np.zeros_like(observation._sumw[()][8:])
    observation._sumw[()] = np.concatenate([unblind, blind])

    print ("Filling signal histogram")
    signal = output[histo_name][('topW_lep', bsm_point, 'central', 'central')].sum('systematic', 'EFT', 'prediction').copy()
    signal = signal.rebin(axis.name, axis)

    if systematics:
        # NOTE this loads background systematics.
        # Not fully complete, but most important systematics are here
        print ("Getting Background systematics")
        systematics = get_systematics(output, histo_name, year, sm_point,
                                        correlated=False,
                                        signal=False,
                                        overflow='none',
                                        samples=samples[year],
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
        systematics = add_signal_systematics(output, histo_name, year, sm_point,
                                                systematics=systematics,
                                                correlated=False,
                                                proc='topW_lep',
                                                overflow='none',
                                                samples=samples[year],
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

    sm_card = makeCardFromHist(
        backgrounds,
        ext=f'SM_{plot_name}',
        #scales = scales,
        #bsm_scales = bsm_scales,
        systematics = systematics,
        data = observation,
        blind = True,
        )

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
    argParser.add_argument('--inclusive', action='store_true', help="Run the inclusive region")
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

    args = argParser.parse_args()

    run_scan = args.run_scan
    inclusive = args.inclusive
    bit = args.bit
    fit = args.fit
    comparison = args.comparison

    import concurrent.futures
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from Tools.config_helpers import get_merged_output, load_yaml

    #ref_values = 'cpt_0p_cpqm_0p_nlo'
    ref_values = [0,0]


    cfg = loadConfig()

    # set directories to save to
    if not args.uaf:
        base_dir = './plots/'
    else:
        base_dir = '/home/users/sjeon/public_html/tW_scattering/'
    dir_name = 'multidim_fits'
    if not args.scaling:
        base_dir = base_dir+dir_name
    else:
        base_dir = base_dir+dir_name+'_scaled'

    dump_dir = './results/' # where to save json files
    finalizePlotDir(dump_dir)

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
    #bit_axis     = hist.Bin("bit",           r"N",               10, 0, 1)
    #bit_axis     = hist.Bin("bit",           r"N",         [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    bit_axis     = hist.Bin("bit",           r"BIT score",         20,0,1)


    all_cards = []
    #card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    card_dir = os.path.expandvars('$TWHOME/data/cards/')

    # Define a scan
    if args.run_scan:
        xr = np.arange(-7,8,1)
        yr = np.arange(-7,8,1)
    else:
        xr = np.array([int(args.cpt)])
        yr = np.array([int(args.cpqm)])

    X, Y = np.meshgrid(xr, yr)
    scan = zip(X.flatten(), Y.flatten())

    # Define Scaling Polynomial
    def scalePolyNLO(xt, xQM):
        return 1 + 0.072813*xt - 0.098492*xQM + 0.005049*xt**2 - 0.002042*xt*xQM + 0.003988*xQM**2

    def scalePolyLO(xt, xQM):
        return 1 + 0.068485*xt - 0.104991*xQM + 0.003982*xt**2 - 0.002534*xt*xQM + 0.004144*xQM**2
    
    years = args.year.split(',')

    if args.overwrite:
        # load outputs (coffea histograms)
        # histograms are created per sample,
        # x-secs and lumi scales are applied on the fly below
        outputs = {}
        samples = {}
        mapping = load_yaml(data_path+"nano_mapping.yaml")

        for year in years:
            ul = str(year)[2:]
            samples[year] = get_samples(f"samples_UL{ul}.yaml")
            outputs[year] = get_merged_output(
                'SS_analysis',
                year,
                select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'] if args.bit else ['LT', 'LT_SR_pp', 'LT_SR_mm'],
            )#, date='20220624')

    results = {}

    sm_cards = {}
    bsm_cards = {}

    cards_to_write = []

    for x,y in scan:

        print (f"Working on point {x}, {y}")

        if inclusive:
            regions = [
                ("bit_score_incl", bit_axis),
            ]

        else:
            regions = [
                ("bit_score_pp", bit_axis),
                ("bit_score_mm", bit_axis),
            ]

        sm_cards[(x,y)] = {}
        bsm_cards[(x,y)] = {}
        
        if args.scaling == 'LO':
            bsm_scales = {'TTZ': scalePolyLO(x,y)}
        elif args.scaling == 'NLO':
            bsm_scales = {'TTZ': scalePolyNLO(x,y)}
        else:
            bsm_scales = {'TTZ': 1}

        for year in years:
            plot_dir = base_dir + '_' + year + '/'
            finalizePlotDir(plot_dir)

            ul = str(year)[2:]
            if year == '2016APV':
                lumi = cfg['lumi'][year]
            else:
                lumi = cfg['lumi'][int(year)]

            if args.overwrite:
                output = outputs[year]

            for region, axis in regions:

                if args.overwrite:
                    cards_to_write.append((output, year, region, axis, x, y, plot_dir, True, bsm_scales))
                #bsm_card, sm_card = write_card(output, year, region, axis, x, y,
                #                               plot_dir='./',
                #                               systematics=True,
                #                               )

                plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
                plot_name = plot_name_short + f'_{region}_{year}'
                bsm_cards[(x,y)][f'BSM_{plot_name}'] = f'/{card_dir}/BSM_{plot_name}_card.txt'
                sm_cards[(x,y)][f'SM_{plot_name}']   = f'/{card_dir}/SM_{plot_name}_card.txt'

    workers = args.workers
    if args.overwrite:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for card_name, result in zip(cards_to_write, executor.map(write_card_wrapper, cards_to_write)):
                print (f"Done with {card_name}")

    X, Y = np.meshgrid(xr, yr)
    scan = zip(X.flatten(), Y.flatten())

    for x,y in scan:

        plot_name_short = f"BIT_cpt_{x}_cpqm_{y}"# if bit else f"LT_cpt_{x}_cpqm_{y}"
        plot_name = plot_name_short + f'_{region}_{year}'
        # NOTE running years individually and then just combining
        # this avoids having to load all the histograms at once
        print ("Combining cards:")
        print (sm_cards[(x,y)])
        sm_card_combined = card.combineCards(sm_cards[(x,y)], name=f'SM_{plot_name_short}.txt')
        bsm_card_combined = card.combineCards(bsm_cards[(x,y)], name=f'BSM_{plot_name_short}.txt')

        all_cards.append(sm_card_combined)
        all_cards.append(bsm_card_combined)

    X, Y = np.meshgrid(xr, yr)
    scan = zip(X.flatten(), Y.flatten())

    if fit:

        print ("Done with the pre-processing and data card making, running fits now.")
        print (f"Using {workers} workers")

        all_nll = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for card_name, result in zip(all_cards, executor.map(get_NLL, all_cards)):

                #print (val, result)
                all_nll[card_name.split('/')[-1].strip('.txt')] = result

        for x,y in scan:
            plot_name_short = f"BIT_cpt_{x}_cpqm_{y}" if bit else f"LT_cpt_{x}_cpqm_{y}"
            print (plot_name_short)
            results[(x,y)] = -2*(all_nll[f'SM_{plot_name_short}'] - all_nll[f'BSM_{plot_name_short}'])

    if fit and len(xr)>4:
        Z = np.array(list(results.values()))
        Z = np.reshape(Z, X.shape)

        fig, ax, = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(
            "Work in progress",
            data=True,
            #year=2018,
            lumi=60,
            loc=0,
            ax=ax,
            )

        ax.set_ylim(-8.1, 8.1)
        ax.set_xlim(-8.1, 8.1)

        CS = ax.contour(X, Y, Z, levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                        linestyles=('-',),linewidths=(4,))
        fmt = {}
        strs = ['68%', '95%']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

        plt.show()

        fig.savefig(plot_dir+'scan_test_bit.png')
        fig.savefig(plot_dir+'scan_test_bit.pdf')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if bit:
            out_file = f'results_{year}_bit_{timestamp}.json'
        else:
            out_file = f'results_{year}_lt_{timestamp}.json'

        results_dump = {}

        for x,y in results:
            results_dump[f'cpt_{x}_cpqm_{y}'] = results[(x,y)]

        with open(dump_dir + out_file, 'w') as f:
            json.dump(results_dump, f)


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

        fig.savefig(plot_dir+'1D_scaling_test.png')
        fig.savefig(plot_dir+'1D_scaling_test.pdf')



    # NOTE re-init dataCard here just so that we always clean up the right dir...
    #card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
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

        fig.savefig(f'{plot_dir}/inclusive_scaling.png')
        fig.savefig(f'{plot_dir}/inclusive_scaling.pdf')

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

        fig.savefig(f'{plot_dir}/high_E_scaling.png')
        fig.savefig(f'{plot_dir}/high_E_scaling.pdf')
