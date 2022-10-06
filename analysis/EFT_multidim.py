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

import numpy as np
import pandas as pd

from coffea import hist
from coffea.processor import accumulate

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot, colors, finalizePlotDir
from Tools.helpers import make_bh
from Tools.config_helpers import get_cache, loadConfig, data_pattern
from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
from Tools.yahist_to_root import yahist_to_root
from Tools.dataCard import dataCard

from Tools.HyperPoly import HyperPoly 
from Tools.limits import regroup_and_rebin, get_systematics
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
    except KeyError:
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

if __name__ == '__main__':


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_scan', action='store_true', help="Run the full 2D scan")
    argParser.add_argument('--inclusive', action='store_true', help="Run the inclusive region")
    argParser.add_argument('--comparison', action='store_true', help="Plot a comparison")
    argParser.add_argument('--overwrite', action='store_true', help="Overwrite existing cards")
    argParser.add_argument('--bit', action='store_true', help="Use boosted information tree (LT otherwise)")
    argParser.add_argument('--fit', action='store_true', help="Run combine (otherwise just plotting)")
    argParser.add_argument('--workers', action='store', default=5, type=int, help="Define how many cores/workers can be used for fitting")
    argParser.add_argument('--year', action='store', default=2016, type=str, help="Select years, comma separated")
    argParser.add_argument('--cpt', action='store', default=0, type=int, help="If run_scan is used, this is the cpt value that's being evaluated")
    argParser.add_argument('--cpqm', action='store', default=0, type=int, help="If run_scan is used, this is the cpqm value that's being evaluated")
    argParser.add_argument('--uaf', action='store_true', help="Store in different directory if on uaf.")
    argParser.add_argument('--scaling', action='store_true', help="run with scaling on")

    args = argParser.parse_args()

    run_scan = args.run_scan
    inclusive = args.inclusive
    bit = args.bit
    fit = args.fit
    comparison = args.comparison

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from Tools.config_helpers import get_merged_output, load_yaml

    #ref_values = 'cpt_0p_cpqm_0p_nlo'
    ref_values = [0,0]


    cfg = loadConfig()
    if not args.uaf:
        plot_dir = './plots/multidim_fits_v2/'
        #plot_dir = os.path.expandvars(cfg['meta']['plots']) + '/multidim_fits_v2/'
    else:
        plot_dir = '/home/users/sjeon/public_html/tW_scattering/multidim_fits/'
    finalizePlotDir(plot_dir)

    # FIXME placeholder systematics....
    systematics= [
        ('signal_norm', 1.1, 'signal'),
        ('TTW_norm', 1.15, 'TTW'),
        ('TTZ_norm', 1.10, 'TTZ'),
        ('TTH_norm', 1.20, 'TTH'),
        ('conv_norm', 1.20, 'conv'),
        ('diboson_norm', 1.20, 'diboson'),
        ('nonprompt_norm', 1.30, 'nonprompt'),
        ('rare_norm', 1.30, 'rare'),
    ]

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
        xr = np.arange(-7,8,2)  # FIXME: switch to steps of 1
        yr = np.arange(-7,8,2)
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
    
    if True:
        # FIXME resolve the selection of what to do in this script...
        years = args.year.split(',')

        if args.overwrite:
            # load outputs (coffea histograms)
            # histograms are created per sample,
            # x-secs and lumi scales are applied on the fly below
            outputs = {}

            for year in years:
                outputs[year] = get_merged_output(
                    'SS_analysis',
                    year,
                    select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'] if args.bit else ['LT', 'LT_SR_pp', 'LT_SR_mm'],
                )#, date='20220624')

        results = {}

        for x,y in scan:

            print (f"Working on point {x}, {y}")

            if bit:

                sm_point = f'cpt_{x}_cpqm_{y}'
                bsm_point = f'bsm_cpt_{x}_cpqm_{y}'
                sm_bkg = f'cpt_{x}_cpqm_{y}'

                print (sm_point, bsm_point, sm_bkg)

                if inclusive:
                    regions = [
                        ("bit_score_incl", bit_axis),
                    ]

                else:
                    regions = [
                        ("bit_score_pp", bit_axis),
                        ("bit_score_mm", bit_axis),
                    ]
            else:
                sm_point = f'eft_cpt_0_cpqm_0'
                bsm_point = f'eft_cpt_{x}_cpqm_{y}'
                sm_bkg = 'central'

                if inclusive:
                    regions = [
                        ("LT", lt_axis),
                    ]
                else:
                    regions = [
                        ("LT_SR_pp", lt_axis),
                        ("LT_SR_mm", lt_axis),
                    ]

            sm_cards = {}
            bsm_cards = {}

            for year in years:

                if year == '2016APV':
                    lumi = cfg['lumi'][year]
                else:
                    lumi = cfg['lumi'][int(year)]

                if args.overwrite:
                    output = outputs[year]

                    if not bit:
                        #eft_mapping = {k[0]:k[0] for k in output['LT_SR_pp'].values().keys() if 'cpt' in k[0]}  # not strictly necessary
                        weights = [ k[0] for k in output['LT_SR_pp'].sum('dataset', 'systematic', 'prediction').values() if '_nlo' in k[0] ]
                        print (weights)

                        hp = HyperPoly(order=2)
                        hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

                        coeff = {}


                for region, axis in regions:

                    histo_name = region
                    plot_name_short = f"BIT_cpt_{x}_cpqm_{y}" if bit else f"LT_cpt_{x}_cpqm_{y}"
                    plot_name = plot_name_short + f'_{region}_{year}'

                    if args.overwrite:
                        print ("Filling background histogram")
                        backgrounds = {
                            'signal':    output[histo_name][('topW_lep', sm_point, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'TTW':       output[histo_name][('TTW', sm_bkg, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'TTH':       output[histo_name][('TTH', sm_bkg, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'TTZ':       output[histo_name][('TTZ', sm_bkg, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'rare':      output[histo_name][('rare', sm_bkg, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'diboson':   output[histo_name][('diboson', sm_bkg, 'central', 'central')].sum('EFT','systematic','prediction').copy(),
                            'conv':      output[histo_name][(wildcard, sm_bkg, 'conv_mc', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
                            'nonprompt': output[histo_name][(wildcard, sm_bkg, 'np_est_data', 'central')].sum('systematic', 'EFT', 'prediction').copy(),
                           }

                        for p in backgrounds.keys():
                            backgrounds[p] = backgrounds[p].rebin(axis.name, axis)

                        total = backgrounds['signal'].copy()
                        total.clear()
                        for k in backgrounds.keys():
                            if not k == 'signal':
                                total.add(backgrounds[k])

                        print ("Filling data histogram. I can still stay blind!")
                        observation = output[histo_name][(data_pattern, sm_bkg, 'central', 'central')].sum('dataset', 'EFT', 'systematic', 'prediction').copy()
                        observation = observation.rebin(axis.name, axis)
                        # unblind the first 8 bins. this is hacky.
                        unblind = observation._sumw[()][:8]
                        blind   = np.zeros_like(observation._sumw[()][8:])
                        observation._sumw[()] = np.concatenate([unblind, blind])

                        if not bit:
                            coeff[region] = hp.get_parametrization(
                                [output[histo_name]['topW_lep'].rebin(axis.name, axis)
                                 .integrate('prediction', 'central')
                                 .integrate('systematic', 'central')
                                 .integrate('EFT', w)
                                 .sum('dataset')
                                 .values()[()] for w in weights],
                            )
                            central, sumw2 = output[histo_name]['topW_lep'].rebin(axis.name, axis).integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', weights[0]).sum('dataset').values(sumw2=True)[()]
                            bsm_vals = hp.eval(coeff[region], [x,y])

                            ## NOTE: this is an over-optimistic simplification of sumw2.
                            ## we can't really use this for a real analysis because the weights are not homogenuous
                            ## def make_bh(sumw, sumw2, edges):
                            #signal = make_bh(
                            #    bsm_vals,
                            #    2*(np.sqrt(sumw2)*bsm_vals/central)**2,
                            #    axis.edges(),
                            #)

                            # NOTE: this is the "correct" way, but we need all the different histograms filled
                            # already from the coffea processor.
                            signal = output[histo_name]['topW_lep'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', bsm_point).copy()
                            signal = signal.rebin(axis.name, axis)
                            #signal = signal.sum('dataset').to_hist()

                        else:
                            print ("Filling signal histogram")
                            signal = output[histo_name][('topW_lep', bsm_point, 'central', 'central')].sum('systematic', 'EFT', 'prediction').copy()
                            signal = signal.rebin(axis.name, axis)
                            #signal = signal.sum('dataset').to_hist()

                        # NOTE get EFT point dependend systematics here




                        # NOTE make some nice plots here
                        #
                        print ("Making first plots")
                        print ("...prepping the plots")
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

                    if fit:
                        if args.overwrite:
                            if args.scaling:
                                LO_scales = {'TTZ': scalePolyLO(x,y)}
                                NLO_scales = {'TTZ': scalePolyNLO(x,y)}
                            else:
                                LO_scales = {'TTZ': 1}
                                NLO_scales = {'TTZ': 1}
                            sm_card = makeCardFromHist(
                                backgrounds,
                                ext=f'SM_{plot_name}',
                                #scales = scales,
                                bsm_scales = LO_scales,
                                systematics = systematics,
                                data = observation,
                                blind = True,
                               )

                            #bsm_hist_for_card = signal if not bit else signal.sum('dataset').to_hist()
                            bsm_hist_for_card = signal.sum('dataset').to_hist()
                            bsm_card = makeCardFromHist(
                                backgrounds,
                                ext=f'BSM_{plot_name}',
                                bsm_hist = bsm_hist_for_card,
                                #scales = scales,
                                bsm_scales = NLO_scales,
                                systematics = systematics,
                                data = observation,
                                blind = True,
                               )
                            bsm_cards[f'BSM_{plot_name}'] = bsm_card
                            sm_cards[f'SM_{plot_name}'] = sm_card
                        else:
                            # FIXME: the paths should not be hard coded
                            bsm_cards[f'BSM_{plot_name}'] = f'/{card_dir}/BSM_{plot_name}_card.txt'
                            sm_cards[f'SM_{plot_name}'] = f'/{card_dir}/SM_{plot_name}_card.txt'


            if fit:
                # FIXME check that this actually works - running years individually and then just combining
                # this avoids having to load all the histograms at once
                print ("Combining cards:")
                print (sm_cards)
                sm_card_combined = card.combineCards(sm_cards, name=f'SM_{plot_name_short}.txt')
                bsm_card_combined = card.combineCards(bsm_cards, name=f'BSM_{plot_name_short}.txt')

                all_cards.append(sm_card_combined)
                all_cards.append(bsm_card_combined)
                #results[(x,y)] = nll
            else:

                results[(x,y)] = 0

        X, Y = np.meshgrid(xr, yr)
        scan = zip(X.flatten(), Y.flatten())

        if fit:
            import concurrent.futures
            workers = args.workers

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
            z = []
            for x, y in results:
                point = [x, y]
                z.append(results[(x,y)])


            Z = np.array(z)
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

            fig.savefig('./scan_test_bit_v3.png')
            fig.savefig('./scan_test_bit_v3.pdf')

            out_path = os.path.expandvars(cfg['caches']['base'])
            if bit:
                out_path += 'results_bit.pkl'
            else:
                out_path += 'results_lt.pkl'

            with open(out_path, 'wb') as f:
                pickle.dump(results, f)


        # NOTE re-init dataCard here just so that we always clean up the right dir...
        #card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
        card = dataCard(releaseLocation=os.path.expandvars('$TWHOME/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
        card.cleanUp()

    elif comparison:
        # NOTE this loads results and draws a comparison plot.
        # kept for legacy, might be broken by now
        out_path = os.path.expandvars(cfg['caches']['base'])
        bit_path = out_path + 'results_bit.pkl'
        lt_path = out_path + 'results_lt.pkl'

        with open(bit_path, 'rb') as f:
            results_bit = pickle.load(f)
        with open(lt_path, 'rb') as f:
            results_lt = pickle.load(f)

        z_bit = []
        z_lt = []
        for x, y in results_bit:
            point = [x, y]
            z_bit.append(results_bit[(x,y)])
            z_lt.append(results_lt[(x,y)])


        Z_bit = np.array(z_bit)
        Z_bit = np.reshape(Z_bit, X.shape)

        Z_lt = np.array(z_lt)
        Z_lt = np.reshape(Z_lt, X.shape)

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

        CS_bit = ax.contour(X, Y, Z_bit, levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                     linestyles='dashed',linewidths=(4,))

        CS_lt = ax.contour(X, Y, Z_lt, levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                     linestyles='solid',linewidths=(4,))

        fmt_bit = {}
        strs_bit = ['BIT, 68%', 'BIT, 95%']
        for l, s in zip(CS_bit.levels, strs_bit):
            fmt_bit[l] = s

        fmt_lt = {}
        strs_lt = ['LT, 68%', 'LT, 95%']
        for l, s in zip(CS_lt.levels, strs_lt):
            fmt_lt[l] = s


        # Label every other level using strings
        ax.clabel(CS_bit, CS_bit.levels, inline=True, fmt=fmt_bit, fontsize=10)
        ax.clabel(CS_lt, CS_lt.levels, inline=True, fmt=fmt_lt, fontsize=10)

        plt.show()

        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_comparison.png')
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_comparison.pdf')

    else:
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
