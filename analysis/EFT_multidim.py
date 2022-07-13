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

import numpy as np
import pandas as pd

from coffea import hist
from coffea.processor import accumulate

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot
from Tools.helpers import make_bh
from Tools.config_helpers import get_cache
from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
from Tools.yahist_to_root import yahist_to_root
from Tools.dataCard import dataCard

from Tools.HyperPoly import HyperPoly 
from Tools.limits import regroup_and_rebin, get_systematics
from Tools.EFT_tools import make_scan

from yahist import Hist1D
'''
Taken from the NanoAOD-tools module
'''

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

def get_NLL(
        sm_card,
        bsm_card,
    ):

    #pt_bins     = hist.Bin('pt', r'$p_{T}\ (GeV)$', [0,100,150,200,400])
    #ht_bins     = hist.Bin('ht', r'$H_{T}\ (GeV)$', [100,200,300,400,500,600,700,800])
    #score_bins  = hist.Bin("score",          r"N", 8, 0, 1)
    #N_bins      = hist.Bin("multiplicity",   r"N", 3, 1.5, 4.5)

    #mapping = {
    #    'rare': ['rare', 'diboson'],
    #    'TTW': ['TTW'],
    #    'TTZ': ['TTZ'],
    #    'TTH': ['TTH'],
    #    'ttbar': ['ttbar'],
    #    'nonprompt': ['np_est_mc'],
    #    'chargeflip': ['cf_est_mc'],
    #    'conversion': ['conv_mc'],
    #    'signal': ['topW_v3'],
    #}

    #ref_point = 'cpt_0p_cpqm_0p'
    #ref_values = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]

    #regions = [
    #    ('%s_SR_1'%year, 'LT_SR_pp'),
    #    ('%s_SR_2'%year, 'LT_SR_mm'),
    #    #('%s_CR'%year, 'node1_score'),
    #    #('%s_CR_norm'%year, 'node'),
    #]


    print (nll)

    return nll


'''

        # then make copies for SR and CR
        new_hists = {}
        for short_name, long_name in regions:
            new_hists['norm'] = output['norm']  # will need this later on
            for x in output.keys():
                if x.startswith('_'):
                    new_hists[x] = output[x]

            all_hists = [ x for x in output.keys() if long_name in x ]  # get all the histograms
            for h in all_hists:
                new_name = h.replace(long_name, short_name)
                new_hists[new_name] = output[h][no_data_or_signal]
                new_hists[new_name] = regroup_and_rebin(new_hists[new_name], ht_bins, mapping)

        correlated = False  # switch on correlations for JES/b/light uncertainties

        output_EFT = get_cache('EFT_ctW_scan_%s'%year)

        eft_mapping = {k[0]:k[0] for k in output_EFT['LT_SR_pp'].values().keys() if 'topW_full_EFT_ctZ' in k[0]}  # not strictly necessary
        weights = [ k[0].replace('topW_full_EFT_','').replace('_nlo','') for k in output_EFT['LT_SR_pp'].values().keys() if 'topW_full_EFT_ctZ' in k[0] ]

        hp = HyperPoly(order=2)
        hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

        coeff = {}
        for region, hist_name in regions:
            output_EFT[hist_name] = regroup_and_rebin(output_EFT[hist_name], ht_bins, eft_mapping)
            print (hist_name)
            coeff[region] = hp.get_parametrization( [histo_values(output_EFT[hist_name], 'topW_full_EFT_%s_nlo'%w) for w in weights] )

        pp_val, pp_unc = output_EFT['LT_SR_pp']['topW_full_EFT_%s_nlo'%weights[0]].sum('dataset').values(sumw2=True, overflow='all')[()]

        bsm_vals = hp.eval(coeff['%s_SR_1'%year], point)
        bsm_hist = Hist1D.from_bincounts(
            bsm_vals,
            ht_bins.edges(overflow='all'),
            errors = (np.sqrt(pp_unc)/pp_val)*bsm_vals,
        )

        sm_hist = Hist1D.from_bincounts(
            pp_val,
            ht_bins.edges(overflow='all'),
            errors = np.sqrt(pp_unc),
        )

        # FIXME need to fix the stat uncertainties for the signal
        # FIXME overflow should be correct, but needs to be double checked
        sys = get_systematics(new_hists, '%s_SR_1'%year, year, correlated=correlated, signal=True) if systematics else False
        bsm_card_sr1 = makeCardFromHist(
            new_hists,
            '%s_SR_1'%year,
            overflow='all',
            ext='_BSM',
            bsm_vals =  bsm_hist,
            sm_vals =  sm_hist,
            scales = scales,
            bsm_scales = bsm_scales,
            systematics = sys,
        )

        pp_val, pp_unc = output_EFT['LT_SR_mm']['topW_full_EFT_%s_nlo'%weights[0]].sum('dataset').values(sumw2=True, overflow='all')[()]

        bsm_vals = hp.eval(coeff['%s_SR_2'%year], point)
        bsm_hist = Hist1D.from_bincounts(
            bsm_vals,
            ht_bins.edges(overflow='all'),
            errors = (np.sqrt(pp_unc)/pp_val)*bsm_vals,
        )

        sm_hist = Hist1D.from_bincounts(
            pp_val,
            ht_bins.edges(overflow='all'),
            errors = np.sqrt(pp_unc),
        )

        sys = get_systematics(new_hists, '%s_SR_2'%year, year, correlated=correlated, signal=True) if systematics else False
        bsm_card_sr2 = makeCardFromHist(
            new_hists,
            '%s_SR_2'%year,
            overflow='all',
            ext='_BSM',
            bsm_vals = bsm_hist,
            sm_vals = sm_hist,
            scales = scales,
            bsm_scales = bsm_scales,
            systematics = sys,
            )

        res_bsm_data_cards.update({
            '%s_SR1'%year: bsm_card_sr1,
            '%s_SR2'%year: bsm_card_sr2,
        })

    combined_res_bsm = card.combineCards(res_bsm_data_cards, name='combined_card_BSM.txt')
    res_bsm = card.calcNLL(combined_res_bsm)

    print ("This took %.2f seconds"%(time.time()-start_time))

    return res_bsm['nll0'][0]+res_bsm['nll'][0]
'''


if __name__ == '__main__':

    run_scan = True

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from Tools.config_helpers import get_merged_output, load_yaml

    #ref_values = 'cpt_0p_cpqm_0p_nlo'
    ref_values = [0,0]

    inclusive = False
    bit = True

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
    bit_axis     = hist.Bin("bit",           r"N",         20,0,1)



    card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

    # Define a scan
    x = np.arange(-7,8,2)
    y = np.arange(-7,8,2)
    X, Y = np.meshgrid(x, y)
    scan = zip(X.flatten(), Y.flatten())

    if run_scan:


        #years = ['2016', '2016APV', '2017', '2018']
        years = ['2018']

        # load outputs
        outputs = {}

        for year in years:
            outputs[year] = get_merged_output('SS_analysis', year)#, date='20220624')

        results = {}

        for x,y in scan:

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
                #sm_point = 'cpt_0p_cpqm_0p_nlo'
                #bsm_point = 'cpt_6p_cpqm_0p_nlo'  # NOTE: redundant
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


            for year in years:
                output = outputs[year]

                if not bit:
                    #eft_mapping = {k[0]:k[0] for k in output['LT_SR_pp'].values().keys() if 'cpt' in k[0]}  # not strictly necessary
                    weights = [ k[0] for k in output['LT_SR_pp'].sum('dataset', 'systematic', 'prediction').values() if '_nlo' in k[0] ]
                    print (weights)

                    hp = HyperPoly(order=2)
                    hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

                    coeff = {}

                sm_cards = {}
                bsm_cards = {}

                for region, axis in regions:

                    histo_name = region

                    backgrounds = {
                        'signal':    output[histo_name]['topW_lep'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_point).copy(),
                        'TTW':       output[histo_name]['TTW'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'TTH':       output[histo_name]['TTH'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'TTZ':       output[histo_name]['TTZ'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'rare':      output[histo_name]['rare'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'diboson':   output[histo_name]['diboson'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'conv':      output[histo_name].integrate('prediction', 'conv_mc').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                        'nonprompt': output[histo_name].integrate('prediction', 'np_est_mc').integrate('systematic', 'central').integrate('EFT', sm_bkg).copy(),
                    }

                    for p in backgrounds.keys():
                        backgrounds[p] = backgrounds[p].rebin(axis.name, axis)

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
                        # already from the processor.
                        signal = output[histo_name]['topW_lep'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', bsm_point).copy()
                        signal = signal.rebin(axis.name, axis)
                        signal = signal.sum('dataset').to_hist()

                    else:
                        signal = output[histo_name]['topW_lep'].integrate('prediction', 'central').integrate('systematic', 'central').integrate('EFT', bsm_point).copy()
                        signal = signal.rebin(axis.name, axis)

                    #print (signal.values())

                    sm_card = makeCardFromHist(
                        backgrounds,
                        ext=f'MultiClass_SM_{region}_{year}',
                        #scales = scales,
                        #bsm_scales = bsm_scales,
                        systematics = systematics,
                    )
                    sm_cards[f'{region}_{year}'] = sm_card

                    bsm_hist_for_card = signal if not bit else signal.sum('dataset').to_hist()
                    bsm_card = makeCardFromHist(
                        backgrounds,
                        ext=f'MultiClass_BSM_{region}_{year}',
                        bsm_hist = bsm_hist_for_card,
                        #scales = scales,
                        #bsm_scales = bsm_scales,
                        systematics = systematics,
                    )
                    bsm_cards[f'{region}_{year}'] = bsm_card

            sm_card_combined = card.combineCards(sm_cards)
            res_sm = card.calcNLL(sm_card_combined)
            nll_sm = res_sm['nll0'][0] + res_sm['nll'][0]

            bsm_card_combined = card.combineCards(bsm_cards)
            res_bsm = card.calcNLL(bsm_card_combined)
            nll_bsm = res_bsm['nll0'][0] + res_bsm['nll'][0]

            nll = -2*(nll_sm - nll_bsm)

            print (nll)
            #res_sm = get_NLL(years=years, point=[0,0])

            results[(x,y)] = nll


        z = []
        for x, y in results:
            point = [x, y]
            z.append(results[(x,y)])


        Z = np.array(z)
        Z = np.reshape(Z, X.shape)

        fig, ax, = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(
            "Preliminary",
            data=True,
            #year=2018,
            lumi=138,
            loc=0,
            ax=ax,
        )

        CS = ax.contour(X, Y, Z, levels = [2.28, 5.99], colors=['blue', 'red'], # 68/95 % CL
                     linestyles=('-',),linewidths=(4,))
        fmt = {}
        strs = ['68%', '95%']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        # Label every other level using strings
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

        plt.show()
        
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_test_bit.png')
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_test_bit.pdf')


        card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
        card.cleanUp()

        raise NotImplementedError

    else:

        ht_bins     = hist.Bin('ht', r'$H_{T}\ (GeV)$', [100,200,300,400,500,600,700,800])

        hist_name = 'LT_SR_mm'

        output_EFT = get_cache('EFT_cpt_scan_2017')
        eft_mapping = {k[0]:k[0] for k in output_EFT[hist_name].values().keys() if 'topW_full_EFT_ctZ' in k[0]}  # not strictly necessary
        weights = [ k[0].replace('topW_full_EFT_','').replace('_nlo','') for k in output_EFT[hist_name].values().keys() if 'topW_full_EFT_ctZ' in k[0] ]

        output_EFT[hist_name] = regroup_and_rebin(output_EFT[hist_name], ht_bins, eft_mapping)
        
        ref_point = 'ctZ_2p_cpt_4p_cpQM_4p_cpQ3_4p_ctW_2p_ctp_2p'
        ref_values = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]

        hp = HyperPoly(order=2)
        hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

        coeff = hp.get_parametrization( [histo_values(output_EFT[hist_name], 'topW_full_EFT_%s_nlo'%w) for w in weights] )


        operator = 'cpQM'
        if operator == 'cpt':
            x_label = r'$C_{\varphi t}$'
        elif operator == 'cpQM':
            x_label = r'$C_{\varphi Q}^{-}$'

        # just an example.
        points = make_scan(operator=operator, C_min=-10, C_max=10, step=1)

        for i in range(0,21):
            print (i-10, hp.eval(coeff, points[i]['point']))

        pred_matrix = np.array([ np.array(hp.eval(coeff,points[i]['point'])) for i in range(21) ])

        # plot the increase in yield 
        
        fig, ax = plt.subplots()
        
        hep.cms.label(
            "Work in progress",
            data=True,
            #year=2018,
            lumi=60.0+41.5+35.9,
            loc=0,
            ax=ax,
        )
        
        plt.plot([i-10 for i in range(21)], np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[10,:]), label=r'inclusive', c='green')
        
        plt.xlabel(x_label)
        plt.ylabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()
        
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/%s_scaling.pdf'%operator)
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/%s_scaling.png'%operator)
 

        fig, ax = plt.subplots()
        
        hep.cms.label(
            "Work in progress",
            data=True,
            #year=2018,
            lumi=60.0+41.5+35.9,
            loc=0,
            ax=ax,
        )
        
        plt.plot([i-10 for i in range(21)], np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[10,:]), label=r'inclusive', c='green')
        plt.plot([i-10 for i in range(21)], np.sum(pred_matrix[:,7:], axis=1)/np.sum(pred_matrix[10,7:]), label=r'$L_{T} \geq 700\ GeV$', c='blue')
        
        plt.xlabel(x_label)
        plt.ylabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()
        
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/%s_scaling_tail.pdf'%operator)
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/%s_scaling_tail.png'%operator)
   
