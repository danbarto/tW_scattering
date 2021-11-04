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

from klepto.archives import dir_archive

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot
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

def get_NLL(years=['2016', '2016APV', '2017', '2018'], point=[0,0,0,0,0,0]):

    start_time = time.time()
    card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TTW/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

    pt_bins     = hist.Bin('pt', r'$p_{T}\ (GeV)$', [0,100,150,200,400])
    ht_bins     = hist.Bin('ht', r'$H_{T}\ (GeV)$', [100,200,300,400,500,600,700,800])
    score_bins  = hist.Bin("score",          r"N", 8, 0, 1)
    N_bins      = hist.Bin("multiplicity",   r"N", 3, 1.5, 4.5)

    mapping = {
        'rare': ['rare', 'diboson'],
        'TTW': ['TTW'],
        'TTZ': ['TTZ'],
        'TTH': ['TTH'],
        'ttbar': ['ttbar'],
        'nonprompt': ['np_est_mc'],
        'chargeflip': ['cf_est_mc'],
        'conversion': ['conv_mc'],
        'signal': ['topW_v3'],
    }

    ref_point = 'ctZ_2p_cpt_4p_cpQM_4p_cpQ3_4p_ctW_2p_ctp_2p'
    ref_values = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]

    res_bsm_data_cards = {}

    for year in years:

        regions = [
            ('%s_SR_1'%year, 'LT_SR_pp'),
            ('%s_SR_2'%year, 'LT_SR_mm'),
            #('%s_CR'%year, 'node1_score'),
            #('%s_CR_norm'%year, 'node'),
        ]

        # SM histograms
        output = get_cache('SS_analysis_%s'%year)
        all_processes = [ x[0] for x in output['N_ele'].values().keys() ]
        data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        data    = data_all
        order   = ['topW_v3', 'np_est_mc', 'conv_mc', 'cf_est_mc', 'TTW', 'TTH', 'TTZ','rare', 'diboson']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]
        no_data_or_signal  = re.compile('(?!(%s))'%('|'.join(omit)))

        # then make copies for SR and CR
        new_hists = {}
        for short_name, long_name in regions:
            new_hists[short_name] = output[long_name][no_data_or_signal]
            if 'LT_SR' in long_name:
                new_hists[short_name] = regroup_and_rebin(new_hists[short_name], ht_bins, mapping)

        correlated = False  # switch on correlations for JES/b/light uncertainties

        output_EFT = get_cache('EFT_cpt_scan_%s'%year)

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
        bsm_card_sr1 = makeCardFromHist(
            new_hists,
            '%s_SR_1'%year,
            overflow='all',
            ext='_BSM',
            bsm_vals =  bsm_hist,
            sm_vals =  sm_hist,
            # get_systematics(output, hist, year, correlated=False, signal=True)
            systematics = get_systematics(output, 'LT_SR_pp', year, correlated=correlated, signal=False), ## FIXME omit signal uncertainties for now 
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

        bsm_card_sr2 = makeCardFromHist(
            new_hists,
            '%s_SR_2'%year,
            overflow='all',
            ext='_BSM',
            bsm_vals = bsm_hist,
            sm_vals = sm_hist,
            # get_systematics(output, hist, year, correlated=False, signal=True)
            systematics = get_systematics(output, 'LT_SR_mm', year, correlated=correlated, signal=False), ## FIXME omit signal uncertainties for now
        )

        res_bsm_data_cards.update({
            '%s_SR1'%year: bsm_card_sr1,
            '%s_SR2'%year: bsm_card_sr2,
        })

    combined_res_bsm = card.combineCards(res_bsm_data_cards, name='combined_card_BSM.txt')
    res_bsm = card.calcNLL(combined_res_bsm)

    print ("This took %.2f seconds"%(time.time()-start_time))

    return res_bsm['nll0'][0]+res_bsm['nll'][0]


if __name__ == '__main__':

    run_scan = True

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    if run_scan:

        #years = ['2016', '2016APV', '2017', '2018']
        years = ['2018']

        res_sm = get_NLL(years=years, point=[0,0,0,0,0,0])

        x = np.arange(-10,11,4)
        y = np.arange(-10,11,4)
        X, Y = np.meshgrid(x, y)

        z = []
        for x, y in zip(X.flatten(), Y.flatten()):
            point = [0, y, x, 0, 0, 0]
            z.append((-2*(res_sm-get_NLL(years=years, point=point))))

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
        
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_test.png')
        fig.savefig('/home/users/dspitzba/public_html/tW_scattering/scan_test.pdf')


        card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TTW/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
        card.cleanUp()


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
   
