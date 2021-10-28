'''
This should run the final analyis:
- pick up cached histograms
- rebin distributions
- create inputs for data card
- run fits
'''

import os
import re

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

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--impacts', action='store_true', default=None, help="Run impacts?")
    argParser.add_argument('--years', action='store', default='2016', help="Which years to include?")
    args = argParser.parse_args()

    years = args.years.split(',')
    run_impacts = args.impacts
    data_cards = {}

    card = dataCard(releaseLocation=os.path.expandvars('/home/users/$USER/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    score_bins = hist.Bin("score",          r"N", 8, 0, 1)
    N_bins     = hist.Bin("multiplicity",   r"N", 3, 1.5, 4.5)

    score_bins_3l = hist.Bin("score",          r"N", 4, 0, 1)
    N_bins_3l     = hist.Bin("multiplicity",   r"N", 2, 1.5, 3.5)

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

    for year in years:
        # load the cache
        output = get_cache('SS_analysis_%s'%year)
        
        # check out that the histogram we want is actually there
        all_processes = [ x[0] for x in output['N_ele'].values().keys() ]
        data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        data    = data_all
        order   = ['topW_v3', 'np_est_mc', 'conv_mc', 'cf_est_mc', 'TTW', 'TTH', 'TTZ','rare', 'diboson']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]
        
        no_data_or_signal  = re.compile('(?!(%s))'%('|'.join(omit)))
        
        regions = [
            ('%s_SR_1'%year, 'node0_score_transform_pp'),
            ('%s_SR_2'%year, 'node0_score_transform_mm'),
            ('%s_CR'%year, 'node1_score'),
            ('%s_CR_norm'%year, 'node'),
        ]

        regions_3l = [
            ('%s_SR_3l'%year, 'node0_score_transform'),
            ('%s_CR_3l'%year, 'node1_score'),
            ('%s_CR_norm_3l'%year, 'node'),
        ]
        
        # FIXME I should just rebin / map everything at the same time.
        from Tools.limits import regroup_and_rebin, get_systematics
        # First, rebin & map
        for k in output.keys():
            if 'node0_score' in k:
                output[k] = regroup_and_rebin(output[k], score_bins, mapping)
            elif 'node1_score' in k:
                output[k] = regroup_and_rebin(output[k], score_bins, mapping)
            elif k.startswith('node') and not k.count('score'):
                output[k] = regroup_and_rebin(output[k], N_bins, mapping)
        
        # then make copies for SR and CR
        new_hists = {}
        for short_name, long_name in regions:
            new_hists[short_name] = output[long_name][no_data_or_signal]

        print ("- scale uncertainties for TTW (total):")
        get_scale_unc(output, 'node0_score_transform_pp', 'TTW', score_bins, quiet=False)
        print ("- scale uncertainties for TTW (shape/acceptance):")
        get_scale_unc(output, 'node0_score_transform_pp', 'TTW', score_bins, quiet=False, keep_norm=True)
                
        print ("- JES uncertainties for TTW:")
        get_unc(output, 'node0_score_transform_pp', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
        get_unc(output, 'node0_score_transform_mm', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
        print ("- JES uncertainties for signal:")
        get_unc(output, 'node0_score_transform_pp', 'signal', '_pt_jesTotal', score_bins, quiet=False)
        get_unc(output, 'node0_score_transform_mm', 'signal', '_pt_jesTotal', score_bins, quiet=False)
        
        correlated = False  # switch on correlations for JES/b/light uncertainties
        sm_card_sr1 = makeCardFromHist(
            new_hists,
            '%s_SR_1'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node0_score_transform_pp', year, correlated=correlated),
        )
        
        sm_card_sr2 = makeCardFromHist(
            new_hists,
            '%s_SR_2'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node0_score_transform_mm', year, correlated=correlated),
        )
        
        sm_card_cr = makeCardFromHist(
            new_hists,
            '%s_CR'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node1_score', year, correlated=correlated),
        )
        
        sm_card_cr_norm = makeCardFromHist(
            new_hists,
            '%s_CR_norm'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node', year, correlated=correlated),
        )
        
        data_cards.update({
            '%s_SR1'%year: sm_card_sr1,
            '%s_SR2'%year: sm_card_sr2,
            '%s_CR'%year: sm_card_cr,
            '%s_CR_norm'%year: sm_card_cr_norm,
        })


        # load the cache
        output = get_cache('trilep_analysis_%s'%year)
        
        # check out that the histogram we want is actually there
        all_processes = [ x[0] for x in output['N_ele'].values().keys() ]
        data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        data    = data_all
        order   = ['topW_v3', 'np_est_mc', 'conv_mc', 'TTW', 'TTH', 'TTZ','rare', 'diboson']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]
        
        no_data_or_signal  = re.compile('(?!(%s))'%('|'.join(omit)))
        
        
        # FIXME I should just rebin / map everything at the same time.
        from Tools.limits import regroup_and_rebin, get_systematics
        # First, rebin & map
        for k in output.keys():
            if 'node0_score' in k:
                output[k] = regroup_and_rebin(output[k], score_bins_3l, mapping)
            elif 'node1_score' in k:
                output[k] = regroup_and_rebin(output[k], score_bins_3l, mapping)
            elif k.startswith('node') and not k.count('score'):
                output[k] = regroup_and_rebin(output[k], N_bins_3l, mapping)
        
        # then make copies for SR and CR
        new_hists = {}
        for short_name, long_name in regions_3l:
            new_hists[short_name] = output[long_name][no_data_or_signal]
                
        correlated = False  # switch on correlations for JES/b/light uncertainties
        sm_card_sr_3l = makeCardFromHist(
            new_hists,
            '%s_SR_3l'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node0_score_transform', year, correlated=correlated),
        )
        
        sm_card_cr_3l = makeCardFromHist(
            new_hists,
            '%s_CR_3l'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node1_score', year, correlated=correlated),
        )
        
        sm_card_cr_norm_3l = makeCardFromHist(
            new_hists,
            '%s_CR_norm_3l'%year,
            overflow='all',
            ext='',
            systematics = get_systematics(output, 'node', year, correlated=correlated),
        )
        
        data_cards.update({
            '%s_SR_3l'%year: sm_card_sr_3l,
            '%s_CR_3l'%year: sm_card_cr_3l,
            '%s_CR_norm_3l'%year: sm_card_cr_norm_3l,
        })
    
    combined = card.combineCards(data_cards)
    
    result_combined = card.nllScan(combined, rmin=0, rmax=3, npoints=61, options=' -v -1')
    
    if run_impacts:
        card.run_impacts(combined, plot_dir='/home/users/dspitzba/public_html/tW_scattering/')
    
    print ("Using the following data card: %s"%combined)
    print ("Significance: %.2f sigma"%np.sqrt(result_combined['deltaNLL'][1]*2))
    
    card.cleanUp()
