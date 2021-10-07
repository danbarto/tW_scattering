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
    argParser.add_argument('--years', action='store', default='2016', help="Which years to include?")
    args = argParser.parse_args()

    years = args.years.split(',')
    data_cards = {}

    #card = dataCard(releaseLocation=os.path.expandvars('/home/users/$USER/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))  # FIXME for whatever reason this version still has the issue with the absolute NLL value...
    card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TTW/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))
    pt_bins     = hist.Bin('pt', r'$p_{T}\ (GeV)$', [0,100,150,200,400])
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

    for year in years:
        # load the cache
        output = get_cache('SS_analysis_%s'%year)

        # FIXME this needs to become flexible
        output_EFT = get_cache('EFT_ctW_scan_%s'%year)
        eft_mapping = {k[0]:k[0] for k in output_EFT['lead_lep_SR_pp'].values().keys() if 'topW_full_EFT' in k[0]}  # not strictly necessary
        
        # check out that the histogram we want is actually there
        all_processes = [ x[0] for x in output['N_ele'].values().keys() ]
        data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        data    = data_all
        order   = ['topW_v3', 'np_est_mc', 'conv_mc', 'cf_est_mc', 'TTW', 'TTH', 'TTZ','rare', 'diboson']
        signals = []
        omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]
        
        no_data_or_signal  = re.compile('(?!(%s))'%('|'.join(omit)))
        
        regions = [
            ('%s_SR_1'%year, 'lead_lep_SR_pp'),
            ('%s_SR_2'%year, 'lead_lep_SR_mm'),
            ('%s_CR'%year, 'node1_score'),
            ('%s_CR_norm'%year, 'node'),
        ]

        from Tools.limits import regroup_and_rebin, get_systematics
        # First, rebin & map
        for k in output.keys():
            if 'lead_lep_SR' in k:
                output[k] = regroup_and_rebin(output[k], pt_bins, mapping)
            elif 'node1_score' in k:
                output[k] = regroup_and_rebin(output[k], score_bins, mapping)
            elif k.startswith('node') and not k.count('score'):
                output[k] = regroup_and_rebin(output[k], N_bins, mapping)
        
        
        for k in output_EFT.keys():
            if 'lead_lep_SR' in k:
                output_EFT[k] = regroup_and_rebin(output_EFT[k], pt_bins, eft_mapping)
            elif 'node1_score' in k:
                output_EFT[k] = regroup_and_rebin(output_EFT[k], score_bins, eft_mapping)
            elif k.startswith('node') and not k.count('score'):
                output_EFT[k] = regroup_and_rebin(output_EFT[k], N_bins, eft_mapping)

        # then make copies for SR and CR
        new_hists = {}
        signal_hists = {}
        for short_name, long_name in regions:
            new_hists[short_name] = output[long_name][no_data_or_signal]
            signal_hists[short_name] = output_EFT[long_name][no_data_or_signal]

        correlated = False  # switch on correlations for JES/b/light uncertainties

        '''
        The BSM interpretation requires a comparison of the BSM (EFT) hypothesis with the SM hypothesis
        We therefore calculate the likelihood of the SM (that's what calcNLL does for a certain data card) as a reference,
        and then for the likelihood for each EFT point.
        The combine tool reports an absolute value (nll0) and the improvement of the likelihood after the fit (nll).
        The total LL is therefore nll0+nll, which is calculated for the SM point and each EFT point.
        We need to find a clever way of looping through the years and the EFT points.
        '''
        
        # First we need to make the SM card. This is just for a single set of SRs
        sm_card_sr1 = makeCardFromHist(
            new_hists,
            '%s_SR_1'%year,
            overflow='all',
            ext='',
            signal_hist=signal_hists['%s_SR_1'%year]['topW_full_EFT_ctW_0.0'],
            #systematics = get_systematics(output, 'node0_score_transform_pp', year, correlated=correlated),  # FIXME we need systematics for the lepton pt distributions, and for signal.
        )
        
        res_sm = card.calcNLL(sm_card_sr1)

        # BSM card for a single set of SRs. # FIXME year and EFT point should not be hard coded
        bsm_card_sr1 = makeCardFromHist(
            new_hists,
            '%s_SR_1'%year,
            overflow='all',
            ext='_BSM',
            signal_hist=signal_hists['%s_SR_1'%year]['topW_full_EFT_ctW_2.5'],
            #systematics = get_systematics(output, 'node0_score_transform_pp', year, correlated=correlated), 
        )
        
        res_sm = card.calcNLL(sm_card_sr1)
        res_bsm = card.calcNLL(bsm_card_sr1)

        deltaNLL = -2*(res_sm['nll0'][0]+res_sm['nll'][0]- (res_bsm['nll0'][0]+res_bsm['nll'][0]))

        print ("DeltaNLL: %.2f"%deltaNLL)

        
    #    data_cards.update({
    #        '%s_SR1'%year: sm_card_sr1,
    #        '%s_SR2'%year: sm_card_sr2,
    #        '%s_CR'%year: sm_card_cr,
    #        '%s_CR_norm'%year: sm_card_cr_norm,
    #    })


    #combined = card.combineCards(data_cards)
    #
    #result_combined = card.nllScan(combined, rmin=0, rmax=3, npoints=61, options=' -v -1')
    #
    
    card.cleanUp()
