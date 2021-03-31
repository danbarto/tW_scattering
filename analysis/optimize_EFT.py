"""
Optimize the analysis, based on very small data frame
"""

import numpy as np
import pandas as pd

from coffea import hist

# Baysian blocks binning optimization
from hepstats.modeling import bayesian_blocks

# Optimal Binning
from optbinning import OptimalBinning

from Tools.dataCard import dataCard
from Tools.limits import makeCardFromHist
from processor.default_accumulators import dataset_axis

def get_processes(df, label='label_cat'):
    processes = {
        'topW_v2': df[((df[label]==0) & (df['label']<99))],
        'TTW':     df[((df[label]==1) & (df['label']<99))],
        'TTZ':     df[((df[label]==2) & (df['label']<99))],
        'TTH':     df[((df[label]==3) & (df['label']<99))],
        'ttbar':   df[((df[label]==4) & (df['label']<99))],
        'rare':    df[((df[label]==5) & (df['label']<99))],
    }
    return processes


if __name__ == '__main__':

    results = {}
    #variables = ['mjj_max', 'delta_eta_jj', 'met', 'ht', 'st', 
    #   'fwd_jet_p', 'fwd_jet_pt', 'lead_jet_pt',
    #   'sublead_jet_pt',
    #   'lead_btag_pt', 
    #   'sublead_btag_pt', 'lead_lep_pt', 'sublead_lep_pt', 'dilepton_mass', 'dilepton_pt', 'min_bl_dR',
    #   'min_mt_lep_met']

    variables = [\
        'st',
        'lead_lep_pt',
        'dilepton_mass',
        'score_topW',
        'score_ttW',
        'score_ttZ',
        'score_ttH',
    ]

    # Load the data
    use_SM = False
    
    df_in = pd.read_hdf('../ML/data/mini_baby_NN_v11.h5')
    
    sel = ((df_in['score_best']==0) & \
           (df_in['n_lep_tight']==2) & \
           (df_in['n_fwd']>0) & \
           (df_in['lead_lep_charge']>0))
    
    df_in = df_in[sel]
    
    tw_sm = df_in[df_in['label']==0]
    bkg   = df_in[((df_in['label']>0) & (df_in['label']<99))]
    eft   = df_in[df_in['label']>99]
    
    sig = tw_sm if use_SM else eft

    x_df = pd.concat([sig,bkg])
    
    y_sig = np.ones(len(sig))
    y_bkg = np.zeros(len(bkg))
    y = np.concatenate((y_sig,y_bkg))
    
    # Try out optimal binning
    for variable in variables:

        #variable = "lead_jet_pt"
        
        
        
        x = x_df[variable].values
        weight = x_df['weight'].values*137
        
        optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
        
        optb.fit(x, y, sample_weight=weight)
        
        print ("Fit status:")
        print (optb.status)
        print ("Binning:")
        print (optb.splits)
        
        binning_table = optb.binning_table
        binning_table.build()
        
        # Baysian binning opt
        #blocks = bayesian_blocks(sig['st'], weights=sig['weight']*137, p0=0.5) # very slow, gives a low number of bins... let's check with bkg as well

        # Make some comparisons
        
        energy_axis     = hist.Bin("e",     r"E", 5, 0, 1000)  # if 9 was already optimal I'll call myself eyeball champ.
        opt_axis        = hist.Bin("e",     r"E", [0]+list(optb.splits) if optb.splits[0]>0 else list(optb.splits))

        processes = get_processes(df_in, label='label_cat')

        h_sm_default = hist.Hist("e", dataset_axis, energy_axis)
        h_bsm_default = hist.Hist("e", dataset_axis, energy_axis)

        h_sm_opt = hist.Hist("e", dataset_axis, opt_axis)
        h_bsm_opt = hist.Hist("e", dataset_axis, opt_axis)

        for proc in processes:
            h_sm_default.fill(dataset=proc, e=processes[proc][variable].values, weight=processes[proc]["weight"].values*137)
            h_sm_opt.fill(dataset=proc, e=processes[proc][variable].values, weight=processes[proc]["weight"].values*137)

        h_bsm_default.fill(dataset='EFT', e=sig[variable].values, weight=sig["weight"].values*137)
        h_bsm_opt.fill(dataset='EFT', e=sig[variable].values, weight=sig["weight"].values*137)

        output = {
            'sm_default': h_sm_default,
            'sm_opt': h_sm_opt,
        }

        try:
            #sm_card_default = makeCardFromHist(output, 'sm_default', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True, categories=True)
            sm_card_opt = makeCardFromHist(output, 'sm_opt', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True, categories=True)

            #bsm_card_default = makeCardFromHist(output, 'sm_default', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='_bsm', systematics=True, categories=True, bsm_hist=h_bsm_default)
            bsm_card_opt = makeCardFromHist(output, 'sm_opt', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='_bsm', systematics=True, categories=True, bsm_hist=h_bsm_opt)

            card = dataCard(releaseLocation='/home/users/dspitzba/TTW/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/')

            #res_sm_default = card.calcNLL(sm_card_default)
            res_sm_opt = card.calcNLL(sm_card_opt)

            #res_bsm_default = card.calcNLL(bsm_card_default)
            res_bsm_opt = card.calcNLL(bsm_card_opt)

            #print ("Default:", 2*(res_sm_default['nll0'][0]+res_sm_default['nll'][0]- (res_bsm_default['nll0'][0]+res_bsm_default['nll'][0])))
            print ("Optimal:", 2*(res_sm_opt['nll0'][0]+res_sm_opt['nll'][0]- (res_bsm_opt['nll0'][0]+res_bsm_opt['nll'][0])))

            card.cleanUp()

            results.update({variable: 2*(res_sm_opt['nll0'][0]+res_sm_opt['nll'][0]- (res_bsm_opt['nll0'][0]+res_bsm_opt['nll'][0]))})

        except:
            results.update({variable: 0})
    
    print (results)
