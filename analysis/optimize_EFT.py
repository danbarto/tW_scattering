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
    variables = []

    #variables += ['mjj_max', 'delta_eta_jj', 'met', 'ht', 'st', 
    #   'fwd_jet_p', 'fwd_jet_pt', 'lead_jet_pt',
    #   'sublead_jet_pt',
    #   'lead_btag_pt', 
    #   'sublead_btag_pt', 'lead_lep_pt', 'sublead_lep_pt', 'dilepton_mass', 'dilepton_pt', 'min_bl_dR',
    #   'min_mt_lep_met',
    #]

    variables += [\
        #'st',
        #'lead_lep_pt',
        #'dilepton_mass',
        'score_topW',
        #'score_ttW',
        #'score_ttZ',
        #'score_ttH',
    ]

    # Load the data
    use_SM = True
    
    df_in = pd.read_hdf('../ML/data/mini_baby_NN_v11.h5')
    
    baseline = ((df_in['n_lep_tight']==2) \
            #& (df_in['score_best']==0) \
            #& (df_in['label_cat']<3) \
            #& (df_in['n_lep_tight']==2) \
            & (df_in['n_fwd']>0) \
            #& (df_in['lead_lep_charge']>0) \
             )
    
    df_in = df_in[baseline]
    
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
        print ("\n\n ### Next Variable ### \n")
        print (" ..:: %s ::.. \n"%variable)
        
        #x_df = x_df[((x_df['score_best']==0) & (x_df['lead_lep_charge']>0))]
        
        x = x_df[variable].values
        weight = x_df['weight'].values*137
        
        #optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
        optb = OptimalBinning(name=variable,
                              dtype="numerical",
                              solver="cp",
                              #prebinning_method='uniform',
                              #max_n_prebins=100,
                              min_n_bins=4,
                             )
        
        optb.fit(x, y, sample_weight=10*weight)
        #optb.fit(x, y, sample_weight=weight)
        
        print ("Fit status:")
        print (optb.status)
        print ("Binning:")
        print (optb.splits)
        
        binning_table = optb.binning_table
        print (binning_table.build())
        
        # Baysian binning opt
        #blocks = bayesian_blocks(sig['st'], weights=sig['weight']*137, p0=0.5) # very slow, gives a low number of bins... let's check with bkg as well

        # Make some comparisons
        
        energy_axis     = hist.Bin("e",     r"E", 8, 0.20, 0.6)  # if 9 was already optimal I'll call myself eyeball champ.
        opt_axis        = hist.Bin("e",     r"E", [0]+list(optb.splits) if optb.splits[0]>0 else list(optb.splits))

        #regions = {'pos': lambda x: x['lead_lep_charge']>0, 'neg': lambda x: x['lead_lep_charge']<0}  # pos/neg charge split
        regions = {'pos_sig': lambda x: ((x['lead_lep_charge']>0)&(x['score_best']==0)),
                   'neg_sig': lambda x: ((x['lead_lep_charge']<0)&(x['score_best']==0)),
                   'pos_ttw': lambda x: ((x['lead_lep_charge']>0)&(x['score_best']==1)),
                   'neg_ttw': lambda x: ((x['lead_lep_charge']<0)&(x['score_best']==1)),
                   }
        #regions = {'pos': lambda x: x['lead_lep_charge']>0}  # pos/neg charge split
        sm_cards = {}
        sm_cards_def = {}
        bsm_cards = {}

        for region in regions:

            processes = get_processes(df_in[regions[region](df_in)], label='label_cat')
            #processes = get_processes(df_in, label='label_cat')

            h_sm_default = hist.Hist("e", dataset_axis, energy_axis)
            h_bsm_default = hist.Hist("e", dataset_axis, energy_axis)

            h_sm_opt = hist.Hist("e", dataset_axis, opt_axis)
            h_bsm_opt = hist.Hist("e", dataset_axis, opt_axis)

            for proc in processes:
                h_sm_default.fill(dataset=proc, e=processes[proc][variable].values, weight=processes[proc]["weight"].values*137)
                h_sm_opt.fill(dataset=proc, e=processes[proc][variable].values, weight=processes[proc]["weight"].values*137)

            sig_sel = regions[region](sig)
            h_bsm_default.fill(dataset='EFT', e=sig[sig_sel][variable].values, weight=sig[sig_sel]["weight"].values*137)
            h_bsm_opt.fill(dataset='EFT', e=sig[sig_sel][variable].values, weight=sig[sig_sel]["weight"].values*137)

            output = {
                'sm_default': h_sm_default,
                'sm_opt': h_sm_opt,
            }


            card = dataCard(releaseLocation='/home/users/dspitzba/TTW/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/')

            sm_card_default = makeCardFromHist(output, 'sm_default', overflow='all', ext='_'+region, systematics=True, categories=True)
            sm_card_opt = makeCardFromHist(output, 'sm_opt', overflow='all', ext='_'+region, systematics=True, categories=True)
            sm_cards[region] = sm_card_opt
            sm_cards_def[region] = sm_card_default

            if not use_SM:
                bsm_card_opt = makeCardFromHist(output, 'sm_opt', overflow='all', ext='_bsm_'+region, systematics=True, categories=True, bsm_hist=h_bsm_opt)
                bsm_cards[region] = bsm_card_opt


        if use_SM:
            deltaNLL_def = 0
            sm_card_opt = card.combineCards(sm_cards)
            res_sm = card.nllScan(sm_card_opt, rmin=0, rmax=3, npoints=61, options=' -v -1')
            sm_card_default = card.combineCards(sm_cards_def)
            #res_def = card.nllScan(sm_card_default, rmin=0, rmax=3, npoints=61, options=' -v -1')
            #deltaNLL_def = (res_def[res_def['r']==0]['deltaNLL']*2)[0]
            deltaNLL = (res_sm[res_sm['r']==0]['deltaNLL']*2)[0]

            print ("Default:", deltaNLL_def)
            print ("Optimum:", deltaNLL)
            
            deltaNLL = (deltaNLL_def, deltaNLL)

        else:
            #res_sm_default = card.calcNLL(sm_card_default)
            sm_card_opt = card.combineCards(sm_cards)
            res_sm_opt = card.calcNLL(sm_card_opt)

            #bsm_card_default = makeCardFromHist(output, 'sm_default', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='_bsm', systematics=True, categories=True, bsm_hist=h_bsm_default)
            #res_bsm_default = card.calcNLL(bsm_card_default)
            bsm_card_opt = card.combineCards(bsm_cards)
            res_bsm_opt = card.calcNLL(bsm_card_opt)
            deltaNLL = -2*(res_sm_opt['nll0'][0]+res_sm_opt['nll'][0]- (res_bsm_opt['nll0'][0]+res_bsm_opt['nll'][0]))

        #print ("Default:", 2*(res_sm_default['nll0'][0]+res_sm_default['nll'][0]- (res_bsm_default['nll0'][0]+res_bsm_default['nll'][0])))

        print ("Final SM data card is:", sm_card_opt)
        print ("Optimal:", deltaNLL)        

        card.cleanUp()

        results.update({variable: deltaNLL})

        # plotting
        if variable.count('score') and True:
            from plots.helpers import makePlot
            plot_dir = "/home/users/dspitzba/public_html/tW_scattering/SR_optimization/"

            my_labels = {
                'topW_v2': 'top-W scat.',
                'TTW': 'prompt',
                'TTZ': 'lost lepton',
                'TTH': 'nonprompt',
                'ttbar': 'charge flip',
                'rare': 'rare',
            }

            my_colors = {
                'topW_v2': '#FF595E',
                'TTZ': '#FFCA3A',
                'TTW': '#8AC926',
                'TTH': '#34623F',
                'rare': '#525B76',
                'ttbar': '#1982C4',
            }

            makePlot(output, 'sm_opt', 'e',
                log=False, normalize=False, axis_label=r'$top-W\ score$',
                new_colors=my_colors, new_labels=my_labels,
                save=plot_dir+'/'+variable,
                order=['rare', 'TTH', 'ttbar', 'TTZ', 'TTW'],
                signals=['topW_v2'],
                lumi=137,
                binwnorm=True,
                ymax=30,
                )

    
    print (results)
