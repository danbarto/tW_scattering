#!/usr/bin/env python3

import os
import warnings
warnings.filterwarnings('ignore')

# data handling and numerical analysis
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import boost_histogram as bh

import scipy
import pickle

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

from Tools.config_helpers import get_samples
from Tools.helpers import make_bh, finalizePlotDir
from Tools.HyperPoly import HyperPoly
from Tools.reweighting import get_coordinates_and_ref, get_coordinates
from plots.helpers import colors

from BoostedInformationTreeP3 import BoostedInformationTree

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils import resample, shuffle

from ML.multiclassifier_tools import store_transformer

from yahist import Hist1D

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def histo_values(histo, weight):
    return histo.integrate('eft', weight).sum('dataset').values()[()]

def get_bit_score(df, cpt=0, cpqm=0, trans=None):
    tmp = (
        df['pred_0'].values + \
        df['pred_1'].values*cpt + \
        df['pred_2'].values*cpqm + \
        0.5*df['pred_3'].values*cpt**2 + \
        0.5*df['pred_4'].values*cpt*cpqm + \
        0.5*df['pred_5'].values*cpqm**2 + \
        0
    )
    if trans:
        return trans.transform(tmp.reshape(-1,1)).flatten()
    else:
        return tmp

def get_bit_score_simple(df, cpt=0, cpqm=0, trans=None):
    tmp = (
        #df['pred_0'].values + \
        df['pred_1'].values + \
        #df['pred_2'].values + \
        0.5*df['pred_3'].values*cpt + \
        #df['pred_4'].values*cpt + \
        #df['pred_4'].values*cpqm + \
        #df['pred_5'].values*cpqm + \
        0
    )
    if trans:
        return trans.transform(tmp.reshape(-1,1)).flatten()
    else:
        return tmp

def make_cdf_map( x, y ):
    import scipy.interpolate
    map__ = scipy.interpolate.interp1d(x, y, kind='linear')
    max_x, min_x = max(x), min(x)
    max_y, min_y = max(y), min(y)
    def map_( x_ ):
        x__ = np.array(x_)
        result = np.zeros_like(x__).astype('float')
        result[x__>max_x] = max_y
        result[x__<min_x] = min_y
        vals = (x__>=min_x) & (x__<=max_x)
        result[vals] = map__(x__[vals])
        return result

    return map_

variables = [
        'n_jet',
        'n_fwd',
        'n_b',
        'n_tau',
        'st',
        'lt',
        'met',
        'mjj_max',
        'delta_eta_jj',
        'lead_lep_pt',
        'lead_lep_eta',
        'sublead_lep_pt',
        'sublead_lep_eta',
        'dilepton_mass',
        'dilepton_pt',
        'fwd_jet_pt',
        'fwd_jet_p',
        'fwd_jet_eta',
        'lead_jet_pt',
        'sublead_jet_pt',
        'lead_jet_eta',
        'sublead_jet_eta',
        'lead_btag_pt',
        'sublead_btag_pt',
        'lead_btag_eta',
        'sublead_btag_eta',
        'min_bl_dR',
        'min_mt_lep_met',
    ]

x_labels = {
        'n_jet': r'$N_{jet}$',
        'n_fwd': r'$N_{fwd jet}$',
        'n_b': r'$N_{b}$',
        'n_tau': r'$N_{\tau}$',
        'st': r'$S_{T}\ (GeV)$',
        'lt': r'$L_{T}\ (GeV)$',
        'met': r'$p_{T}^{miss}\ (GeV)$',
        'mjj_max': r'max$M_{jj}\ (GeV)$',
        'delta_eta_jj': r'$\delta\eta(jj)$',
        'lead_lep_pt': r'$p_{T} (lead\ lep)\ (GeV)$',
        'lead_lep_eta': r'$\eta (lead\ lep)$',
        'sublead_lep_pt': r'$p_{T} (sublead\ lep)\ (GeV)$',
        'sublead_lep_eta': r'$\eta (sublead\ lep)$',
        'dilepton_mass': r'$M_{ll}\ (GeV)$',
        'dilepton_pt': r'$p_{T}(ll)\ (GeV)$',
        'fwd_jet_pt': r'$p_{T}(fwd\ jet)\ (GeV)$',
        'fwd_jet_p': r'$p(fwd\ jet)\ (GeV)$',
        'fwd_jet_eta': r'$\eta (fwd\ jet)$',
        'lead_jet_pt': r'$p_{T} (lead\ jet)\ (GeV)$',
        'sublead_jet_pt': r'$p_{T} (sublead\ jet)\ (GeV)$',
        'lead_jet_eta': r'$\eta (lead\ jet)$',
        'sublead_jet_eta': r'$\eta (sublead\ jet)$',
        'lead_btag_pt': r'$p_{T} (lead\ b-tagged\ jet)\ (GeV)$',
        'sublead_btag_pt': r'$p_{T} (sublead\ b-tagged\ jet)\ (GeV)$',
        'lead_btag_eta': r'$\eta (lead\ b-tagged\ jet)$',
        'sublead_btag_eta': r'$\eta (sublead\ b-tagged\ jet)$',
        'min_bl_dR': r'min$M(l,\ b-tagged\ jet)\ (GeV)$',
        'min_mt_lep_met': r'min$M_{T}(l, p_{T}^{miss})\ (GeV)$',
}

inputs_binning = {
        'n_jet': "9,3.5,12.5",
        'n_fwd': "4,0.5,4.5",
        'n_b': "5,0.5,5.5",
        'n_tau': "3,-0.5,2.5",
        'st': "20,200,1200",
        'lt': "20,200,1200",
        'met': "20,200,1200",
        'mjj_max': "20,200,1200",
        'delta_eta_jj': "10,0,5",
        'lead_lep_pt': "20,200,1200",
        'lead_lep_eta': "20,200,1200",
        'sublead_lep_pt': "20,200,1200",
        'sublead_lep_eta': "20,200,1200",
        'dilepton_mass': "20,200,1200",
        'dilepton_pt': "20,200,1200",
        'fwd_jet_pt': "20,200,1200",
        'fwd_jet_p': "20,200,1200",
        'fwd_jet_eta': "20,200,1200",
        'lead_jet_pt': "20,200,1200",
        'sublead_jet_pt': "20,200,1200",
        'lead_jet_eta': "20,200,1200",
        'sublead_jet_eta': "20,200,1200",
        'lead_btag_pt': "20,200,1200",
        'sublead_btag_pt': "20,200,1200",
        'lead_btag_eta': "20,200,1200",
        'sublead_btag_eta': "10,-5,5",
        'min_bl_dR': "10,0,10",
        'min_mt_lep_met': "20,0,250",
}


variables = sorted(variables)

if __name__ == '__main__':

    from Tools.config_helpers import loadConfig
    cfg = loadConfig()

    #print (variables)
    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--retrain', action='store_true', default=None, help="Retrain the BDT?")
    argParser.add_argument('--fit', action='store_true', default=None, help="Run combine fit?")
    argParser.add_argument('--plot', action='store_true', default=None, help="Make all the plots")
    argParser.add_argument('--signalOnly', action='store_true', default=None, help="Use only signal for training / evaluation")
    argParser.add_argument('--scan', action='store_true', default=None, help="Run the entire 2D scan (slow)")
    argParser.add_argument('--allBkg', action='store_true', default=None, help="Use backgrounds (not just ttW)")
    argParser.add_argument('--max_label', action='store', type=int, default=100, help="Maximum label for backgrounds in training (strictly smaller!)")  # labels go from 1-7 (see below for definitions)
    argParser.add_argument('--use_weight', action='store_true', help="Use weights in training")
    argParser.add_argument('--runLT', action='store_true', default=None, help="Run classical LT analysis (does not change with different training versions)")
    argParser.add_argument('--version', action='store', default='v1', help="Version number for the output tree")
    argParser.add_argument('--year', action='store', default='2018', help="Version number for the output tree")

    # NOTE: need to add the full Run2 training option back
    args = argParser.parse_args()

    ## NOTE: Labels
    # labels = {'topW': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'top': 4, 'rare':5, 'diboson':6, 'XG': 7, 'topW_lep': 0}

    bit_file = f'bits_{args.version}.pkl'  # list of all trees
    signal_bit_file = f'bit_{args.version}.pkl'  # Signal only tree

    # load data
    #sample_list =  ['TTW', 'TTZ','TTH']
    sample_list =  ['TTW', 'TTZ','TTH', 'top', 'rare', 'diboson', 'XG']
    years = [args.year] if not args.year == 'all' else ['2016APV', '2016', '2017', '2018']

    # simplified systematics
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

    data_dir = '/ceph/cms/store/user/dspitzba/tW_scattering/MVA/'

    df_in = pd.DataFrame()
    for year in years:
        tmp = pd.concat([pd.read_hdf(f"{data_dir}/multiclass_input_{sample}_{year}.h5") for sample in sample_list])
        df_in = pd.concat([df_in, tmp])
        df_in = df_in[((df_in['SS']==1) | (df_in['AR']==1))]
    del tmp

    df_signal = pd.concat([pd.read_hdf(f"{data_dir}/multiclass_input_topW_lep_{year}.h5") for year in years])
    df_signal['weight'] = df_signal['weight']  # NOTE this needs a factor 2 if mixed with inclusive signal!

    plot_dir = os.path.expandvars(f"{cfg['meta']['plots']}/BIT/{args.version}/")
    finalizePlotDir(plot_dir)

    # Prepare hyper poly inputs
    samples = get_samples("samples_UL18.yaml")
    f_in    = samples['/ceph/cms/store/user/dspitzba/ProjectMetis/TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO_RunIISummer20UL18_NanoAODv9_NANO_v14/']['files'][0]
    tree    = uproot.open(f_in)['Events']

    hp = HyperPoly(2)
    coordinates, ref_coordinates = get_coordinates_and_ref(f_in)
    ref_coordinates = [0.0,0.0]

    hp.initialize( coordinates, ref_coordinates )
    weights = [ x.replace('LHEWeight_','') for x in tree.keys() if x.startswith('LHEWeight_c') ]

    coeff = hp.get_parametrization( [df_signal[w].values for w in weights] )


    # Preprocessing inputs
    df_signal['lt'] = (df_signal['lead_lep_pt'].values + df_signal['sublead_lep_pt'].values + df_signal['met'].values)
    df_in['lt'] = (df_in['lead_lep_pt'].values + df_in['sublead_lep_pt'].values + df_in['met'].values)

    # this roughly corresponds to labeling
    for i in range(6):
        df_signal['coeff_%s'%i] = coeff[i,:]
        df_in['coeff_%s'%i] = np.zeros(len(df_in))

    df_signal = df_signal[((df_signal['n_fwd']>=1))]
    df_np  = df_in[((df_in['AR']==1) & (df_in['label']==4) &(df_in['n_fwd']>=1))]
    df_bkg = df_in[((df_in['SS']==1)&(df_in['n_fwd']>=1))]

    for var in variables:
        for df in [df_signal, df_np, df_bkg]:
            df[var] = abs(df[var])

    print ("Sample sizes:")
    print ("Bkg: {}".format(len(df_bkg)))
    print ("Signal: {}".format(len(df_signal)))

    # signal
    sig_train_sel   = ((df_signal['OS']==1) & (df_signal['nLepFromTop']==1)) #((df_signal['event']%2)==1).values
    sig_test_sel    = (df_signal['SS']==1)#~sig_train_sel
    sig_train       = df_signal[sig_train_sel]
    sig_test        = df_signal[sig_test_sel]

    # bkg
    bkg_train_sel   = (df_bkg['event']%2)==1
    bkg_test_sel    = ~bkg_train_sel
    if args.signalOnly:
        bkg_train       = df_bkg[((bkg_train_sel) & (df_bkg['label']==10))]
        bkg_test        = df_bkg[((bkg_test_sel) & (df_bkg['label']==10))]
    else:
        if args.allBkg:
            bkg_train       = df_bkg[((bkg_train_sel) & (df_bkg['label']<args.max_label))]
            bkg_test        = df_bkg[((bkg_test_sel) & (df_bkg['label']<100))]
        else:
            bkg_train       = df_bkg[((bkg_train_sel) & (df_bkg['label']==1))]
            bkg_test        = df_bkg[((bkg_test_sel) & (df_bkg['label']==1))]

    np_test = df_np

    train = pd.concat([sig_train, bkg_train])
    print (len(train))

    ## Plotting the input coefficients ##
    #coeff_bins = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    coeff_bins = "20,-1,12"
    for i in range(6):

        h_signal = Hist1D(sig_train[f'coeff_{i}'].values, bins=coeff_bins)

        fig, ax = plt.subplots(figsize=(10,10))

        hep.histplot(
            [ h_signal.counts ],
            h_signal.edges,
            #w2 = [ h_train.errors**2, h_rew_train.errors**2, h_rew2_train.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            #linestyle="--",
            label= ["Signal"],
            color = ["#FF595E"],
            ax=ax)

        ax.set_yscale('log')
        ax.legend()

        fig.savefig(f"{plot_dir}/diff_weight_{i}.png")


    ## Train the tree for signal only ##
    n_trees       = 50
    learning_rate = 0.3
    max_depth     = 2
    min_size      = 20
    calibrated    = False

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(sig_train[variables])
    params = scaler.get_params()

    if os.path.isfile(signal_bit_file) and not args.retrain:
        with open(signal_bit_file, 'rb') as f:
            bit = pickle.load(f)

    else:
        bit = BoostedInformationTree(
            training_features     = train_scaled,
            training_weights      = abs(sig_train['weight'].values),
            training_diff_weights = sig_train['coeff_1'].values,# coeff[3,:][sig_train_sel],
            learning_rate         = learning_rate,
            n_trees               = n_trees,
            max_depth             = max_depth,
            min_size              = min_size,
            calibrated            = True)

        bit.boost()

        with open(signal_bit_file, 'wb') as f:
            pickle.dump(bit, f)

    bins = "10,0.0,1.0"

    sig_test_scaled = scaler.transform(sig_test[variables].values)
    sig_train_scaled = scaler.transform(sig_train[variables].values)

    coeffs = np.concatenate([np.expand_dims(sig_test['coeff_%s'%i], axis=1) for i in range(6)], axis=1)
    pred = bit.vectorized_predict(sig_test_scaled)

    h = Hist1D(pred, bins=bins)
    h_rew = Hist1D(pred, weights=hp.eval(coeffs.transpose(), [4.,0.]), bins=bins)
    h_rew2 = Hist1D(pred, weights=hp.eval(coeffs.transpose(), [8.,0.]), bins=bins)

    coeffs_train = np.concatenate([np.expand_dims(sig_train['coeff_%s'%i], axis=1) for i in range(6)], axis=1)
    pred_train = bit.vectorized_predict(sig_train_scaled)

    h_train = Hist1D(pred_train, bins=bins)
    h_rew_train = Hist1D(pred_train, weights=hp.eval(coeffs_train.transpose(), [4.,0.]), bins=bins)
    h_rew2_train = Hist1D(pred_train, weights=hp.eval(coeffs_train.transpose(), [8.,0.]), bins=bins)

    if args.plot or True:
        fig, ax = plt.subplots(figsize=(10,10))

        # solid - testing
        hep.histplot(
            [ h.counts, h_rew.counts, h_rew2.counts ],
            h.edges,
            #w2 = [ h.errors**2, h_rew.errors**2, h_rew2.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            label = ["SM test", "C=4 test", "C=8 test"],
            color = ["#FF595E", "#8AC926", "#1982C4"],
            ax=ax)

        # dashed - training
        hep.histplot(
            [ h_train.counts, h_rew_train.counts, h_rew2_train.counts ],
            h.edges,
            #w2 = [ h_train.errors**2, h_rew_train.errors**2, h_rew2_train.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            linestyle="--",
            color = ["#FF595E", "#8AC926", "#1982C4"],
            ax=ax)

        ax.set_yscale('log')
        ax.legend()

        fig.savefig(f"{plot_dir}/1D_signal_only.png")

    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=40, random_state=0)
    qt.fit(pred.reshape(-1, 1))

    pred_trans = qt.transform(pred.reshape(-1,1)).flatten()
    pred_train_trans = qt.transform(pred_train.reshape(-1,1)).flatten()

    h = Hist1D(pred_trans, bins=bins)
    h_rew = Hist1D(pred_trans, weights=hp.eval(coeffs.transpose(), [4.,0.]), bins=bins)
    h_rew2 = Hist1D(pred_trans, weights=hp.eval(coeffs.transpose(), [8.,0.]), bins=bins)

    h_train = Hist1D(pred_train_trans, bins=bins)
    h_rew_train = Hist1D(pred_train_trans, weights=hp.eval(coeffs_train.transpose(), [4.,0.]), bins=bins)
    h_rew2_train = Hist1D(pred_train_trans, weights=hp.eval(coeffs_train.transpose(), [8.,0.]), bins=bins)

    if args.plot:
        fig, ax = plt.subplots(figsize=(10,10))

        # solid - testing
        hep.histplot(
            [ h.counts, h_rew.counts, h_rew2.counts ],
            h.edges,
            #w2 = [ h.errors**2, h_rew.errors**2, h_rew2.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            label = ["SM test", "C=4 test", "C=8 test"],
            color = ["#FF595E", "#8AC926", "#1982C4"],
            ax=ax)

        # dashed - training
        hep.histplot(
            [ h_train.counts, h_rew_train.counts, h_rew2_train.counts ],
            h.edges,
            #w2 = [ h_train.errors**2, h_rew_train.errors**2, h_rew2_train.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            linestyle="--",
            color = ["#FF595E", "#8AC926", "#1982C4"],
            ax=ax)

        ax.set_yscale('log')
        ax.legend()

        fig.savefig(f"{plot_dir}/1D_signal_only_transformed.png")



    ## Make plots of all the input variables ##
    eft_weight = hp.eval(coeffs_train.transpose(), [6,0])
    for var in variables:

        h_signal = Hist1D(sig_train[var].values, bins=inputs_binning[var], overflow=True)
        h_bkg = Hist1D(bkg_train[var].values, bins=h_signal.edges, overflow=True)
        h_bsm = Hist1D(sig_train[var].values, bins=h_signal.edges, weights=eft_weight, overflow=True)

        fig, ax = plt.subplots(figsize=(10,10))

        hep.cms.label(
            "Preliminary",
            data=False,
            #year=2018,
            lumi=1,
            loc=0,
            ax=ax,
           )

        hep.histplot(
            [ h_signal.counts, h_bkg.counts, h_bsm.counts ],
            h_signal.edges,
            #w2 = [ h_train.errors**2, h_rew_train.errors**2, h_rew2_train.errors**2 ],
            histtype="step",
            stack=False,
            density=True,
            linewidth=3,
            #linestyle="--",
            label= [\
                r"top-W scat. ($C_{\varphi t}=0, C_{\varphi Q}^{-}=0$)",
                "All Bkg",
                r"top-W scat. ($C_{\varphi t}=6, C_{\varphi Q}^{-}=0$)",
            ],
            color = ["#FF595E", "#8AC926", "#1982C4"],
            ax=ax,
        )

        ax.set_yscale('log')
        ax.legend()

        ax.set_xlabel(x_labels[var])
        ax.set_ylabel(r'a.u.')

        fig.savefig(f"{plot_dir}/input_{var}.png")



    ## Train all the trees! ##

    # NOTE default
    n_trees       = 100  # from 100
    learning_rate = 0.3
    max_depth     = 4  # v21: 3, v22: 5, v23: 7, v24: signal only
    min_size      = 25  # v18: 20, v20: 5, v21: 1, v25: 25

    ## NOTE what I determined to be the best hyperparameters
    #n_trees       = 30  # from 100
    #learning_rate = 0.1
    #max_depth     = 5  # v21: 3, v22: 5, v23: 7, v24: signal only
    #min_size      = 25  # v18: 20, v20: 5, v21: 1, v25: 25

    ## NOTE the actual best hyper parameters
    #n_trees       = 10  # from 100
    #learning_rate = 0.1
    #max_depth     = 5  # v21: 3, v22: 5, v23: 7, v24: signal only
    #min_size      = 25  # v18: 20, v20: 5, v21: 1, v25: 25


    training_features = train[variables].values
    if args.use_weight:
        training_weights = abs(train['weight'].values)
    else:
        training_weights = np.ones_like(train['weight'].values)
    scaler = RobustScaler()
    #training_features_scaled = scaler.fit_transform(training_features)
    training_features_scaled = training_features
    params = scaler.get_params()

    if os.path.isfile(bit_file) and not args.retrain:
        with open(bit_file, 'rb') as f:
            bits = pickle.load(f)

    else:
        bits = []

        for i in range(6):
            training_diff_weights = train['coeff_%s'%i].values
            bits.append(
                BoostedInformationTree(
                    training_features     = training_features_scaled,
                    training_weights      = training_weights,
                    training_diff_weights = training_diff_weights,
                    learning_rate         = learning_rate,
                    n_trees               = n_trees,
                    max_depth             = max_depth,
                    min_size              = min_size,
                    calibrated            = False)
            )

            bits[-1].boost()

        with open(bit_file, 'wb') as f:
            pickle.dump(bits, f)


    # Evaluate all the BITS and store results
    #scaled_test = scaler.transform(sig_test[variables].values)
    scaled_test = sig_test[variables].values
    if len(bkg_test[variables].values)==0:
        scaled_bkg_test = bkg_test[variables].values
    else:
        #scaled_bkg_test = scaler.transform(bkg_test[variables].values)
        scaled_bkg_test = bkg_test[variables].values

    if len(np_test[variables].values)==0:
        scaled_np_test = np_test[variables].values
    else:
        #scaled_np_test = scaler.transform(np_test[variables].values)
        scaled_np_test = np_test[variables].values

    for i in range(6):
        sig_test['pred_%s'%i] = bits[i].vectorized_predict(scaled_test)
        #sig_train['pred_%s'%i] = bits[i].vectorized_predict(scaler.transform(sig_train[variables].values))
        sig_train['pred_%s'%i] = bits[i].vectorized_predict(sig_train[variables].values)
        bkg_test['pred_%s'%i] = bits[i].vectorized_predict(scaled_bkg_test)
        np_test['pred_%s'%i] = bits[i].vectorized_predict(scaled_np_test)


    # Labels are defined assert
    # labels = {'topW': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'top': 4, 'rare':5, 'diboson':6, 'XG': 7, 'topW_lep': 0}
    ttW = bkg_test[bkg_test['label']==1]
    ttZ = bkg_test[bkg_test['label']==2]
    ttH = bkg_test[bkg_test['label']==3]
    NP = np_test #[bkg_test['label']==4]
    rare = bkg_test[bkg_test['label']==5]
    diboson = bkg_test[bkg_test['label']==6]
    XG = bkg_test[bkg_test['label']==7]


    # Run the analysis

    from coffea import hist

    from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
    from Tools.dataCard import dataCard
    card = dataCard(releaseLocation=os.path.expandvars('/home/users/$USER/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

    dataset_axis = hist.Cat("dataset", "Primary dataset")
    eft_axis = hist.Cat("eft", "EFT point")

    if args.scan:
        x = np.arange(-7,8,1)
        y = np.arange(-7,8,1)
    else:
        x = np.array([6])
        y = np.array([0])
    X, Y = np.meshgrid(x, y)

    res_LT = {}

    #runLT = True
    if args.runLT:
        # First run the classical LT analysis.
        # We can use histogram based reweighting here?
        #lt_axis      = hist.Bin("lt",      r"$L_{T}$ (GeV)",   [100,200,300,400,500,600,700,6500])
        lt_axis      = hist.Bin("lt",      r"$L_{T}$ (GeV)",   [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0])

        q_event = sig_test['lt'].values

        q_event_argsort     = np.argsort(q_event)
        q_event_argsort_inv = np.argsort(q_event_argsort)
        cdf_sm = np.cumsum(sig_test['weight'].values[q_event_argsort])
        #cdf_sm = np.cumsum(sig_train['weight'].values[q_event_argsort])
        cdf_sm/=cdf_sm[-1]

        # map to the SM CDF of q
        cdf_map = make_cdf_map( q_event[q_event_argsort], cdf_sm )

        #q_event_cdf = cdf_sm[q_event_argsort_inv] #uniformly distributed under the SM hypothesis
        q_event_cdf = cdf_map( q_event )

        lt_hist = hist.Hist("lt", dataset_axis, lt_axis)
        lt_hist.fill(dataset="TTW", lt=cdf_map(ttW['lt'].values), weight=ttW['weight']*2)
        lt_hist.fill(dataset="TTZ", lt=cdf_map(ttZ['lt'].values), weight=ttZ['weight']*2)
        lt_hist.fill(dataset="TTH", lt=cdf_map(ttH['lt'].values), weight=ttH['weight']*2)
        lt_hist.fill(dataset="NP", lt=cdf_map(NP['lt'].values), weight=NP['weight']*NP['weight_np'])
        lt_hist.fill(dataset="rare", lt=cdf_map(rare['lt']), weight=rare['weight']*2)
        lt_hist.fill(dataset="diboson", lt=cdf_map(diboson['lt']), weight=diboson['weight']*2)
        lt_hist.fill(dataset="XG", lt=cdf_map(XG['lt']), weight=XG['weight']*2)
        lt_hist.fill(dataset="signal", lt=cdf_map(sig_test['lt'].values), weight=sig_test['weight'])

        hist_dict = {
            'TTW': lt_hist["TTW"],
            'TTZ': lt_hist["TTZ"],
            'TTH': lt_hist["TTH"],
            'nonprompt': lt_hist["NP"],
            'conv': lt_hist["XG"],
            'rare': lt_hist["rare"],
            'diboson': lt_hist["diboson"],
            'signal': lt_hist["signal"],
            }
            #'SR': lt_hist}

        # fill all the signal histograms
        sig_lt_hist = hist.Hist("lt", dataset_axis, eft_axis, lt_axis)
        for weight in weights:
            sig_lt_hist.fill(
                dataset='signal',
                lt=cdf_map(sig_test['lt'].values),
                eft=weight,
                weight=sig_test['weight']*sig_test[weight],
            )

        sig_lt_hist_SM = sig_lt_hist.integrate('eft', weights[0])


        if args.fit:
            sm_card = makeCardFromHist(
                        hist_dict,
                        #'SR',
                        ext='_SM_LT',
                        #bsm_hist=sig_lt_hist_SM['signal'].sum('dataset').to_hist(),
                        systematics=systematics,
                    )
            res_sm = card.calcNLL(sm_card)
            res_sm_ll = res_sm['nll0'][0]+res_sm['nll'][0]

        # Now get the polynom for histogram reweighting
        hp = HyperPoly(order=2)
        hp.initialize( coordinates, ref_coordinates )
        hist_coeff = hp.get_parametrization( [histo_values(sig_lt_hist, w) for w in weights] )

        pp_val, pp_unc = sig_lt_hist_SM.sum('dataset').values(sumw2=True)[()]

        z = []

        sm_hist = make_bh(pp_val, pp_unc, lt_axis.edges())
        res_bsm = {}

        for x, y in zip(X.flatten(), Y.flatten()):
            print (f"Working on cpt {x} and cpqm {y} for classical LT analysis")
            point = [x, y]

            bsm_vals = hp.eval(hist_coeff, point)
            bsm_hist = make_bh(
                sumw = bsm_vals,
                sumw2 = (np.sqrt(pp_unc)/pp_val)*bsm_vals,
                edges = lt_axis.edges(),
            )

            if args.plot:
                fig, ax = plt.subplots(figsize=(10,10))

                hep.histplot(
                    [
                        lt_hist['TTW'].sum('dataset').values()[()],
                        lt_hist['TTZ'].sum('dataset').values()[()],
                        lt_hist['TTH'].sum('dataset').values()[()],
                    ],
                    lt_axis.edges(),
                    histtype="fill",
                    stack=True,
                    density=False,
                    label = ["ttW", "ttZ", "ttH"],
                    color = ["#FF595E", "#8AC926", "#1982C4"],
                    ax=ax)

                hep.histplot(
                    [ bsm_hist.values() ],
                    lt_axis.edges(),
                    yerr = [bsm_hist.variances()],
                    histtype="step",
                    stack=False,
                    density=False,
                    linestyle="--",
                    color = ["black"],
                    label = ["top-W scattering (BSM)"],
                    ax=ax)

                ax.set_yscale('log')
                ax.set_xscale('linear')
                ax.legend()

                fig.savefig(f"{plot_dir}/lt_cpt_{x}_cpqm_{y}_bsm.png")

            if args.fit:
                print ("Counts and Variances")
                print (bsm_hist.counts())
                print (np.sqrt(bsm_hist.variances()))
                bsm_card = makeCardFromHist(
                    hist_dict,
                    #'SR',
                    ext='_BSM_LT',
                    bsm_hist = bsm_hist,
                    #sm_vals = sm_hist,
                    systematics=systematics,
                    )

                simple_NLL = 2*np.sum(sm_hist.counts()[:8] - bsm_hist.counts()[:8] - bsm_hist.counts()[:8]*np.log(sm_hist.counts()[:8]/bsm_hist.counts()[:8]))
                print ("### Sanity checks ###")
                print ("- SM integral: {:.2f} +/- {:.2f}".format(sm_hist.sum().value, np.sqrt(sm_hist.sum().variance)))
                print ("- BSM integral: {:.2f} +/- {:.2f}".format(bsm_hist.sum().value, np.sqrt(bsm_hist.sum().variance)))
                print (f"Robert NLL:", simple_NLL)

                res_bsm[(x,y)] = card.calcNLL(bsm_card)
                res_tmp = res_bsm[(x,y)]['nll0'][0]+res_bsm[(x,y)]['nll'][0]
                nll = -2*(res_sm_ll-res_tmp)
                z.append(nll)
                print ('NLL: {:.2f}'.format(nll))
                res_LT[(x,y)] = nll

        if args.fit and args.plot and len(X)>1:
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

            fig.savefig(f'{plot_dir}/lt_scan_test.png')
            fig.savefig(f'{plot_dir}/lt_scan_test.pdf')


    res_BIT = {}

    run_BIT = True
    if run_BIT:
        z = []

        res_bsm = {}

        ## BIT results
        bit_bins = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        #bit_bins = [0., 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
        bit_axis = hist.Bin("bit", r"BIT score", bit_bins)

        # this is just plotting the predicted coefficients
        eft_weight = hp.eval(coeffs.transpose(), [6,0])
        for i in range(6):
            bit_hist = hist.Hist("bit", dataset_axis, bit_axis)
            bit_hist.fill(dataset="TTW", bit=ttW['pred_%s'%i].values, weight=ttW['weight']*2)
            bit_hist.fill(dataset="TTZ", bit=ttZ['pred_%s'%i].values, weight=ttZ['weight']*2)
            bit_hist.fill(dataset="TTH", bit=ttH['pred_%s'%i].values, weight=ttH['weight']*2)
            bit_hist.fill(dataset="signal", bit=sig_test['pred_%s'%i].values, weight=sig_test['weight'])
            bit_hist.fill(dataset="signal_bsm", bit=sig_test['pred_%s'%i].values, weight=sig_test['weight']*eft_weight)
            bit_hist.fill(dataset="signal_train", bit=sig_train['pred_%s'%i].values, weight=sig_train['weight'])

            fig, ax = plt.subplots(figsize=(10,10))

            hep.histplot(
                [
                    bit_hist['TTW'].sum('dataset').values()[()],
                    bit_hist['TTZ'].sum('dataset').values()[()],
                    bit_hist['TTH'].sum('dataset').values()[()],
                ],
                bit_axis.edges(),
                histtype="fill",
                stack=True,
                density=True,
                label = ["ttW", "ttZ", "ttH"],
                color = ["#FF595E", "#8AC926", "#1982C4"],
                ax=ax)

            hep.histplot(
                [ bit_hist['signal'].sum('dataset').values()[()], bit_hist['signal_bsm'].sum('dataset').values()[()] ],
                bit_axis.edges(),
                histtype="step",
                stack=False,
                density=True,
                linestyle="--",
                color = ["black", "green"],
                label = ["top-W scattering (SM)", "top-W (cpt=6,cpQM=0)" ],
                ax=ax)

            ax.set_yscale('log')
            ax.legend()

            print (f"Saving plot for coeff {i}")
            fig.savefig(f"{plot_dir}/pred_coeff_{i}.png")

            fig, ax = plt.subplots(figsize=(10,10))

            h_test = Hist1D(sig_test[f'pred_{i}'].values, bins=coeff_bins, overflow=True)
            h_train = Hist1D(sig_train[f'pred_{i}'].values, bins=coeff_bins, overflow=True)
            h_signal = Hist1D(sig_train[f'coeff_{i}'].values, bins=coeff_bins, overflow=True)

            hep.histplot(
                [ h_test.counts, h_train.counts, h_signal.counts ],
                h_signal.edges,
                histtype="step",
                stack=False,
                density=True,
                #linestyle="--",
                color = ["black", "green", "red"],
                label = ["Signal, test", "Signal, training", "Theory (reference)" ],
                ax=ax)

            ax.set_yscale('log')
            ax.legend()

            print (f"Saving plot for coeff {i}")
            fig.savefig(f"{plot_dir}/train_test_pred_coeff_{i}.png")



        #bit_bins = [0., 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
        bit_bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        #bit_bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bit_axis = hist.Bin("bit", r"BIT score", bit_bins)

        for x, y in zip(X.flatten(), Y.flatten()):
            point = [x, y]
            print (f"Working on cpt={x}, cpqm={y} now")

            bit_hist = hist.Hist("bit", dataset_axis, bit_axis)

            sig_bit = get_bit_score(sig_test, cpt=x, cpqm=y, trans=None)
            #sig_bit = get_bit_score(sig_train, cpt=x, cpqm=y, trans=None)
            q_event = sig_bit

            print (np.mean(sig_bit))

            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(n_quantiles=40, random_state=0)
            qt.fit(q_event.reshape(-1, 1))

            q_event_cdf = qt.transform(q_event.reshape(-1,1)).flatten()

            store_transformer(qt, version=f'{args.version}_cpt_{x}_cpqm_{y}')

            bit_hist.fill(dataset="TTW",        bit=get_bit_score(ttW, cpt=x, cpqm=y, trans=qt), weight=ttW['weight']*2)
            bit_hist.fill(dataset="TTZ",        bit=get_bit_score(ttZ, cpt=x, cpqm=y, trans=qt), weight=ttZ['weight']*2)
            bit_hist.fill(dataset="TTH",        bit=get_bit_score(ttH, cpt=x, cpqm=y, trans=qt), weight=ttH['weight']*2)
            bit_hist.fill(dataset="NP",         bit=get_bit_score(NP, cpt=x, cpqm=y, trans=qt), weight=NP['weight']*NP['weight_np'])
            bit_hist.fill(dataset="rare",       bit=get_bit_score(rare, cpt=x, cpqm=y, trans=qt), weight=rare['weight']*2)
            bit_hist.fill(dataset="diboson",    bit=get_bit_score(diboson, cpt=x, cpqm=y, trans=qt), weight=diboson['weight']*2)
            bit_hist.fill(dataset="XG",         bit=get_bit_score(XG, cpt=x, cpqm=y, trans=qt), weight=XG['weight']*2)
            bit_hist.fill(dataset="signal",     bit=q_event_cdf, weight=sig_test['weight'])

            print (bit_hist['signal'].values())

            hist_dict = {
                'TTW': bit_hist['TTW'],
                'TTZ': bit_hist['TTZ'],
                'TTH': bit_hist['TTH'],
                'signal': bit_hist['signal'],
                'nonprompt': bit_hist['NP'],
                'rare': bit_hist['rare'],
                'conv': bit_hist['XG'],
                'diboson': bit_hist['diboson'],
                }

            if args.fit:
                sm_card = makeCardFromHist(
                    hist_dict,
                    #'SR',
                    ext='_SM2',
                    #bsm_hist=bit_hist['signal'].sum('dataset').to_hist(),
                    systematics=systematics,
                   )

                res_sm = card.calcNLL(sm_card)
                res_sm_ll = res_sm['nll0'][0]+res_sm['nll'][0]

            eft_weight = hp.eval(coeffs.transpose(), [x,y])
            #eft_weight = hp.eval(coeffs_train.transpose(), [x,y])

            bsm_hist = bh.Histogram(bh.axis.Variable(bit_bins), storage=bh.storage.Weight())
            bsm_hist.fill(q_event_cdf, weight=sig_test['weight']*eft_weight)

            #print (bsm_hist.values())
            #print (bsm_hist.variances())
            #bsm_hist.fill(q_event_cdf, weight=sig_train['weight']*eft_weight)

            #sm_hist = bh.numpy.histogram([], bins=bit_bins, histogram=bh.Histogram)
            sm_hist = bh.Histogram(bh.axis.Variable(bit_bins), storage=bh.storage.Weight())
            sm_hist.fill(q_event_cdf, weight=sig_test['weight'])
            #sm_hist.fill(q_event_cdf, weight=sig_train['weight'])

            if args.plot:
                fig, ax = plt.subplots(figsize=(10,10))

                hep.cms.label(
                    "Preliminary",
                    data=True,
                    lumi=60,
                    com=13,
                    loc=0,
                    ax=ax,
                )

                hep.histplot(
                    [
                        bit_hist['NP'].sum('dataset').values()[()],
                        bit_hist['rare'].sum('dataset').values()[()],
                        bit_hist['XG'].sum('dataset').values()[()],
                        bit_hist['diboson'].sum('dataset').values()[()],
                        bit_hist['TTZ'].sum('dataset').values()[()],
                        bit_hist['TTW'].sum('dataset').values()[()],
                        bit_hist['TTH'].sum('dataset').values()[()],
                    ],
                    bit_hist.axes()[1].edges(),
                    yerr=[
                        np.sqrt(bit_hist['NP'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['rare'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['XG'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['diboson'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['TTZ'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['TTW'].sum('dataset').values(sumw2=True)[()][1]),
                        np.sqrt(bit_hist['TTH'].sum('dataset').values(sumw2=True)[()][1]),
                    ],
                    histtype="fill",
                    stack=True,
                    density=False,
                    label = [ "Nonprompt", "Rare", "XG", "diboson", "ttZ", "ttW", "ttH"],
                    color = [colors["non prompt"], colors["rare"], colors["XG"], colors["diboson"], colors["TTZ"], colors["TTW"], colors["TTH"]],
                    ax=ax)

                hep.histplot(
                    [ bsm_hist.values() ],
                    bsm_hist.axes[0].edges,
                    yerr = [np.sqrt(bsm_hist.variances())],
                    histtype="step",
                    stack=False,
                    density=False,
                    linestyle="--",
                    color = ["black"],
                    label = ["Signal"],
                    ax=ax)

                ax.set_yscale('log')
                ax.legend(ncol=3)

                ax.set_xlabel(r'transformed score')
                ax.set_ylabel(r'Events')

                fig.savefig(f"{plot_dir}/bit_cpt_{x}_cpqm_{y}_transformed_bsm.png")


                fig, ax = plt.subplots(figsize=(10,10))

                hep.histplot(
                    [
                        bit_hist['TTW'].sum('dataset').values()[()],
                        bit_hist['TTZ'].sum('dataset').values()[()],
                        bit_hist['TTH'].sum('dataset').values()[()],
                    ],
                    bit_hist.axes()[1].edges(),
                    histtype="fill",
                    stack=True,
                    density=False,
                    label = ["ttW", "ttZ", "ttH"],
                    color = ["#FF595E", "#8AC926", "#1982C4"],
                    ax=ax)

                hep.histplot(
                    [ sm_hist.values() ],
                    sm_hist.axes[0].edges,
                    histtype="step",
                    stack=False,
                    density=False,
                    linestyle="--",
                    color = ["black"],
                    label = ["top-W scattering (SM)"],
                    ax=ax)

                ax.set_yscale('log')
                ax.legend()

                fig.savefig(f"{plot_dir}/bit_cpt_{x}_cpqm_{y}_transformed_sm.png")

            if args.fit:

                print ("Counts and Variances")
                print (bsm_hist.counts())
                print (np.sqrt(bsm_hist.variances()))

                bsm_card = makeCardFromHist(
                    hist_dict,
                    #'SR',
                    ext='_BSM2',
                    bsm_hist = bsm_hist,
                    #sm_vals = sm_hist,
                    systematics=systematics,
                )

                simple_NLL = 2*np.sum(sm_hist.counts()[:8] - bsm_hist.counts()[:8] - bsm_hist.counts()[:8]*np.log(sm_hist.counts()[:8]/bsm_hist.counts()[:8]))
                print ("### Sanity checks ###")
                print ("- SM integral: {:.2f} +/- {:.2f}".format(sm_hist.sum().value, np.sqrt(sm_hist.sum().variance)))
                print ("- BSM integral: {:.2f} +/- {:.2f}".format(bsm_hist.sum().value, np.sqrt(bsm_hist.sum().variance)))
                print (f"Robert NLL:", simple_NLL)


                res_bsm[(x,y)] = card.calcNLL(bsm_card)
                res_tmp = res_bsm[(x,y)]['nll0'][0]+res_bsm[(x,y)]['nll'][0]
                nll = -2*(res_sm_ll-res_tmp)
                z.append(nll)
                print ('NLL: {:.2f}'.format(nll))
                res_BIT[(x,y)] = nll

        if args.fit and args.plot and len(X)>1:
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

            fig.savefig(f'{plot_dir}/bit_scan_test.png')
            fig.savefig(f'{plot_dir}/bit_scan_test.pdf')
