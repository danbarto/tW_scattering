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

from BoostedInformationTreeP3 import BoostedInformationTree

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils import resample, shuffle

from yahist import Hist1D

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

def histo_values(histo, weight):
    return histo.integrate('eft', weight).sum('dataset').values()[()]

def get_bit_score(df, cpt=0, cpqm=0, trans=None):
    tmp = (
        df['pred_1'].values*cpt + \
        df['pred_2'].values*cpqm + \
        #0.5*df['pred_3'].values*cpt**2 + \
        #0.5*df['pred_4'].values*cpt*cpqm + \
        #0.5*df['pred_5'].values*cpqm**2 + \
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

variables = sorted(variables)


if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--retrain', action='store_true', default=None, help="Retrain the BDT?")
    argParser.add_argument('--fit', action='store_true', default=None, help="Run combine fit?")
    argParser.add_argument('--plot', action='store_true', default=None, help="Make all the plots")
    argParser.add_argument('--version', action='store', default='v1', help="Version number")
    # NOTE: need to add the full Run2 training option back
    args = argParser.parse_args()

    # load data
    sample_list =  ['TTW', 'TTZ','TTH']
    years = ['2016APV', '2016', '2017', '2018']

    df_bkg = pd.DataFrame()
    for year in years:
        tmp = pd.concat([pd.read_hdf(f"../processor/multiclass_input_{sample}_{year}.h5") for sample in sample_list])
        df_bkg = pd.concat([df_bkg, tmp])
        df_bkg = df_bkg[df_bkg['SS']==1]
    del tmp

    df_signal = pd.concat([pd.read_hdf(f"../processor/multiclass_input_topW_{year}.h5") for year in years])


    # Prepare hyper poly inputs
    samples = get_samples("samples_UL17.yaml")
    f_in    = samples['/ceph/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL17_NanoAODv9_NANO_v11/']['files'][0]
    tree    = uproot.open(f_in)['Events']

    hp = HyperPoly(2)
    coordinates, ref_coordinates = get_coordinates_and_ref(f_in)
    ref_coordinates = [0.0,0.0]

    hp.initialize( coordinates, ref_coordinates )
    weights = [ x.replace('LHEWeight_','') for x in tree.keys() if x.startswith('LHEWeight_c') ]

    coeff = hp.get_parametrization( [df_signal[w].values for w in weights] )


    # Preprocessing inputs
    df_signal['lt'] = (df_signal['lead_lep_pt'].values + df_signal['sublead_lep_pt'].values + df_signal['met'].values)
    df_bkg['lt'] = (df_bkg['lead_lep_pt'].values + df_bkg['sublead_lep_pt'].values + df_bkg['met'].values)

    # this roughly corresponds to labeling
    for i in range(6):
        df_signal['coeff_%s'%i] = coeff[i,:]
        df_bkg['coeff_%s'%i] = np.zeros(len(df_bkg))

    df_signal = df_signal[((df_signal['n_fwd']>=1))]
    df_bkg = df_bkg[((df_bkg['SS']==1)&(df_bkg['n_fwd']>=1))]

    # signal
    sig_train_sel   = ((df_signal['OS']==1) & (df_signal['nLepFromTop']==1)) #((df_signal['event']%2)==1).values
    sig_test_sel    = (df_signal['SS']==1)#~sig_train_sel
    sig_train       = df_signal[sig_train_sel]
    sig_test        = df_signal[sig_test_sel]

    # bkg
    bkg_train_sel   = (df_bkg['event']%2)==1
    bkg_test_sel    = ~bkg_train_sel
    bkg_train       = df_bkg[bkg_train_sel]
    bkg_test        = df_bkg[bkg_test_sel]

    train = pd.concat([sig_train, bkg_train])
    train = shuffle(train)

    # Train the tree for signal only
    n_trees       = 50
    learning_rate = 0.3
    max_depth     = 2
    min_size      = 20
    calibrated    = False

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(sig_train[variables])
    params = scaler.get_params()

    if os.path.isfile('bit.pkl') and not args.retrain:
        with open('bit.pkl', 'rb') as f:
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

        with open('bit.pkl', 'wb') as f:
            pickle.dump(bit, f)

    # Plots
    plot_dir = f'/home/users/dspitzba/public_html/tW_scattering/BIT/{args.version}/'
    finalizePlotDir(plot_dir)
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


    # Train all the trees!

    n_trees       = 50
    learning_rate = 0.3
    max_depth     = 2
    min_size      = 20

    training_features = train[variables].values
    training_weights = abs(train['weight'].values)
    scaler = RobustScaler()
    training_features_scaled = scaler.fit_transform(training_features)
    params = scaler.get_params()

    if os.path.isfile('bits.pkl') and not args.retrain:
        with open('bits.pkl', 'rb') as f:
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
                    calibrated            = True)
            )

            bits[-1].boost()

        with open('bits.pkl', 'wb') as f:
            pickle.dump(bits, f)


    # Evaluate all the BITS and store results
    scaled_test = scaler.transform(sig_test[variables].values)
    scaled_bkg_test = scaler.transform(bkg_test[variables].values)

    for i in range(6):
        sig_test['pred_%s'%i] = bits[i].vectorized_predict(scaled_test)
        bkg_test['pred_%s'%i] = bits[i].vectorized_predict(scaled_bkg_test)

    ttW = bkg_test[bkg_test['label']==1]
    ttZ = bkg_test[bkg_test['label']==2]
    ttH = bkg_test[bkg_test['label']==3]

    # Run the analysis

    from coffea import hist

    from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
    from Tools.dataCard import dataCard
    card = dataCard(releaseLocation=os.path.expandvars('/home/users/dspitzba/TOP/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

    dataset_axis = hist.Cat("dataset", "Primary dataset")
    eft_axis = hist.Cat("eft", "EFT point")

    run_LT = True
    if run_LT:
        # First run the classical LT analysis.
        # We can use histogram based reweighting here?
        lt_axis      = hist.Bin("lt",      r"$L_{T}$ (GeV)",   [100,200,300,400,500,600,700,6500])

        lt_hist = hist.Hist("lt", dataset_axis, lt_axis)
        lt_hist.fill(dataset="TTW", lt=ttW['lt'].values, weight=ttW['weight']*2)
        lt_hist.fill(dataset="TTZ", lt=ttZ['lt'].values, weight=ttZ['weight']*2)
        lt_hist.fill(dataset="TTH", lt=ttH['lt'].values, weight=ttH['weight']*2)
        lt_hist.fill(dataset="signal", lt=sig_test['lt'].values, weight=sig_test['weight'])

        hist_dict = {'SR': lt_hist}

        sig_lt_hist = hist.Hist("lt", dataset_axis, eft_axis, lt_axis)
        for weight in weights:
            sig_lt_hist.fill(
                dataset='signal',
                lt=sig_test['lt'].values,
                eft=weight,
                weight=sig_test['weight']*sig_test[weight],
            )

        sig_lt_hist_SM = sig_lt_hist.integrate('eft', weights[0])

        if args.fit:
            sm_card = makeCardFromHist(
                        hist_dict,
                        'SR',
                        ext='_SM',
                        signal_hist=sig_lt_hist_SM,
                    )
            res_sm = card.calcNLL(sm_card)
            res_sm_ll = res_sm['nll0'][0]+res_sm['nll'][0]

        # Now get the polynom for histogram reweighting
        hp = HyperPoly(order=2)
        hp.initialize( coordinates, ref_coordinates )
        hist_coeff = hp.get_parametrization( [histo_values(sig_lt_hist, w) for w in weights] )

        pp_val, pp_unc = sig_lt_hist_SM.sum('dataset').values(sumw2=True)[()]

        x = np.arange(-10,11,4)
        y = np.arange(-10,11,4)
        X, Y = np.meshgrid(x, y)
        z = []

        sm_hist = make_bh(pp_val, pp_unc, lt_axis.edges())
        res_bsm = {}

        for x, y in zip(X.flatten(), Y.flatten()):
            point = [x, y]

            bsm_vals = hp.eval(hist_coeff, point)
            print (sum(bsm_vals))
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
                    histtype="step",
                    stack=False,
                    density=False,
                    linestyle="--",
                    color = ["black"],
                    label = ["top-W scattering (BSM)"],
                    ax=ax)

                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.legend()

                fig.savefig(f"{plot_dir}/lt_cpt_{x}_cpqm_{y}_bsm.png")

            if args.fit:
                bsm_card = makeCardFromHist(
                    hist_dict,
                    'SR',
                    ext='_BSM',
                    bsm_vals = bsm_hist,
                    sm_vals = sm_hist,
                    )

                res_bsm[(x,y)] = card.calcNLL(bsm_card)
                res_tmp = res_bsm[(x,y)]['nll0'][0]+res_bsm[(x,y)]['nll'][0]
                z.append((-2*(res_sm_ll-res_tmp)))

        if args.fit:
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


    run_BIT = True
    if run_BIT:
        x = np.arange(-10,11,4)
        y = np.arange(-10,11,4)
        X, Y = np.meshgrid(x, y)
        z = []

        res_bsm = {}

        ## BIT results
        bit_axis = hist.Bin("bit", r"BIT score",  [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

        for x, y in zip(X.flatten(), Y.flatten()):
            point = [x, y]
            print (f"Working on cpt={x}, cpqm={y} now")

            bit_hist = hist.Hist("bit", dataset_axis, bit_axis)

            sig_bit = get_bit_score(sig_test, cpt=x, cpqm=y, trans=None)
            q_event = sig_bit

            q_event_argsort     = np.argsort(q_event)
            q_event_argsort_inv = np.argsort(q_event_argsort)
            cdf_sm = np.cumsum(sig_test['weight'].values[q_event_argsort])
            cdf_sm/=cdf_sm[-1]

            # map to the SM CDF of q
            cdf_map = make_cdf_map( q_event[q_event_argsort], cdf_sm )

            #q_event_cdf = cdf_sm[q_event_argsort_inv] #uniformly distributed under the SM hypothesis
            q_event_cdf = cdf_map( q_event )

            #qt = QuantileTransformer(n_quantiles=40, random_state=0)
            #qt.fit(sig_bit.reshape(-1, 1))
            #sig_bit_trans = qt.transform(sig_bit.reshape(-1,1)).flatten()

            bit_hist.fill(dataset="TTW", bit=cdf_map(get_bit_score(ttW, cpt=x, cpqm=y, trans=None)), weight=ttW['weight']*2)
            bit_hist.fill(dataset="TTZ", bit=cdf_map(get_bit_score(ttZ, cpt=x, cpqm=y, trans=None)), weight=ttZ['weight']*2)
            bit_hist.fill(dataset="TTH", bit=cdf_map(get_bit_score(ttH, cpt=x, cpqm=y, trans=None)), weight=ttH['weight']*2)
            bit_hist.fill(dataset="signal", bit=q_event_cdf, weight=sig_test['weight'])

            hist_dict = {'SR': bit_hist}

            if args.fit:
                sm_card = makeCardFromHist(
                    hist_dict,
                    'SR',
                    ext='_SM',
                    signal_hist=bit_hist['signal'],
                   )

                res_sm = card.calcNLL(sm_card)
                res_sm_ll = res_sm['nll0'][0]+res_sm['nll'][0]

            eft_weight = hp.eval(coeffs.transpose(), [x,y])

            bsm_hist = bh.Histogram(bh.axis.Regular(10, 0, 1))
            bsm_hist.fill(q_event_cdf, weight=sig_test['weight']*eft_weight)

            sm_hist = bh.Histogram(bh.axis.Regular(10, 0, 1))
            sm_hist.fill(q_event_cdf, weight=sig_test['weight'])

            print (sum(sig_test['weight']*eft_weight))

            if args.plot:
                fig, ax = plt.subplots(figsize=(10,10))

                hep.histplot(
                    [
                        bit_hist['TTW'].sum('dataset').values()[()],
                        bit_hist['TTZ'].sum('dataset').values()[()],
                        bit_hist['TTH'].sum('dataset').values()[()],
                    ],
                    h.edges,
                    histtype="fill",
                    stack=True,
                    density=False,
                    label = ["ttW", "ttZ", "ttH"],
                    color = ["#FF595E", "#8AC926", "#1982C4"],
                    ax=ax)

                hep.histplot(
                    [ bsm_hist.values() ],
                    h.edges,
                    histtype="step",
                    stack=False,
                    density=False,
                    linestyle="--",
                    color = ["black"],
                    label = ["top-W scattering (BSM)"],
                    ax=ax)

                ax.set_yscale('log')
                ax.legend()

                fig.savefig(f"{plot_dir}/bit_cpt_{x}_cpqm_{y}_transformed_bsm.png")


                fig, ax = plt.subplots(figsize=(10,10))

                hep.histplot(
                    [
                        bit_hist['TTW'].sum('dataset').values()[()],
                        bit_hist['TTZ'].sum('dataset').values()[()],
                        bit_hist['TTH'].sum('dataset').values()[()],
                    ],
                    h.edges,
                    histtype="fill",
                    stack=True,
                    density=False,
                    label = ["ttW", "ttZ", "ttH"],
                    color = ["#FF595E", "#8AC926", "#1982C4"],
                    ax=ax)

                hep.histplot(
                    [ sm_hist.values() ],
                    h.edges,
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
                bsm_card = makeCardFromHist(
                    hist_dict,
                    'SR',
                    ext='_BSM',
                    bsm_vals = bsm_hist,
                    sm_vals = sm_hist,
                )

                res_bsm[(x,y)] = card.calcNLL(bsm_card)
                res_tmp = res_bsm[(x,y)]['nll0'][0]+res_bsm[(x,y)]['nll'][0]
                z.append((-2*(res_sm_ll-res_tmp)))

        if args.fit:
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
