import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import copy

import scipy
from yahist import Hist1D

import tensorflow as tf
from keras.utils import np_utils

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

from Tools.dataCard import dataCard
from Tools.helpers import finalizePlotDir, mt
from Tools.limits import makeCardFromHist
from processor.default_accumulators import dataset_axis

import warnings
warnings.filterwarnings('ignore')

from coffea import processor, hist

import matplotlib.pyplot as plt
from plots.helpers import makePlot

import mplhep
plt.style.use(mplhep.style.CMS)


def baseline_model(input_dim, out_dim):
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2*input_dim, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization() )
    model.add( tf.keras.layers.Dropout( rate = 0.3 ) )
    model.add(tf.keras.layers.Dense(2*input_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization() )
    model.add( tf.keras.layers.Dropout( rate = 0.3 ) ) # this introduces some stochastic behavior
    model.add(tf.keras.layers.Dense(out_dim, activation='softmax'))
    
    #opt = tf.keras.optimizers.SGD(lr=0.1)
    #opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.RMSprop(lr=0.001) ## performs best.

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model


def simpleAccuracy(pred, dummy_y):
    pred_Y = np.argmax(pred, axis=1)
    true_Y = np.argmax(dummy_y, axis=1)
    
    # this gives the measure of correctly tagged events, over the total
    return sum(pred_Y == true_Y)/len(true_Y)


def test_train(test, train, y_test, y_train, labels=[], bins=25, node=0, plot_dir=None, weight_test=None, weight_train=None):
    ks = {}

    fig, ax = plt.subplots(1,1,figsize=(10,10))

    h = {}
    for i, label in enumerate(labels):
        
        _ks, _p = scipy.stats.kstest(
            train[:,node][(y_train==i)],
            test[:,node][(y_test==i)]
        )
        
        ks[label] = (_p, _ks)

        h[label+'_test'] = Hist1D(test[:,node][(y_test==i)], bins=bins, weights=weight_test[(y_test==i)]).normalize()
        h[label+'_train'] = Hist1D(train[:,node][(y_train==i)], bins=bins, label=label+' (p=%.2f, KS=%.2f)'%(_p, _ks), weights=weight_train[(y_train==i)]).normalize()
        

        h[label+'_test'].plot(color=colors[i], histtype="step", ls='--', linewidth=2)
        h[label+'_train'].plot(color=colors[i], histtype="step", linewidth=2)

    if plot_dir:
        finalizePlotDir(plot_dir)
        fig.savefig("{}/score_node_{}.png".format(plot_dir, node))
        fig.savefig("{}/score_node_{}.pdf".format(plot_dir, node))
    
    return ks


def test_train_cat(test, train, y_test, y_train, labels=[], n_cat=5, plot_dir=None, weight_test=None, weight_train=None):
    ks = {}
    bins = [x-0.5 for x in range(n_cat+1)]
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    
    h = {}
    for i, label in enumerate(labels):
        
        _ks, _p = scipy.stats.kstest(
            np.argmax(train, axis=1)[(y_train==i)],
            np.argmax(test, axis=1)[(y_test==i)]
        )

        ks[label] = (_p, _ks)
        
        h[label+'_test'] = Hist1D(np.argmax(test, axis=1)[(y_test==i)], bins=bins, weights=weight_test[(y_test==i)]).normalize()
        h[label+'_train'] = Hist1D(np.argmax(train, axis=1)[(y_train==i)], bins=bins, label=label+' (p=%.2f, KS=%.2f)'%(_p, _ks), weights=weight_train[(y_train==i)]).normalize()
        

        h[label+'_test'].plot(color=colors[i], histtype="step", ls='--', linewidth=2)
        h[label+'_train'].plot(color=colors[i], histtype="step", linewidth=2)
        
    ax.set_ylabel('a.u.')
    ax.set_xlabel('category')

    ax.set_ylim(0,1/n_cat*5)

    if plot_dir:
        finalizePlotDir(plot_dir)
        fig.savefig("{}/categories.png".format(plot_dir))
        fig.savefig("{}/categories.pdf".format(plot_dir))

    return ks


def get_ROC(test, train, y_test, y_train, node=0):

    y_test_binary = (y_test!=node)*0 + (y_test==node)*1

    fpr_test, tpr_test, thresholds_test = roc_curve( y_test_binary, test[:,node] )
    auc_val_test = auc(fpr_test, tpr_test)

    plt.plot( tpr_test, 1-fpr_test, 'b', label= 'AUC NN (test)=' + str(round(auc_val_test,4) ))

    y_train_binary = (y_train!=node)*0 + (y_train==node)*1
    
    fpr_train, tpr_train, thresholds_test = roc_curve( y_train_binary, train[:,node]  )
    auc_val_train = auc(fpr_train, tpr_train)

    plt.plot( tpr_train, 1-fpr_train, 'r', label= 'AUC NN (train)=' + str(round(auc_val_train,4) ))

    plt.xlabel('$\epsilon_{Sig}$', fontsize = 20) # 'False positive rate'
    plt.ylabel('$1-\epsilon_{Back}$', fontsize = 20) #  '1-True positive rate' 
    plt.legend(loc ='lower left')


def prepare_data(f_in, selection, robust=False, reuse=False, fout=None):

    if not reuse:
        df = pd.read_hdf(f_in) # load data processed with ML_processor.py
        df = df[selection]

        labels = df['label'].values
        
        df = df[(labels<5)]
        labels = labels[labels<5]

        df_train, df_test, y_train, y_test = train_test_split(df, labels, train_size= int( 0.9*labels.shape[0] ), random_state=42 )

        df_train['test_label'] = np.zeros(len(df_train))
        df_test['test_label'] = np.ones(len(df_test))

        df = pd.concat([df_train, df_test])

        if fout:
            df.to_hdf(fout, key='df', format='table', mode='w')

    else:
        df = pd.read_hdf(f_in)

        df_train = df[df['test_label']==0]
        df_test = df[df['test_label']==1]

        y_train = df_train['label'].values
        y_test = df_test['label'].values

    print ("Number of non-prompt events (=ttbar):", len(df[df['label']==4]))

    return df_train, df_test, y_train, y_test#, df_mean, df_std


def get_one_hot(labels):

    encoder = LabelEncoder()
    encoder.fit(labels)
    return np_utils.to_categorical(labels)


def store_model(model, scaler, version='v5'):

    model.save('networks/weights_%s.h5a'%version)
    joblib.dump(scaler, 'networks/scaler_%s.joblib'%version)


def load_model(version='v5'):

    model = tf.keras.models.load_model(os.path.expandvars('$TWHOME/ML/networks/weights_%s.h5a'%version))
    scaler = joblib.load(os.path.expandvars('$TWHOME/ML/networks/scaler_%s.joblib'%version))
    return model, scaler


def get_class_weight(df):
    '''
    balance the total yield of the different classes.
    '''
    return {i: 1/sum(df[df['label']==i]['weight']) for i in range(5)}


if __name__ == '__main__':

    load_weights = False
    version = 'v7'

    plot_dir = "/home/users/dspitzba/public_html/tW_scattering/ML/"

    df = pd.read_hdf('data/multiclass_input.h5')

    variables = [
        ## best results with all variables, but should get pruned at some point...
        'n_jet',
        #'n_central',
        ##'n_fwd',
        'n_tau',
        'n_track',
        'st',
        #'ht',
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
        ###'weight', # this does of course _not_ work 🤡 
    ]

    preselection = ((df['n_jet']>2) & (df['n_btag']>0) & (df['n_lep_tight']==2) & (df['n_fwd']>0))
    
    colors = ['gray', 'blue', 'red', 'green', 'orange']
    
    bins = [x/20 for x in range(21)]

    #df_train, df_test, y_train_int, y_test_int = prepare_data('data/multiclass_input.h5', preselection, reuse=False, fout='data/multiclass_input_split.h5')
    df_train, df_test, y_train_int, y_test_int  = prepare_data('data/multiclass_input_split.h5', preselection, reuse=True)
    
    X_train = df_train[variables].values
    X_test  = df_test[variables].values

    y_train = get_one_hot(y_train_int)
    y_test = get_one_hot(y_test_int)
    
    class_weight = get_class_weight(df_train)

    input_dim = len(variables)
    out_dim = len(y_train[0])

    '''
    # Can't use pipelines, unfortunately
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('NN', baseline_model()),
    ])
    '''

    if not load_weights:

        epochs = 50
        batch_size = 5120
        validation_split = 0.2

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        params = scaler.get_params()

        model = baseline_model(input_dim, out_dim)

        history = model.fit(
            X_train_scaled,
            y_train,
            epochs = epochs,
            batch_size = batch_size,
            verbose = 0,
            class_weight = class_weight,
            sample_weight = df_train['weight'].values,
        )

        store_model(model, scaler, version=version)

    else:
        model, scaler = load_model(version=version)

        X_train_scaled = scaler.transform(X_train)
        print ("Loaded weights.")

    X_all = df[variables].values

    X_all_scaled  = scaler.transform(X_all)
    X_test_scaled = scaler.transform(X_test)

    pred_all    = model.predict( X_all_scaled )
    pred_train  = model.predict( X_train_scaled )
    pred_test   = model.predict( X_test_scaled )
    
    df['score_topW'] = pred_all[:,0]
    df['score_ttW'] = pred_all[:,1]
    df['score_ttZ'] = pred_all[:,2]
    df['score_ttH'] = pred_all[:,3]
    df['score_ttbar'] = pred_all[:,4]
    df['score_best'] = np.argmax(pred_all, axis=1)
    
    processes = {
        'topW_v2': df[df['label']==0],
        'TTW': df[df['label']==1],
        'TTZ': df[df['label']==2],
        'TTH': df[df['label']==3],
        'ttbar': df[df['label']==4],
        'rare': df[df['label']==5],
    }

    sel_baseline = ((df['n_lep_tight']==2) & (df['n_fwd']>0) & (df['n_jet']>4) & (df['n_central']>3) & (df['st']>600) & (df['met']>50) & (df['delta_eta_jj']>2.0) & (df['fwd_jet_p']>500))

    signal_baseline = sum(df[(sel_baseline & (df['label']==0))]['weight']) * 137
    bkg_baseline = sum(df[(sel_baseline & (df['label']!=0))]['weight']) * 137

    print ("w/o NN, baseline expectations are:")
    print (" - signal: %.2f"%signal_baseline)
    print (" - bkg: %.2f"%bkg_baseline)

    sel_topW = ((df['score_best']==0) & (df['n_lep_tight']==2) & (df['n_fwd']>0) )

    signal_NN_baseline = sum(df[(sel_topW & (df['label']==0))]['weight']) * 137
    bkg_NN_baseline = sum(df[(sel_topW & (df['label']!=0))]['weight']) * 137

    print ("w/ NN, baseline expectations are:")
    print (" - signal: %.2f"%signal_NN_baseline)
    print (" - nonprompt: %.2f"%bkg_NN_baseline)

    sel_topW_pos = (sel_topW & (df['lead_lep_charge']>0))
    sel_topW_neg = (sel_topW & (df['lead_lep_charge']<0))

    sel_BL_pos = (sel_baseline & (df['lead_lep_charge']>0))
    sel_BL_neg = (sel_baseline & (df['lead_lep_charge']<0))

    sel_ttW = ((df['score_best']==1))
    sel_ttZ = ((df['score_best']==2))
    sel_ttH = ((df['score_best']==3))
    sel_ttbar = ((df['score_best']==4))

    score_axis      = hist.Bin("score", r"N", 8, 0.20, 0.6)
    momentum_axis   = hist.Bin("p",     r"p", 20, 0, 500)
    
    h_score_topW_pos = hist.Hist("score", dataset_axis, score_axis)
    h_score_topW_neg = hist.Hist("score", dataset_axis, score_axis)
    
    h_p_topW_pos = hist.Hist("p", dataset_axis, momentum_axis)
    h_p_topW_neg = hist.Hist("p", dataset_axis, momentum_axis)

    for proc in processes:
        h_score_topW_pos.fill(dataset=proc, score=processes[proc][sel_topW_pos]["score_topW"].values, weight=processes[proc][sel_topW_pos]["weight"].values*137)
        h_score_topW_neg.fill(dataset=proc, score=processes[proc][sel_topW_neg]["score_topW"].values, weight=processes[proc][sel_topW_neg]["weight"].values*137)

        h_p_topW_pos.fill(dataset=proc, p=processes[proc][sel_BL_pos]["lead_lep_pt"].values, weight=processes[proc][sel_BL_pos]["weight"].values*137)
        h_p_topW_neg.fill(dataset=proc, p=processes[proc][sel_BL_neg]["lead_lep_pt"].values, weight=processes[proc][sel_BL_neg]["weight"].values*137)
    
    output = {
        'score_topW_pos': h_score_topW_pos,
        'score_topW_neg': h_score_topW_neg,

        'p_topW_pos': h_p_topW_pos,
        'p_topW_neg': h_p_topW_neg,

    }

    makePlot(output, 'score_topW_pos', 'score',
         log=False, normalize=False, axis_label=r'$top-W\ score$',
         save=plot_dir+'/score_topW_pos',
        )

    makePlot(output, 'score_topW_pos', 'score',
         log=False, normalize=False, shape=True, axis_label=r'$top-W\ score$',
         save=plot_dir+'/score_topW_pos_shape', ymax=0.5,
        )

    makePlot(output, 'score_topW_neg', 'score',
         log=False, normalize=False, axis_label=r'$top-W\ score$',
         save=plot_dir+'/score_topW_neg',
        )

    # cards with NN    
    SR_NN_card_pos = makeCardFromHist(output, 'score_topW_pos', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True)
    SR_NN_card_neg = makeCardFromHist(output, 'score_topW_neg', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True)

    # as a comparison, cut based cards
    SR_card_pos = makeCardFromHist(output, 'p_topW_pos', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True)
    SR_card_neg = makeCardFromHist(output, 'p_topW_neg', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True)
    
    card = dataCard()
    
    SR_NN_card = card.combineCards({'pos': SR_NN_card_pos, 'neg':SR_NN_card_neg})
    results_NN = card.nllScan(SR_NN_card, rmin=0, rmax=3, npoints=61, options=' -v -1')
    
    SR_card = card.combineCards({'pos': SR_card_pos, 'neg':SR_card_neg})
    results = card.nllScan(SR_card, rmin=0, rmax=3, npoints=61, options=' -v -1')

    card.cleanUp()
    
    print ("NN: NLL for r=0: %.2f"%(results_NN[results_NN['r']==0]['deltaNLL']*2)[0])
    print ("cut: NLL for r=0: %.2f"%(results[results['r']==0]['deltaNLL']*2)[0])

    print ("Checking for overtraining in max node asignment...")

    ks = test_train_cat(
        pred_test,
        pred_train,
        y_test_int,
        y_train_int,
        labels=['top-W', 'ttW', 'ttZ', 'ttH', 'nonprompt'],
        n_cat=5,
        plot_dir=plot_dir,
        weight_test = df_test['weight'].values,
        weight_train = df_train['weight'].values,
    )

    for label in ks:
        if ks[label][0]<0.05:
            print ("- !! Found small p-value for process %s: %.2f"%(label, ks[label][0]))

    print ("Checking for overtraining in the different nodes...")

    for node in [0,1,2,3,4]:
        ks = test_train(
            pred_test,
            pred_train,
            y_test_int,
            y_train_int,
            labels=['top-W', 'ttW', 'ttZ', 'ttH', 'nonprompt'],
            node=node,
            bins=bins,
            plot_dir=plot_dir,
            weight_test = df_test['weight'].values,
            weight_train = df_train['weight'].values,
        )
        for label in ks:
            if ks[label][0]<0.05:
                print ("- !! Found small p-value for process %s in node %s: %.2f"%(label, node, ks[label][0]))

