import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import copy

import scipy
from yahist import Hist1D

import onnxruntime as rt

import tensorflow as tf
from keras.utils import np_utils

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

from Tools.dataCard import dataCard
from Tools.helpers import finalizePlotDir, mt

import warnings
warnings.filterwarnings('ignore')

from memory_profiler import profile

@profile
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


@profile
def load_model(version='v5'):

    model = tf.keras.models.load_model(os.path.expandvars('$TWHOME/ML/networks/weights_%s.h5a'%version))
    scaler = joblib.load(os.path.expandvars('$TWHOME/ML/networks/scaler_%s.joblib'%version))
    return model, scaler

@profile
def predict(model, data):
    return model.predict(data)

@profile
def scale(scaler, data):
    return scaler.transform(data)


@profile
def load_onnx_model(version='v8'):
    model = rt.InferenceSession("networks/weights_v8.onnx")
    scaler = joblib.load(os.path.expandvars('$TWHOME/ML/networks/scaler_%s.joblib'%version))
    return model, scaler

@profile
def predict_onnx(model, data):
    input_name = model.get_inputs()[0].name
    pred_onx = model.run(None, {input_name: data.astype(np.float32)})[0]
    return pred_onx

if __name__ == '__main__':

    load_weights = True
    version = 'v8'

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
    #df_train, df_test, y_train_int, y_test_int  = prepare_data('data/multiclass_input_split.h5', preselection, reuse=True)
    


    model, scaler = load_model(version=version)

    print ("Loaded weights.")

    X_all = df[variables].values

    #X_all_scaled  = scaler.transform(X_all)
    X_all_scaled  = scale(scaler, X_all)

    #pred_all    = model.predict( X_all_scaled )
    pred_all    = predict(model, X_all_scaled)
    
    df['score_topW'] = pred_all[:,0]
    df['score_ttW'] = pred_all[:,1]
    df['score_ttZ'] = pred_all[:,2]
    df['score_ttH'] = pred_all[:,3]
    df['score_ttbar'] = pred_all[:,4]
    df['score_best'] = np.argmax(pred_all, axis=1)
    
