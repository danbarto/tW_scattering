import os
import numpy as np

import onnxruntime as rt
from keras.utils import np_utils

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

import matplotlib.pyplot as plt

def get_one_hot(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    return np_utils.to_categorical(labels)

def load_onnx_model(version='v8'):
    model = rt.InferenceSession(os.path.expandvars('$TWHOME/ML/networks/weights_%s.onnx'%version))
    scaler = joblib.load(os.path.expandvars('$TWHOME/ML/networks/scaler_%s.joblib'%version))
    return model, scaler

def dump_onnx_model(model, version='v8'):
    '''
    Takes keras model, dumps an onnx version
    '''
    import tf2onnx.convert
    import onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    onnx.save(onnx_model, os.path.expandvars('$TWHOME/ML/networks/weights_%s.onnx'%version))
    return True

def predict_onnx(model, data):
    input_name = model.get_inputs()[0].name
    pred_onx = model.run(None, {input_name: data.astype(np.float32)})[0]
    return pred_onx

def store_model(model, scaler, version='v5'):
    model.save('networks/weights_%s.h5a'%version)
    joblib.dump(scaler, 'networks/scaler_%s.joblib'%version)

def store_transformer(transformer, version='v5'):
    joblib.dump(transformer, 'networks/transformer_%s.joblib'%version)

def load_model(version='v5'):
    model = tf.keras.models.load_model(os.path.expandvars('$TWHOME/ML/networks/weights_%s.h5a'%version))
    scaler = joblib.load(os.path.expandvars('$TWHOME/ML/networks/scaler_%s.joblib'%version))
    return model, scaler

def load_transformer(version='v5'):
    return joblib.load(os.path.expandvars('$TWHOME/ML/networks/transformer_%s.joblib'%version))

def get_class_weight(df, dim=6):
    '''
    balance the total yield of the different classes.
    '''
    return {i: 1/sum(df[df['label']==i]['weight']) for i in range(dim)}

def get_sob(sig, bkg, var='fwd_jet_p', start=500, step=1, threshold=9):
    for i in range(int(start/step), 10000, 1):
        s = sum(sig[sig[var]>(i*step)]['weight'])*137
        b = sum(bkg[bkg[var]>(i*step)]['weight'])*137
        if s<threshold: break
    print (s, b, i*step)
    return s/b



def get_correlation_matrix(df, f_out='./correlation.pdf'):
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    
    im = ax.matshow(df.corr())
    ax.set_xticks(range(df.select_dtypes(['number']).shape[1]))
    ax.set_xticklabels(df.select_dtypes(['number']).columns, rotation=90, fontdict={'fontsize':12})
    ax.set_yticks(range(df.select_dtypes(['number']).shape[1]))
    ax.set_yticklabels(df.select_dtypes(['number']).columns, fontdict={'fontsize':12})
    cbar = ax.figure.colorbar(im)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title('Correlation Matrix', fontsize=16)

    fig.savefig(f_out)

def get_confusion_matrix(y_true, y_pred, f_out='./confusion'):
    f_out = f_out.strip('.pdf')
    f_out = f_out.strip('.png')
    CM = confusion_matrix(y_true, y_pred, normalize='true')
    # normalize='true' essentially does the same thing as
    # CM / (np.ones_like(CM) * np.sum(CM, axis=1)[:, np.newaxis])
    # which of course is ugly.
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    im = ax.matshow(CM)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')  # loc doesn't move the label to a different axis
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            c = CM[j,i]
            ax.text(i, j, "%.2f"%c, va='center', ha='center')
    cbar = ax.figure.colorbar(im)
    cbar.ax.tick_params(labelsize=12)

    for ext in ['.png', '.pdf']:
        fig.savefig(f_out+ext)


def simple_accuracy(pred, dummy_y):
    pred_Y = pred.argmax(axis=1)
    true_Y = dummy_y.argmax(axis=1)
    
    # this gives the measure of correctly tagged events, over the total
    return sum(pred_Y == true_Y)/len(true_Y)

