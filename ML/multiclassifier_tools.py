import os
import numpy as np

import onnxruntime as rt
from keras.utils import np_utils

from sklearn.metrics import roc_curve, roc_auc_score, auc
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
