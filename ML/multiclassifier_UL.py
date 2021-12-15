import os
import warnings
warnings.filterwarnings('ignore')

# data handling and numerical analysis
import uproot
import awkward as ak
import numpy as np
import pandas as pd
from coffea import processor, hist

import scipy

# Plotting / histogramming
from yahist import Hist1D
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

# Machine learning packages
import tensorflow as tf
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils
import onnxruntime as rt
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

# private modules and functions
from Tools.dataCard import dataCard
from Tools.helpers import finalizePlotDir, mt
from Tools.limits import makeCardFromHist
from processor.default_accumulators import dataset_axis
from plots.helpers import makePlot
from ML.multiclassifier_tools import get_one_hot, get_class_weight, get_sob,\
            load_onnx_model, predict_onnx, dump_onnx_model,\
            store_model, load_model,\
            store_transformer, load_transformer
from ML.models import baseline_model, create_model


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
            train.argmax(axis=1)[(y_train==i)],
            test.argmax(axis=1)[(y_test==i)]
        )

        ks[label] = (_p, _ks)
        
        h[label+'_test'] = Hist1D(test.argmax(axis=1)[(y_test==i)], bins=bins, weights=weight_test[(y_test==i)]).normalize()
        h[label+'_train'] = Hist1D(train.argmax(axis=1)[(y_train==i)], bins=bins, label=label+' (p=%.2f, KS=%.2f) train'%(_p, _ks), weights=weight_train[(y_train==i)]).normalize()

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

def get_cat_plot(X, y, labels=[], n_cat=5, plot_dir=None, weight=None):
    ks = {}
    bins = [x-0.5 for x in range(n_cat+1)]
    
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    
    h = {}
    for i, label in enumerate(labels):
        
        h[label+'_train'] = Hist1D(X.argmax(axis=1)[(y==i)], bins=bins, label=label, weights=weight[(y==i)])
        
        h[label+'_train'].plot(color=colors[i], histtype="step", linewidth=2)
        
    ax.set_ylabel('a.u.')
    ax.set_xlabel('category')

    ax.set_ylim(0,200)

    if plot_dir:
        finalizePlotDir(plot_dir)
        fig.savefig("{}/abs_categories.png".format(plot_dir))
        fig.savefig("{}/abs_categories.pdf".format(plot_dir))


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


if __name__ == '__main__':


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--load', action='store_true', default=None, help="Load weights?")
    argParser.add_argument('--cat', action='store_true', default=None, help="Use categories?")
    argParser.add_argument('--fit', action='store_true', default=None, help="Do combine fit?")
    argParser.add_argument('--version', action='store', default='v21', help="Version number")
    argParser.add_argument('--year', action='store', default='2018', help="Which year?")
    argParser.add_argument('--seed', action='store', default=-1, help="Fix the random seed in numpy?")
    argParser.add_argument('--optimize', action='store_true', help="Run grid search for hyper parameters?")
    # NOTE: need to add the full Run2 training option back
    args = argParser.parse_args()


    load_weights = args.load
    version = "_".join([args.year, args.version])
    is_cat = args.cat

    if args.seed>0:
        np.random.seed(args.seed)

    plot_dir = os.path.expandvars("/home/users/$USER/public_html/tW_scattering/ML/%s/"%(version))

    # Load the input data.
    # This data frame is produced with the SS_analysis processor:
    # ipython -i SS_analysis.py -- --dump
    #df = pd.read_hdf('/hadoop/cms/store/user/dspitzba/ML/multiclass_input_2018_v2.h5')
    df = pd.read_hdf('../processor/multiclass_input_%s_v2.h5'%args.year)

    variables = [
        ## best results with all variables, but should get pruned at some point...
        'n_jet',
        ##'n_central',
        'n_fwd',
        'n_b',
        'n_tau', ## important for ttZ
        #'n_track', ## not so important, and very bad data/MC agreement
        'st',
        ##'ht',
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

    baseline = (df['n_fwd']>=0)
    #baseline = (df['n_fwd']>0)

    df['label_orig'] = df['label']

    # Take input dataframe, rearrange the categories, calculate the correct weights, and relabel
    # Signal is easy. Just take events that are in the SS category: df['SS']==1, passing the baseline selection
    # Asigned label: 0
    df_signal       = df[((df['label']==0)&(df['SS']==1)&baseline)]
    df_signal['label'] = np.ones(len(df_signal))*0

    # Prompt backgrounds from the various processes that contribute.
    # We only take events from the SS category that are not labeled as containing a lost lepton (LL).
    # Asigned label: 1
    df_prompt       = df[((df['label']<7)&(df['label']>0)&(df['label']!=4)&(df['LL']==0)&(df['SS'])&baseline)] # every prompt background except ttbar (which shouldn't have prompt anyway)
    df_prompt['label'] = np.ones(len(df_prompt))*1

    # Lost lepton backgrounds from the various processes that contribute.
    # Very similar to above, but now requiring a lost lepton
    # Asigned label: 2
    df_LL           = df[((df['label']<7)&(df['label']>0)&(df['label']!=4)&(df['LL']>0)&(df['SS'])&baseline)]
    df_LL['label']  = np.ones(len(df_LL))*2

    # Nonprompt leptons, taken from top quark process (input label 4).
    # We use the data driven background estimate, so we use events from the AR region, and adjust the weight accordingly
    # Asigned label: 3
    df_NP           = df[((df['label']==4)&(df['AR']==1)&baseline)]
    df_NP['weight'] = df_NP['weight']*df_NP['weight_np']
    df_NP['label']  = np.ones(len(df_NP))*3

    # Charge flip category. We don't use all the events to train because they are just too many
    # We adjust the weight according to our background prediction, the rescaling, and only take events from the OS category: df['OS']==1
    # Asigned label: 4
    rescaler        = len(df[((df['label']==4)&(df['OS']==1)&baseline)])/100000
    df_CF           = resample(df[((df['label']==4)&(df['OS']==1)&baseline)], n_samples=100000)
    df_CF['weight'] = df_CF['weight']*df_CF['weight_cf']*rescaler
    df_CF['label']  = np.ones(len(df_CF))*4

    # These data frames are currently not used in training, but just for visualizations
    df_TTW          = df[(((df['label']==1)|(df['label']==3)|(df['label']==5))&(df['SS']==1)&baseline)]  # assume that rares (4-top, VVV) is prompt, too
    df_TTW['label'] = np.ones(len(df_TTW))
    df_TTZ          = df[((df['label']==2)&(df['SS']==1)&baseline)]
    df_TTH          = df[((df['label']==3)&(df['SS']==1)&baseline)]

    print ()
    print ("Yields after training preselection (baseline selection):")
    df_list = [\
        ('signal',      df_signal),
        ('TTW/TTH',     df_TTW),
        ('TTZ',         df_TTZ),
        ('TTH',         df_TTH),
        ('prompt',      df_prompt),
        ('LL',          df_LL),
        ('nonprompt',   df_NP),
        ('charge flip', df_CF),
    ]

    print ("{:30}{:>10}{:>10}".format("Name", "Weighted", "Raw"))
    for name, df in df_list:
        print ("{:30}{:10.2f}{:10}".format(name, sum(df['weight']), len(df)))

    print ()

    # Now, merge all the separate dataframes into one again
    df_in = pd.concat([df_signal, df_prompt, df_LL, df_NP, df_CF])
    labels = df_in['label'].values
    df_train, df_test, y_train_int, y_test_int = train_test_split(df_in, labels, train_size= int( 0.9*labels.shape[0] ), random_state=42 )

    X_train = df_train[variables].values
    X_test  = df_test[variables].values

    y_train = get_one_hot(y_train_int.astype(int))
    y_test = get_one_hot(y_test_int.astype(int))
    

    input_dim = len(variables)
    out_dim = len(y_train[0])

    # Adjust the weights of every category so that they have equal importance
    # FIXME: We can try to increase e.g. the importance of prompt or lost lepton backgrounds vs others by multiplying
    # their weight by a constant factor. get_class_weight is imported from ML.multiclassifier_tools
    class_weight = get_class_weight(df_train, dim=out_dim)

    '''
    # Can't use pipelines, unfortunately
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('NN', baseline_model()),
    ])
    '''

    if not load_weights:

        epochs = 10 #100  # 50 -> 200
        batch_size = 100 #5120
        validation_split = 0.2

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        params = scaler.get_params()

        if args.optimize:
            model = KerasClassifier(build_fn=create_model,
                                    input_dim=input_dim, out_dim=out_dim,
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    verbose = 0,
                                    #class_weight = class_weight,
                                    #sample_weight = np.abs(df_train['weight'].values),  # this doesn't work here.
                                    #dropout_rate=0.1,
                                    #neurons=150,
                                    #learning_rate=0.001,
                                    #activation='relu',
                                    #optimizer='RMSprop',  # this doesn't do anything atm?
                                    )
            print ("Input dim: %s"%input_dim)
            # define the grid search parameters
            neurons = [1, 10, 20, 50, 75, 100, 150]
            #param_grid = dict(neurons=neurons)
            # Best: 0.499523 using {'neurons': 100}
            dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            #param_grid = dict(dropout_rate=dropout_rate)
            # Best: 0.506186 using {'dropout_rate': 0.5}
            activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            #param_grid = dict(activation=activation)
            # Best: 0.648132 using {'activation': 'relu'} NOTE: this was without sample_weight
            # Best: 0.498214 using {'activation': 'relu'} NOTE: this was with sample_weight
            batch_size = [100, 1000]
            epochs = [10, 100]
            param_grid = dict(batch_size=batch_size, epochs=epochs)
            # Best: 0.516183 using {'batch_size': 100, 'epochs': 10}
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=5, cv=3)
            grid_result = grid.fit(
                X_train_scaled,
                y_train,
                class_weight = class_weight,
                sample_weight = np.abs(df_train['weight'].values),
            )
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            model = create_model(input_dim=input_dim, out_dim=out_dim,
                                 dropout_rate=0.5,
                                 neurons=100,  # FIXME
                                 learning_rate=0.001,
                                 activation='relu',
                                 optimizer='RMSprop',
                                 )
        else:
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

        plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)

    else:
        #model, scaler = load_model(version=version)
        model, scaler = load_onnx_model(version=version)

        X_train_scaled = scaler.transform(X_train)
        print ("Loaded weights.")

    X_all = df_in[variables].values

    X_all_scaled  = scaler.transform(X_all)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate the model for the entire data frame (pred_all), just the training set (pred_train) or the test set (pred_test)
    if not load_weights:
        pred_all    = model.predict( X_all_scaled )
        pred_train  = model.predict( X_train_scaled )
        pred_test   = model.predict( X_test_scaled )
    else:
        # always use ONNX for inference
        pred_all    = predict_onnx(model, X_all_scaled )
        pred_train  = predict_onnx(model, X_train_scaled )
        pred_test   = predict_onnx(model, X_test_scaled )

    

    # We can now evaluate the performance
    df_in['score_topW'] = pred_all[:,0]
    df_in['score_prompt'] = pred_all[:,1]
    df_in['score_ll'] = pred_all[:,2]
    df_in['score_np'] = pred_all[:,3]
    df_in['score_cf'] = pred_all[:,4]
    df_in['score_best'] = pred_all.argmax(axis=1)

    # For quantile transformation on the top-W node:
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=40, random_state=0)
    qt.fit(df_in[((df_in['label']==0)&(df_in['score_best']==0))]['score_topW'].values.reshape(-1, 1))
    
    store_transformer(qt, version=version)

    df_in['score_topW_transform'] = qt.transform(df_in['score_topW'].values.reshape(-1, 1))
    df_in['score_prompt_transform'] = qt.transform(df_in['score_prompt'].values.reshape(-1, 1))

    for i in range(3):
        print ("Checking assignment for cat %s"%i)
        for x in range(5):
            print (x, round(sum(df_in[((df_in['SS']==1)&(df_in['label']==i)&(df_in['score_best']==x))]['weight'])/sum(df_in[((df_in['SS']==1)&(df_in['label']==i))]['weight']), 3))

    print ("NP assignment")
    for x in range(5):
        print (x, round(sum(df_in[((df_in['AR']==1)&(df_in['label']==3)&(df_in['score_best']==x))]['weight'])/sum(df_in[((df_in['AR']==1)&(df_in['label']==3))]['weight']), 3))

    print ("CF assignment")
    for x in range(5):
        print (x, round(sum(df_in[((df_in['OS']==1)&(df_in['label']==4)&(df_in['score_best']==x))]['weight'])/sum(df_in[((df_in['OS']==1)&(df_in['label']==4))]['weight']), 3))

    def get_bkg(x):
        bkg = sum(df_in[((df_in['SS']==1)&(df_in['label']==1)&(df_in['score_best']==0)&(df_in['score_topW']>x))]['weight']) + \
                sum(df_in[((df_in['SS']==1)&(df_in['label']==2)&(df_in['score_best']==0)&(df_in['score_topW']>x))]['weight']) + \
                sum(df_in[((df_in['AR']==1)&(df_in['label']==3)&(df_in['score_best']==0)&(df_in['score_topW']>x))]['weight']) + \
                sum(df_in[((df_in['OS']==1)&(df_in['label']==4)&(df_in['score_best']==0)&(df_in['score_topW']>x))]['weight'])
        return bkg

    def get_sig(x):
        return sum(df_in[((df_in['SS']==1)&(df_in['label']==0)&(df_in['score_best']==0)&(df_in['score_topW']>x))]['weight'])

    print ("Signal yield in node 0: %.2f"%get_sig(0))
    print ("Baseline S/b: %.3f"%(get_sig(0)/get_bkg(0)))

    # find the cut where there are only 9 bkg events left (arbitrary threshold)
    for i in range(0, 500, 1):
        bkg = get_bkg(i/500)
        if bkg < 9:
            break
    thresh = i/500
    
    print ("S/B for bkg=9: %.3f"%(get_sig(thresh)/get_bkg(thresh)))

    for i in range(0, 500, 1):
        sig = get_sig(i/500)
        if sig < 1:
            break
    thresh = i/500
    
    print ("S/B for sig=1: %.3f"%(get_sig(thresh)/get_bkg(thresh)))


    print ("Checking for overtraining in max node asignment...")

    colors = ['gray', 'blue', 'red', 'green', 'orange']
    hist_labels = ['top-W', 'prompt', 'LL', 'NP', 'CF']

    ks = test_train_cat(
        pred_test,
        pred_train,
        y_test_int,
        y_train_int,
        labels = hist_labels,
        n_cat = len(hist_labels),
        plot_dir = plot_dir,
        weight_test = df_test['weight'].values,
        weight_train = df_train['weight'].values,
    )

    for label in ks:
        if ks[label][0]<0.05:
            print ("- !! Found small p-value for process %s: %.2f"%(label, ks[label][0]))


    get_cat_plot(
        pred_all,
        df_in['label'].values,
        labels = hist_labels,
        n_cat = len(hist_labels),
        plot_dir = plot_dir,
        weight = df_in['weight'].values,
    )

    print ("Checking for overtraining in the different nodes...")

    bins = [x/20 for x in range(21)]

    for node in [0,1,2,3,4]:
        ks = test_train(
            pred_test,
            pred_train,
            y_test_int,
            y_train_int,
            labels=hist_labels,
            node=node,
            bins=bins,
            plot_dir=plot_dir,
            weight_test = df_test['weight'].values,
            weight_train = df_train['weight'].values,
        )
        for label in ks:
            if ks[label][0]<0.05:
                print ("- !! Found small p-value for process %s in node %s: %.2f"%(label, node, ks[label][0]))


    if not load_weights:
        dump_onnx_model(model, version=version)

    # Correlations
    from ML.multiclassifier_tools import get_correlation_matrix, get_confusion_matrix
    get_correlation_matrix(
        df_in[(df_in['label']==0)][(variables+['score_topW', 'score_prompt', 'score_ll', 'score_np', 'score_cf'])], 
        f_out=plot_dir+'/correlation.png'
    )
    

    # make this a function for two histograms with ratio, for a certain binning

    def shape_comparison(array1, array2, bins=20, weight1=[], weight2=[], normalize=True, labels=["first", "second"], save=False):

        import mplhep as hep
        plt.style.use(hep.style.CMS)

        hist1 = Hist1D(array1, bins=np.arange(0,bins+1)/bins, weights=weight1)
        hist2 = Hist1D(array2, bins=np.arange(0,bins+1)/bins, weights=weight2)

        if normalize:
            hist1 = hist1.normalize()
            hist2 = hist1.normalize()

        ratio = hist1.divide(hist2)
        
        fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)

        hep.cms.label(
            "Preliminary",
            data=False,
            lumi=60.0,
            loc=0,
            ax=ax,
        )
        
        hep.histplot(
            [ hist1.counts, hist2.counts ],
            hist1.edges,
            w2=[ hist1.errors**2, hist2.errors**2 ],
            histtype="step",
            stack=False,
            label=labels,
            ax=ax)

        hep.histplot(
            ratio.counts,
            ratio.edges,
            w2=ratio.errors,
            histtype="errorbar",
            color='black',
            ax=rax)

        rax.set_ylim(0,1.99)
        rax.set_xlabel(r'$score$')
        rax.set_ylabel(r'Ratio')
        ax.set_ylabel(r'Events')
        
        #add_uncertainty(total_mc, rax, ratio=True)
        #add_uncertainty(total_mc, ax)
        
        ax.legend()
        
        plt.show()
        
        if save:
            fig.savefig("{}.png".format(save))
            fig.savefig("{}.pdf".format(save))

    shape_comparison(
        df_in[((df_in['label']==0)&(df_in['score_best']==0))]['score_topW_transform'].values,
        df_in[((df_in['label']==1)&(df_in['score_best']==0))]['score_topW_transform'].values,
        bins=8,
        weight1=df_in[((df_in['label']==0)&(df_in['score_best']==0))]['weight'].values,
        weight2=df_in[((df_in['label']==1)&(df_in['score_best']==0))]['weight'].values,
        normalize = False,
        save = "{}/score_transformed".format(plot_dir),
    )

    ## Confusion matrix
    get_confusion_matrix(
        y_train_int,
        np.argmax(pred_train, axis=1),
        f_out=plot_dir+'/confusion_train.pdf',
    )
    get_confusion_matrix(
        y_test_int,
        np.argmax(pred_test, axis=1),
        f_out=plot_dir+'/confusion_test.pdf',
    )
    # normalize='true' essentially does the same thing as
    # CM_test / (np.ones_like(CM_test) * np.sum(CM_test, axis=1)[:, np.newaxis])
    # which of course is ugly.
