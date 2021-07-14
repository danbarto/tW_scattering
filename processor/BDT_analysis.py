import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd
from yahist import Hist1D, Hist2D

# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *
from Tools.helpers import mt
from Tools.fake_rate import fake_rate
from Tools.SS_selection import SS_selection
import production.weights

import uproot
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import time

def load_category(category, baby_dir="/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/2018/dilep/"):
    file_name = baby_dir + "{}.root".format(category)
    try:
        tree = uproot.open(file_name)['Events']
    except:
        tree = uproot.open(file_name)['T']
    process_name = file_name[(file_name.rfind('/')+1):(file_name.rfind('.'))]
    tmp_df = pd.DataFrame()
    df_values = tree.arrays()
    tmp_df["Most_Forward_pt"] = np.array(df_values["Most_Forward_pt"])
    tmp_df["HT"] = np.array(df_values["HT"])
    tmp_df["LeadLep_eta"] = np.array(df_values["LeadLep_eta"])
    tmp_df["LeadLep_pt"] = np.array(df_values["LeadLep_pt"])
    tmp_df["LeadLep_dxy"] = np.array(df_values["LeadLep_dxy"])
    tmp_df["LeadLep_dz"] = np.array(df_values["LeadLep_dz"])
    tmp_df["SubLeadLep_pt"] = np.array(df_values["SubLeadLep_pt"])
    tmp_df["SubLeadLep_eta"] = np.array(df_values["SubLeadLep_eta"])
    tmp_df["SubLeadLep_dxy"] = np.array(df_values["SubLeadLep_dxy"])
    tmp_df["SubLeadLep_dz"] = np.array(df_values["SubLeadLep_dz"])
    tmp_df["nJet"] = np.array(df_values["nJets"])
    tmp_df["nbtag"] = np.array(df_values["nBtag"])
    tmp_df["LeadJet_pt"] = np.array(df_values["LeadJet_pt"])
    tmp_df["SubLeadJet_pt"] = np.array(df_values["SubLeadJet_pt"])
    tmp_df["SubSubLeadJet_pt"] = np.array(df_values["SubSubLeadJet_pt"])
    tmp_df["nElectron"] = np.array(df_values["nElectron"])
    tmp_df["MET_pt"] = np.array(df_values["MET_pt"])
    tmp_df["LeadBtag_pt"] = np.array(df_values["LeadBtag_pt"])
    tmp_df["MT_LeadLep_MET"] = np.array(df_values["MT_LeadLep_MET"])
    tmp_df["MT_SubLeadLep_MET"] = np.array(df_values["MT_SubLeadLep_MET"])
    tmp_df["LeadLep_SubLeadLep_Mass"] = np.array(df_values["LeadLep_SubLeadLep_Mass"])
    tmp_df["SubSubLeadLep_pt"] = np.array(df_values["SubSubLeadLep_pt"])
    tmp_df["SubSubLeadLep_eta"] = np.array(df_values["SubSubLeadLep_eta"])
    tmp_df["SubSubLeadLep_dxy"] = np.array(df_values["SubSubLeadLep_dxy"])
    tmp_df["SubSubLeadLep_dz"] = np.array(df_values["SubSubLeadLep_dz"])
    tmp_df["MT_SubSubLeadLep_MET"] = np.array(df_values["MT_SubSubLeadLep_MET"])
    tmp_df["LeadBtag_score"] = np.array(df_values["LeadBtag_score"])
    tmp_df["weight"] = np.array(df_values["Weight"])
    if "signal" in process_name:
        tmp_df["Label"] = "s"
    else:
        tmp_df["Label"] = "b"
    return tmp_df

def BDT_train_test_split(full_data, verbose=True):
    if verbose:
        print('Size of data: {}'.format(full_data.shape))
        print('Number of events: {}'.format(full_data.shape[0]))
        print('Number of columns: {}'.format(full_data.shape[1]))

        print ('\nList of features in dataset:')
        for col in full_data.columns:
            print(col)
        # look at column labels --- notice last one is "Label" and first is "EventId" also "Weight"
        print('Number of signal events: {}'.format(len(full_data[full_data.Label == 's'])))
        print('Number of background events: {}'.format(len(full_data[full_data.Label == 'b'])))
        print('Fraction signal: {}'.format(len(full_data[full_data.Label == 's'])/(float)(len(full_data[full_data.Label == 's']) + len(full_data[full_data.Label == 'b']))))

    full_data['Label'] = full_data.Label.astype('category')
    (data_train, data_test) = train_test_split(full_data, train_size=0.5)
    if verbose:
        print('Number of training samples: {}'.format(len(data_train)))
        print('Number of testing samples: {}'.format(len(data_test)))

        print('\nNumber of signal events in training set: {}'.format(len(data_train[data_train.Label == 's'])))
        print('Number of background events in training set: {}'.format(len(data_train[data_train.Label == 'b'])))
        print('Fraction signal: {}'.format(len(data_train[data_train.Label == 's'])/(float)(len(data_train[data_train.Label == 's']) + len(data_train[data_train.Label == 'b']))))
        
    return data_train, data_test

def gen_BDT(signal_name, data_train, data_test, param, num_trees, output_dir, booster_name="", flag_load=False, verbose=True):
    feature_names = data_train.columns[1:-2]  #full_data
    train_weights = data_train.weight
    test_weights = data_test.weight
    # we skip the first and last two columns because they are the ID, weight, and label
    train = xgb.DMatrix(data=data_train[feature_names],label=data_train.Label.cat.codes,
                        missing=-999.0,feature_names=feature_names, weight=np.abs(train_weights))
    test = xgb.DMatrix(data=data_test[feature_names],label=data_test.Label.cat.codes,
                       missing=-999.0,feature_names=feature_names, weight=np.abs(test_weights))
    evals_result = {}
    if booster_name == "":
        booster_path = output_dir + "booster_{}.model".format(signal_name)
    else:
        booster_path = output_dir + "booster_{}.model".format(booster_name)
    #breakpoint()
    if verbose:
        print(feature_names)
        print(data_test.Label.cat.codes)
        print("weights:\n")
        print(train_weights)
    if flag_load:
        print("Loading saved model...")
        booster = xgb.Booster({"nthread": 4})  # init model
        booster.load_model(booster_path)  # load data

    else:
        print("Training new model...")
        evals = [(train, "train"), (test,"test")]
        booster = xgb.train(param,train,num_boost_round=num_trees, evals=evals, evals_result=evals_result)
        print(booster.eval(test))

    #if the tree is of interest, we can save it
    if not flag_load:
        booster.save_model(booster_path)

    return booster, train, test, evals_result

def optimize_BDT_params(data_train, n_iter=20, num_folds=3, param_grid={}):
    y_train = data_train.Label.cat.codes
    feature_names = data_train.columns[1:-2] 
    x_train = data_train[feature_names]
    clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric="logloss", use_label_encoder=False)
    if len(param_grid.keys())==0:
        param_grid = {
                      'max_depth': [3, 4, 5, 6],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
                      'subsample': [0.7, 0.8, 0.9, 1.0],
                      'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                      'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                      'gamma': [0, 0.25, 0.5, 1.0],
                      'reg_lambda': [0.1, 1.0, 5.0, 10.0],
                      'n_estimators': [20, 40, 50, 60, 100, 150, 200]
                     }
    fit_params = {'eval_metric': 'mlogloss',
                  'early_stopping_rounds': 10}
    rs_clf = RandomizedSearchCV(clf_xgb, param_grid, n_iter=n_iter,
                                n_jobs=1, verbose=2, cv=num_folds,
                                refit=False)
    print("Randomized search..")
    search_time_start = time.time()
    rs_clf.fit(x_train, y_train)
    print("Randomized search time:", time.time() - search_time_start)

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    return best_params

class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}, debug=False, BDT_params=[], version= "fcnc_v6_SRonly_5may2021", SS_region="SS", ):
        self.variations = variations
        self.year = year
        self.debug = debug
        self.btagSF = btag_scalefactor(year)
        self._accumulator = processor.dict_accumulator(accumulator)
        self.BDT_params = BDT_params
        self.version=version
        self.SS_region=SS_region

    @property
    def accumulator(self):
        return self._accumulator  
  
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        events = events[ak.num(events.Jet)>0] #corrects for rare case where there isn't a single jet 
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=2
        
        ev = events[presel]
        ##Jets
        Jets = events.Jet
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi
    
         ### For FCNC, we want electron -> tightTTH
        ele_t = Collections(ev, "Electron", "tightFCNC", year=self.year).get()
        ele_l = Collections(ev, "Electron", "fakeableFCNC", year=self.year).get()    
        mu_t  = Collections(ev, "Muon", "tightFCNC", year=self.year).get()
        mu_l  = Collections(ev, "Muon", "fakeableFCNC", year=self.year).get()
        
        dataset = ev.metadata["dataset"]
        rares_dataset = ['ttw', 'www', 'wzz', 'wz', 'tg', 'wzg', 'tth_nobb', 'ttg_dilep', 'tttj', 'tttt', 'tttw', 
                         'tthh', 'vh_nobb', 'tzq', 'ttwh', 'ttg_1lep', 'wwz', 'wwg', 'ttwz', 'zz', 'ttww', 'wg', 
                         'ttz_m10', 'qqww', 'zzz', 'ggh', 'ttzh', 'ttz_m1-10', 'tw_dilep', 'ttzz']
        
        flips_dataset = ["dyjets_m10-50", "dyjets_m50", "zg", "tw_dilep"]#, "ww"]
        if dataset in rares_dataset:
            ele_t = ele_t[((ele_t.genPartFlav==1)|(ele_t.genPartFlav==15))]
            ele_l = ele_l[((ele_l.genPartFlav==1)|(ele_l.genPartFlav==15))]
            mu_t = mu_t[((mu_t.genPartFlav==1)|(mu_t.genPartFlav==15))]
            mu_l = mu_l[((mu_l.genPartFlav==1)|(mu_l.genPartFlav==15))]
        
        #SS preselection 
        lepton  = ak.concatenate([mu_l, ele_l], axis=1)
        tight_lepton  = ak.concatenate([mu_t, ele_t], axis=1)
        sorted_index_nofilter = ak.argsort(tight_lepton.pt, axis=-1, ascending=False)
        sorted_lep_nofilter = tight_lepton[sorted_index_nofilter]
        leadlep_nofilter = sorted_lep_nofilter[:,0:1]
        subleadlep_nofilter = sorted_lep_nofilter[:,1:2]
        
        #M(ee) > 12
        diele_mass = choose(ele_t, 2).mass

        #clean jets :
        # we want at least two jets that are outside of the lepton jets by deltaR > 0.4
        jets = getJets(ev, maxEta=2.4, minPt=40, pt_var='pt')
        jets_for_btag = getJets(ev, maxEta=2.5, minPt=25, pt_var='pt')
        #jet_sel = (ak.num(jets[~(match(jets, ele_l, deltaRCut=0.4) | match(jets, mu_l, deltaRCut=0.4))])>=2)
        btag = getBTagsDeepFlavB(jets_for_btag, year=self.year)
        
        selection = PackedSelection()
        selection.add("njets", (ak.num(jets[~(match(jets, tight_lepton, deltaRCut=0.4))]) >= 2))
        selection.add("nlep", (ak.num(lepton, axis=1) == 2))
        selection.add("nlep_tight", (ak.num(tight_lepton, axis=1) == 2))
        selection.add("SS", (ak.sum(ak.concatenate([leadlep_nofilter.charge, subleadlep_nofilter.charge], axis=1), axis=1) != 0))
        selection.add("nbtag", (ak.num(btag, axis=1) >= 0))
        selection.add("M(ee)>12", ((ak.num(ele_t) < 2) | (ak.sum(diele_mass, axis=1) > 12.0))) #ak.sum here to flatten the diele_mass array
        selection_reqs = ["njets", "nbtag", "nlep", "SS", "nlep_tight", "M(ee)>12"]
        
        if dataset in flips_dataset:
            flip_evts = ((ak.sum(ev.GenPart.pdgId[ele_t.genPartIdx]==((-1)*ele_t.pdgId), axis=1) == 1))
            selection.add("flip", flip_evts)
            selection_reqs += ["flip"]
        
        fcnc_reqs_d = { sel: True for sel in selection_reqs}
        FCNC_sel = selection.require(**fcnc_reqs_d)

        #sorting
        sorted_index = ak.argsort(lepton[FCNC_sel].pt, axis=-1, ascending=False)
        sorted_pt = lepton[FCNC_sel].pt[sorted_index]
        sorted_eta = lepton[FCNC_sel].eta[sorted_index]
        sorted_phi = lepton[FCNC_sel].phi[sorted_index]
        sorted_dxy = lepton[FCNC_sel].dxy[sorted_index]
        sorted_dz = lepton[FCNC_sel].dz[sorted_index]

        if (np.array(ak.num(jets[FCNC_sel])).any()==1): #if there is at least one event with a jet
            sorted_jet_index = ak.argsort(jets[FCNC_sel].pt, axis=-1, ascending=False)
            sorted_jet_pt = jets[FCNC_sel].pt[sorted_jet_index]
            #njets
            njets = ak.num(jets, axis=1)[FCNC_sel]
            most_forward_pt = ak.flatten(jets[FCNC_sel].pt[ak.singletons(ak.argmax(abs(jets[FCNC_sel].eta), axis=1))])
            leadjet_pt = ak.flatten(sorted_jet_pt[:,0:1])
            subleadjet_pt = ak.flatten(sorted_jet_pt[:,1:2])
            #this sometimes is not defined, so ak.firsts relpaces the empty arrays with None, then we can set all None to zero
            subsubleadjet_pt = ak.fill_none(ak.firsts(sorted_jet_pt[:,2:3]), 0)
        else: #if there are no events with jets
            njets = np.zeros_like(FCNC_sel[FCNC_sel])
            most_forward_pt = np.zeros_like(FCNC_sel[FCNC_sel])
            leadjet_pt = np.zeros_like(FCNC_sel[FCNC_sel])
            subleadjet_pt = np.zeros_like(FCNC_sel[FCNC_sel])
            subsubleadjet_pt = np.zeros_like(FCNC_sel[FCNC_sel])

        if (np.array(ak.num(btag[FCNC_sel])).any()==1): #if there is at least one event with a btag
            sorted_btag_index = ak.argsort(btag[FCNC_sel].pt, axis=-1, ascending=False)
            sorted_btag_pt = btag[FCNC_sel].pt[sorted_btag_index]
            #btags
            nbtag = ak.num(btag)[FCNC_sel]
            leadbtag_pt = sorted_btag_pt[:,0:1] #this sometimes is not defined (some of the arrays are empty)
            # ak.firsts() relpaces the empty arrays with None, then we can set all None to zero
            leadbtag_pt = ak.fill_none(ak.firsts(leadbtag_pt), 0)    
        else:
            nbtag = np.zeros_like(FCNC_sel[FCNC_sel])
            leadbtag_pt = np.zeros_like(FCNC_sel[FCNC_sel])
        
        leadlep_pt = ak.flatten(sorted_pt[:,0:1])
        subleadlep_pt = ak.flatten(sorted_pt[:,1:2])
        leadlep_eta = ak.flatten(sorted_eta[:,0:1])
        subleadlep_eta = ak.flatten(sorted_eta[:,1:2])
        leadlep_phi = ak.flatten(sorted_phi[:,0:1])
        subleadlep_phi = ak.flatten(sorted_phi[:,1:2])
        leadlep_dxy = ak.flatten(sorted_dxy[:,0:1])
        subleadlep_dxy = ak.flatten(sorted_dxy[:,1:2])    
        leadlep_dz = ak.flatten(sorted_dz[:,0:1])
        subleadlep_dz = ak.flatten(sorted_dz[:,1:2])
        
        sorted_lep = lepton[FCNC_sel][sorted_index]
        leadlep = sorted_lep[:,0:1]
        subleadlep = sorted_lep[:,1:2]
        nelectron = ak.num(ele_l[FCNC_sel], axis=1)
        MET_pt = ev[FCNC_sel].MET.pt
        MET_phi = ev[FCNC_sel].MET.phi
        #HT
        ht = ak.sum(jets.pt, axis=1)[FCNC_sel]
        
        if (np.array(ak.num(leadlep)).any()==1) and (np.array(ak.num(subleadlep)).any()==1):
            leadlep_subleadlep_mass = ak.flatten((leadlep + subleadlep).mass)
            #MT of lead and subleading lepton with ptmiss (MET)
            mt_leadlep_met = mt(leadlep_pt, leadlep_phi, MET_pt, MET_phi)
            mt_subleadlep_met = mt(subleadlep_pt, subleadlep_phi, MET_pt, MET_phi)
        else:
            leadlep_subleadlep_mass = np.zeros_like(FCNC_sel[FCNC_sel])
            mt_leadlep_met = np.zeros_like(FCNC_sel[FCNC_sel])
            mt_subleadlep_met = np.zeros_like(FCNC_sel[FCNC_sel])
        
        #get weights of events (scale1fb * generator_weights * lumi)
        weight = production.weights.get_weight(ev.metadata["dataset"], self.year, self.version) #scale1fb
        weight = weight * (ev.Generator.weight / abs(ev.Generator.weight)) #generator weights (can sometimes be negative)
        lumi_dict = {2018:59.71, 2017:41.5, 2016:35.9} #lumi weights
        weight = weight * lumi_dict[self.year]
        weight = weight[FCNC_sel]
        
        if len(FCNC_sel[FCNC_sel]) > 0:
            BDT_param_dict = {"Most_Forward_pt":most_forward_pt,
                              "HT":ht,
                              "LeadLep_eta":np.abs(leadlep_eta),
                              "MET_pt":MET_pt,
                              "LeadLep_pt":leadlep_pt,
                              "LeadLep_dxy":np.abs(leadlep_dxy),
                              "LeadLep_dz":np.abs(leadlep_dz),
                              "SubLeadLep_pt":subleadlep_pt,
                              "SubLeadLep_eta":np.abs(subleadlep_eta),
                              "SubLeadLep_dxy":np.abs(subleadlep_dxy),
                              "SubLeadLep_dz":np.abs(subleadlep_dz),
                              "nJet":njets,
                              "nbtag":nbtag,
                              "LeadJet_pt":leadjet_pt,
                              "SubLeadJet_pt":subleadjet_pt,
                              "SubSubLeadJet_pt":subsubleadjet_pt,
                              "nElectron":nelectron,
                              "MET_pt":MET_pt,
                              "LeadBtag_pt":leadbtag_pt,
                              "MT_LeadLep_MET":mt_leadlep_met,
                              "MT_SubLeadLep_MET":mt_leadlep_met,
                              "LeadLep_SubLeadLep_Mass":leadlep_subleadlep_mass,
                              "weight":weight
                              }
            #create pandas dataframe
            passed_events = ev[FCNC_sel]
            event_p = ak.to_pandas(passed_events[["event"]])
            for param in self.BDT_params:
                event_p[param] = BDT_param_dict[param]
            output['BDT_df'] += processor.column_accumulator(event_p.to_numpy())
        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.default_accumulators import desired_output, add_processes_to_output, dataset_axis, pt_axis, eta_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset

    overwrite = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'nano_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    fileset = make_fileset(['QCD'], samples, redirector=redirector_ucsd, small=True)
    
    meta = get_sample_meta(fileset, samples)

    add_processes_to_output(fileset, desired_output)
    
    desired_output.update({
        "single_mu_fakeable": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis),
        "single_mu": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis)
    })

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            nano_analysis(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()
