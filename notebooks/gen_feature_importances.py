import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import awkward as ak
import pickle
from sklearn.preprocessing import quantile_transform, QuantileTransformer
import uproot
import glob
import os
from yahist import Hist1D
import processor.BDT_analysis as BDT_analysis
from coffea import processor, hist

import Tools.objects

import postProcessing.makeCards
import postProcessing.datacard_comparison.compare_datacards as compare_datacards
import uuid, os, uproot, shutil

files_all_categories = ["signal_tch.root", "signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
dd_files = ["signal_tch.root", "signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]
HCT_cat = ["signal_tch.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HUT_cat = ["signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HCT_cat_dd = ["signal_tch.root", "data_fakes.root", "data_flips.root", "rares.root"]
HUT_cat_dd = ["signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]

all_files    = (["2016/MC/" + f for f in files_all_categories] + ["2017/MC/" + f for f in files_all_categories] + ["2018/MC/" + f for f in files_all_categories])
HCT_files    = (["2016/MC/" + f for f in HCT_cat] + ["2017/MC/" + f for f in HCT_cat] + ["2018/MC/" + f for f in HCT_cat])
HUT_files    = (["2016/MC/" + f for f in HUT_cat] + ["2017/MC/" + f for f in HUT_cat] + ["2018/MC/" + f for f in HUT_cat])

all_files_dd = (["2016/data_driven/" + f for f in dd_files] + ["2017/data_driven/" + f for f in dd_files] + ["2018/data_driven/" + f for f in dd_files])
HCT_files_dd = (["2016/data_driven/" + f for f in HCT_cat_dd] + ["2017/data_driven/" + f for f in HCT_cat_dd] + ["2018/data_driven/" + f for f in HCT_cat_dd])
HUT_files_dd = (["2016/data_driven/" + f for f in HUT_cat_dd] + ["2017/data_driven/" + f for f in HUT_cat_dd] + ["2018/data_driven/" + f for f in HUT_cat_dd])

input_baby_dir   = "/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/"
base_output_dir  = "/home/users/cmcmahon/public_html/BDT"

input_baby_dir   = "/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/"
HCT              = BDT_analysis.BDT(input_baby_dir, HCT_files, base_output_dir, label="HCT", year="all")
HUT              = BDT_analysis.BDT(input_baby_dir, HUT_files, base_output_dir, label="HUT", year="all")

every_BDT = [HCT, HUT]

quantile_dict = {
    '0.025': '-2sig',
    '0.16': '-1sig',
    '0.5': 'expected',
    '0.84': '+1sig',
    '0.975': '+2sig',
    '-1.0': 'observed'
}

def readResFile(fname):
    f = uproot.open(fname)
    t = f["limit;1"]
    limits = t.arrays()["limit"]
    quantiles = t.arrays()["quantileExpected"]
    #quantiles = quantiles.astype(str)
    limit = { quantile_dict[str(round(q, 3))]:limits[i] for i,q in enumerate(quantiles) }
    #print(limit)
    return limit

def calcLimit(release_location, fname=None, options="", verbose=False):
#    ustr          = str(uuid.uuid4())
    uniqueDirname = os.path.join(release_location)#, ustr)
#     if verbose: print("Creating %s"%uniqueDirname)
#     os.makedirs(uniqueDirname)
    os.makedirs(release_location, exist_ok=True)
    if fname is not None:  # Assume card is already written when fname is not none
        filename = os.path.abspath(fname)
    else:
        filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
        #self.writeToFile(filename)
    resultFilename = filename.replace('.txt','')+'.root'
    assert os.path.exists(filename), "File not found: %s"%filename
    combineCommand = "cd /home/users/cmcmahon/CMS_releases/CMSSW_8_1_0/src;eval `scramv1 runtime -sh`;" + "cd "+release_location+";combine --saveWorkspace -M AsymptoticLimits %s %s"%(options,filename)
    if verbose: print("Executing command:", combineCommand)
    os.system(combineCommand)

#     tempResFile = uniqueDirname+"/higgsCombineTest.AsymptoticLimits.mH120.root"
    tempResFile = release_location+"higgsCombineTest.AsymptoticLimits.mH120.root"
    
    try:
        res= readResFile(tempResFile)
        res['card'] = fname
    except:
        res=None
        print("[cardFileWrite] Did not succeed reading result.")
    if res:
        shutil.copyfile(tempResFile, resultFilename)
    os.system("rm -f {}".format(tempResFile))
    return res

#print(calcLimit("/home/users/cmcmahon/public_html/BDT/HCT/datacards/","/home/users/cmcmahon/public_html/BDT/HCT/datacards/HCT/datacard_HCT_2016_QT.txt", verbose=True))
class combine_result:
    def __init__(self, bdt, sig_type, base_data_dir="/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/"):
        self.bdt=bdt
        self.sig_type = sig_type #"HCT" or "HUT"
        self.base_data_dir = base_data_dir
        self.QT_result = {}
        self.standard_result = {}
        
    def get_limits(self, bin_spacing, QT=True, verbose=False, dir_label="all_features"):
        for y in [2016, 2017, 2018]:
            data_dir = self.base_data_dir + "{}/data_driven/".format(y)
            self.bdt.gen_datacard(self.sig_type, y, [data_dir], quantile_transform=QT, data_driven=True, plot=False, BDT_bins=np.linspace(0, 1, bin_spacing+1), flag_tmp_directory=True, dir_label=dir_label)
            #self.bdt.gen_datacards(data_dir, y, quantile_transform=True, data_driven=True, BDT_bins=np.linspace(0, 1, bin_spacing+1), flag_tmp_directory=True, plot=False)
        bdt_basedir = "{0}/{1}/datacards/tmp/{2}/".format(self.bdt.out_base_dir, self.bdt.label, dir_label)
        combine_years_cmd = "cd /home/users/cmcmahon/CMS_releases/CMSSW_8_1_0/src;eval `scramv1 runtime -sh`;"
        combine_years_cmd += "cd /home/users/cmcmahon/public_html/BDT/combine_scripts/;"
        combine_years_cmd += "bash year_combine2.sh {0} {1} {2}".format(bdt_basedir, self.sig_type, int(QT))
        os.system(combine_years_cmd) #generate a combined years datacard
        card_path = "{0}{1}/dc_{1}_combined_years.txt".format(bdt_basedir, self.sig_type)
        res = calcLimit(bdt_basedir, card_path, verbose=False)
        #move datacards to another folder when done
        #os.system("rm {0}{1}/*.txt".format(bdt_basedir, self.sig_type))
        #os.system("rm {0}{1}/*.root".format(bdt_basedir, self.sig_type))
        if QT:
            self.QT_result[bin_spacing] = res
        else:
            self.standard_result[bin_spacing] = res
        if verbose:
            print("{0}\tnum_bins={1}\texpected limit={2:.5f}\tQT={3}".format(self.sig_type, bin_spacing, res["expected"], QT))
        return res

base_BDT_features = ["Most_Forward_pt",
              "HT",
              "LeadLep_eta",
              "LeadLep_pt",
              "LeadLep_dxy",
              "LeadLep_dz",
              "SubLeadLep_pt",
              "SubLeadLep_eta",
              "SubLeadLep_dxy",
              "SubLeadLep_dz",
              "nJets",
              "nBtag",
              "LeadJet_pt",
              "SubLeadJet_pt",
              "SubSubLeadJet_pt",
              "LeadJet_BtagScore",
              "SubLeadJet_BtagScore",
              "SubSubLeadJet_BtagScore",
              "nElectron",
              "MET_pt",
              "LeadBtag_pt",
              "MT_LeadLep_MET",
              "MT_SubLeadLep_MET",
              "LeadLep_SubLeadLep_Mass",
              "SubSubLeadLep_pt",
              "SubSubLeadLep_eta",
              "SubSubLeadLep_dxy",
              "SubSubLeadLep_dz",
              "MT_SubSubLeadLep_MET",
              "LeadBtag_score",
              "Weight"]

base_limits={"HCT":0.06294, "HUT":0.05581}
base_errors={"HCT":0.00137, "HUT":0.001706}

# limit_importance_result = {}
# print("training BDTs while removing single features:")
# for feature_idx in range(len(base_BDT_features)-1):
#     remaining_features = base_BDT_features.copy()
#     feature = remaining_features.pop(feature_idx)
#     print(feature)
#     tmp_res = {"HCT":[], "HUT":[]}
#     for n in range(5):
#         print("\tpass {}".format(n+1))
#         tmp_HCT = BDT_analysis.BDT(input_baby_dir, HCT_files, base_output_dir, label="HCT", year="all", BDT_features=remaining_features)
#         tmp_HUT = BDT_analysis.BDT(input_baby_dir, HUT_files, base_output_dir, label="HUT", year="all", BDT_features=remaining_features)
        
#         for bdt in [tmp_HCT, tmp_HUT]:
#             bdt.gen_BDT_and_plot(load_BDT=True, optimize=False, retrain=True, flag_save_booster=False, plot=False)
#             bdt_result = combine_result(bdt, bdt.label)
#             lim = bdt_result.get_limits(20, QT=True, verbose=False, dir_label=feature)
#             tmp_res[bdt.label].append(lim)
#             #tmp_res[bdt.label] = (lim["expected"] / base_limits[bdt.label]) - 1.0
#             print("\t\t{0}\t{1:.6f}".format(bdt.label, lim["expected"]))
#     limit_importance_result[feature] = tmp_res.copy()
    
# print(limit_importance_result)
# pickle.dump(limit_importance_result, open("importance_lims_3.p", "wb"))

feature_groups = {
    "Btag_Scores":["LeadJet_BtagScore", "SubLeadJet_BtagScore", "SubSubLeadJet_BtagScore", "LeadBtag_score"],
    "SubSubLeadLep":["SubSubLeadLep_pt", "SubSubLeadLep_eta", "SubSubLeadLep_dxy", "SubSubLeadLep_dz", "MT_SubSubLeadLep_MET"],
    "Lepton_pt":["LeadLep_pt", "SubLeadLep_pt", "SubSubLeadLep_pt"],
    "Jet_pt":["LeadJet_pt", "SubLeadJet_pt", "SubSubLeadJet_pt", "LeadBtag_pt", "Most_Forward_pt", "HT"],
    "dxy":["LeadLep_dxy", "SubLeadLep_dxy", "SubSubLeadLep_dxy"],
    "dz":["LeadLep_dz", "SubLeadLep_dz", "SubSubLeadLep_dz"],
    "mt":["MT_LeadLep_MET", "MT_SubLeadLep_MET", "MT_SubSubLeadLep_MET"],
    "eta":["LeadLep_eta", "SubLeadLep_eta", "SubSubLeadLep_eta"],
    "multiplicity":["nJets", "nBtag", "nElectron"]
}

grouped_limit_importance_result = {}
print("training BDTs while removing feature groups:")
for fg in feature_groups.keys():
    remaining_features = base_BDT_features.copy()
    [remaining_features.pop(remaining_features.index(f)) for f in feature_groups[fg]] #remove all features in the group
    print("{}: {}".format(fg, feature_groups[fg]))
    tmp_res = {"HCT":[], "HUT":[]}
    for n in range(5):
        print("\tpass {}".format(n+1))
        tmp_HCT = BDT_analysis.BDT(input_baby_dir, HCT_files, base_output_dir, label="HCT", year="all", BDT_features=remaining_features)
        tmp_HUT = BDT_analysis.BDT(input_baby_dir, HUT_files, base_output_dir, label="HUT", year="all", BDT_features=remaining_features)
        
        for bdt in [tmp_HCT, tmp_HUT]:
            bdt.gen_BDT_and_plot(load_BDT=True, optimize=False, retrain=True, flag_save_booster=False, plot=False)
            bdt_result = combine_result(bdt, bdt.label)
            lim = bdt_result.get_limits(20, QT=True, verbose=False, dir_label=fg)
            tmp_res[bdt.label].append(lim)
            #tmp_res[bdt.label] = (lim["expected"] / base_limits[bdt.label]) - 1.0
            print("\t\t{0}\t{1:.6f}".format(bdt.label, lim["expected"]))
    grouped_limit_importance_result[fg] = tmp_res.copy()
    
print(grouped_limit_importance_result)
pickle.dump(grouped_limit_importance_result, open("grouped_importance_lims.p", "wb"))