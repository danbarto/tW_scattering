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
#!source ~/scripts/renew_cms.sh
from yahist import Hist1D
import processor.BDT_analysis as BDT_analysis
from coffea import processor, hist

import Tools.objects

import postProcessing.makeCards
import postProcessing.datacard_comparison.compare_datacards as compare_datacards

files_all_categories = ["signal_tch.root", "signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
dd_files = ["signal_tch.root", "signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]
HCT_cat = ["signal_tch.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HUT_cat = ["signal_tuh.root", "fakes_mc.root", "flips_mc.root", "rares.root"]
HCT_cat_dd = ["signal_tch.root", "data_fakes.root", "data_flips.root", "rares.root"]
HUT_cat_dd = ["signal_tuh.root", "data_fakes.root", "data_flips.root", "rares.root"]

all_files              = (["2016/MC/" + f for f in files_all_categories] + ["2017/MC/" + f for f in files_all_categories] + ["2018/MC/" + f for f in files_all_categories])
HCT_files              = (["2016/MC/" + f for f in HCT_cat] + ["2017/MC/" + f for f in HCT_cat] + ["2018/MC/" + f for f in HCT_cat])
HUT_files              = (["2016/MC/" + f for f in HUT_cat] + ["2017/MC/" + f for f in HUT_cat] + ["2018/MC/" + f for f in HUT_cat])

all_files_dd           = (["2016/data_driven/" + f for f in dd_files] + ["2017/data_driven/" + f for f in dd_files] + ["2018/data_driven/" + f for f in dd_files])
HCT_files_dd           = (["2016/data_driven/" + f for f in HCT_cat_dd] + ["2017/data_driven/" + f for f in HCT_cat_dd] + ["2018/data_driven/" + f for f in HCT_cat_dd])
HUT_files_dd           = (["2016/data_driven/" + f for f in HUT_cat_dd] + ["2017/data_driven/" + f for f in HUT_cat_dd] + ["2018/data_driven/" + f for f in HUT_cat_dd])

input_baby_dir   = "/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/" ###change this if the babies change
base_output_dir  = "/home/users/cmcmahon/public_html/BDT"

#all_categories   = BDT_analysis.BDT(input_baby_dir, all_files, base_output_dir, label="all_categories", year="all")
HCT              = BDT_analysis.BDT(input_baby_dir, HCT_files, base_output_dir, label="HCT", year="all")
HUT              = BDT_analysis.BDT(input_baby_dir, HUT_files, base_output_dir, label="HUT", year="all")

every_BDT = [HCT, HUT]

for bdt in every_BDT:
    bdt.gen_BDT_and_plot(load_BDT=True, optimize=False, retrain=True)
    
retrain_BDTs = [HCT, HUT]#[HCT_dilep, HCT_flips, HCT_trilep_fakes, HCT_trilep_flips, HUT_dilep, HUT_trilep, HUT_trilep_fakes, HUT_trilep_flips, trilep_fakes]
for bdt in retrain_BDTs:
    bdt.gen_BDT_and_plot(load_BDT=False, optimize=True, retrain=True)
    
flag_gen_datacards = True
if flag_gen_datacards:
    for y in [2016, 2017, 2018]:
        data_dir = ["/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies/{}/data_driven/".format(y)]
        for bdt in every_BDT:
            bdt.gen_datacards(data_dir, y, quantile_transform=True, data_driven=True)
    
flag_gen_categories = True
if flag_gen_categories:
    all_directories = ["/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies" + f for f in ["/2016/MC/", "/2017/MC/", "/2018/MC/"]]
    all_directories_data_driven = ["/home/users/cmcmahon/fcnc/ana/analysis/helpers/BDT/babies" + f for f in ["/2016/data_driven/", "/2017/data_driven/", "/2018/data_driven/"]]
    #run this later, compare data driven vs mc
    for bdt in [HCT, HUT]:#, all_categories]:
    #HCT.fill_dicts(all_directories_data_driven, data_driven=True)
        bdt.fill_dicts(all_directories, data_driven=False)
        bdt.plot_categories(plot=True, savefig=True)
        for l in ["all", "signal", "fakes", "flips", "rares"]:
            bdt.plot_response(plot=True, savefig=True, label=l)