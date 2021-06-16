import os.path

samples_2018 = {
    "dyjets_m10-50" :  "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "dyjets_m50" :  "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "ggh" :  "GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "tw_dilep" :  "ST_tWll_5f_LO_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "tg" :  "TGJets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "signal_hct_atop" :  "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct_TuneCP5-MadGraph5-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_tauDecays_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "signal_hut_atop" :  "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hut_TuneCP5-MadGraph5-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_tauDecays_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "signal_hct_top" :  "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hct_TuneCP5-MadGraph5-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_tauDecays_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "signal_hut_top" :  "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hut_TuneCP5-MadGraph5-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_tauDecays_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "ttg_dilep" :  "TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "ttg_1lep" :  "TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "tthh" :  "TTHH_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "tth_nobb" :  "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "ttjets" :  "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1_NANOAODSIM",
    "tt1lep" : "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "tt2lep" : "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "tttj" :  "TTTJ_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "tttt" :  "TTTT_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1_NANOAODSIM",
    "tttw" :  "TTTW_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttwh" :  "TTWH_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttw" :  "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttww" :  "TTWW_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1_NANOAODSIM",
    "ttwz" :  "TTWZ_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttzh" :  "TTZH_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttz_m1-10" :  "TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "ttz_m10" :  "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ttzz" :  "TTZZ_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "tzq" :  "tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "vh_nobb" :  "VHToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "wg" :  "WGToLNuG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "wjets" :  "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "qqww" :  "WpWpJJ_EWK-QCD_TuneCP5_13TeV-madgraph-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "wwg" :  "WWG_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "ww" :  "WWTo2L2Nu_DoubleScattering_13TeV-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "www" :  "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "wwz" :  "WWZ_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "wzg" :  "WZG_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1_NANOAODSIM",
    "wz" :  "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "wzz" :  "WZZ_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "zg" :  "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1_NANOAODSIM",
    "zz" :  "ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1_NANOAODSIM",
    "zzz" :  "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1_NANOAODSIM",
    "data" : "",
    "eg_data" : "EGamma_Run2018A-02Apr2020-v1_NANOAOD EGamma_Run2018B-02Apr2020-v1_NANOAOD EGamma_Run2018C-02Apr2020-v1_NANOAOD EGamma_Run2018D-02Apr2020-v1_NANOAOD",
    "double_muon_data" : "DoubleMuon_Run2018A-02Apr2020-v1_NANOAOD DoubleMuon_Run2018B-02Apr2020-v1_NANOAOD DoubleMuon_Run2018C-02Apr2020-v1_NANOAOD DoubleMuon_Run2018D-02Apr2020-v1_NANOAOD",
    "muon_eg_data" : "MuonEG_Run2018A-02Apr2020-v1_NANOAOD MuonEG_Run2018B-02Apr2020-v1_NANOAOD MuonEG_Run2018C-02Apr2020-v1_NANOAOD MuonEG_Run2018D-02Apr2020-v1_NANOAOD",
}
samples_2017 = {
    "dyjets_m10-50" : "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8_ext1-v1_NANOAODSIM",
    "dyjets_m50" : "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17NanoAODv7-PU2017RECOSIMstep_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8_ext1-v1_NANOAODSIM",
    "ggh" : "GluGluHToZZTo4L_M125_13TeV_powheg2_JHUGenV7011_pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8_ext3-v1_NANOAODSIM",
    "tw_dilep" : "ST_tWll_5f_LO_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8_ext1-v1_NANOAODSIM",
    "tg" : "TGJets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "signal_hct_atop" : "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "signal_hut_atop" : "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hut-MadGraph5-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "signal_hct_top" : "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "signal_hut_top" : "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hut-MadGraph5-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttg_dilep" : "TTGamma_Dilept_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttg_1lep" : "TTGamma_SingleLept_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tthh" : "TTHH_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v2_NANOAODSIM",
    "tth_nobb" : "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttjets" : "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tt1lep" : "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tt2lep" : "TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tttj" : "TTTJ_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tttt" : "TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tttw" : "TTTW_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttwh" : "TTWH_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttw" : "TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttww" : "TTWW_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8_ext1-v1_NANOAODSIM",
    "ttwz" : "TTWZ_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttzh" : "TTZH_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttz_m1-10" : "TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttz_m10" : "TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ttzz" : "TTZZ_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "tzq" : "tZq_ll_4f_ckm_NLO_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "vh_nobb" : "VHToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wg" : "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wjets" : "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8_ext1-v1_NANOAODSIM",
    "qqww" : "WpWpJJ_EWK-QCD_TuneCP5_13TeV-madgraph-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wwg" : "WWG_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "ww" : "WWTo2L2Nu_DoubleScattering_13TeV-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "www" : "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wwz" : "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wzg" : "WZG_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wz" : "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "wzz" : "WZZ_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "zg" : "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "zz" : "ZZTo4L_13TeV_powheg_pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "zzz" : "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1_NANOAODSIM",
    "data": "",
    "eg_data" : "DoubleEG_Run2017B-02Apr2020-v1_NANOAOD DoubleEG_Run2017C-02Apr2020-v1_NANOAOD DoubleEG_Run2017D-02Apr2020-v1_NANOAOD DoubleEG_Run2017E-02Apr2020-v1_NANOAOD DoubleEG_Run2017F-02Apr2020-v1_NANOAOD",
    "double_muon_data" : "DoubleMuon_Run2017B-02Apr2020-v1_NANOAOD DoubleMuon_Run2017C-02Apr2020-v1_NANOAOD DoubleMuon_Run2017D-02Apr2020-v1_NANOAOD DoubleMuon_Run2017E-02Apr2020-v1_NANOAOD DoubleMuon_Run2017F-02Apr2020-v1_NANOAOD",
    "muon_eg_data" : "MuonEG_Run2017B-02Apr2020-v1_NANOAOD MuonEG_Run2017C-02Apr2020-v1_NANOAOD MuonEG_Run2017D-02Apr2020-v1_NANOAOD MuonEG_Run2017E-02Apr2020-v1_NANOAOD MuonEG_Run2017F-02Apr2020-v1_NANOAOD",
}
samples_2016 = {
    "dyjets_m10-50" : "DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "dyjets_m50" : "DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext2-v1_NANOAODSIM",
    "ggh" : "GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tw_dilep" : "ST_tWll_5f_LO_13TeV-MadGraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tg" : "TGJets_TuneCUETP8M1_13TeV_amcatnlo_madspin_pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "signal_hct_atop" : "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "signal_hut_atop" : "TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hut-MadGraph5-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "signal_hct_top" : "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "signal_hut_top" : "TT_FCNC-TtoHJ_aTleptonic_HToWWZZtautau_eta_hut-MadGraph5-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "ttg_dilep" : "TTGamma_Dilept_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "ttg_1lep" : "TTGamma_SingleLept_TuneCP5_PSweights_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tthh" : "TTHH_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "tth_nobb" : "ttHToNonbb_M125_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "ttjets" : "TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tt1lep" : "TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tt2lep" : "TTTo2L2Nu_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tttj" : "TTTJ_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "tttt" : "TTTT_TuneCUETP8M2T4_PSweights_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "tttw" : "TTTW_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ttwh" : "TTWH_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ttw" : "TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext2-v1_NANOAODSIM",
    "ttww" : "TTWW_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ttwz" : "TTWZ_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ttzh" : "TTZH_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ttz_m1-10" : "TTZToLL_M-1to10_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv7-Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "ttz_m10" : "TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext3-v1_NANOAODSIM",
    "ttzz" : "TTZZ_TuneCUETP8M2T4_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "tzq" : "tZq_ll_4f_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "vh_nobb" : "VHToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "wg" : "WGToLNuG_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext3-v1_NANOAODSIM",
    "wjets" : "WJetsToLNu_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext2-v1_NANOAODSIM",
    "qqww" : "WpWpJJ_EWK-QCD_TuneCUETP8M1_13TeV-madgraph-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "wwg" : "WWG_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "ww" : "WWTo2L2Nu_DoubleScattering_13TeV-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "www" : "WWW_4F_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "wwz" : "WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "wzg" : "WZG_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "wz" : "WZTo3LNu_TuneCUETP8M1_13TeV-powheg-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "wzz" : "WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "zg" : "ZGTo2LG_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1_NANOAODSIM",
    "zz" : "ZZTo4L_13TeV_powheg_pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "zzz" : "ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM",
    "data" : "",
    "eg_data" : "DoubleEG_Run2016B-02Apr2020_ver2-v1_NANOAOD DoubleEG_Run2016C-02Apr2020-v1_NANOAOD DoubleEG_Run2016D-02Apr2020-v1_NANOAOD DoubleEG_Run2016E-02Apr2020-v1_NANOAOD DoubleEG_Run2016F-02Apr2020-v1_NANOAOD DoubleEG_Run2016G-02Apr2020-v1_NANOAOD DoubleEG_Run2016H-02Apr2020-v1_NANOAOD",
    "double_muon_data" : "DoubleMuon_Run2016B-02Apr2020_ver2-v1_NANOAOD DoubleMuon_Run2016C-02Apr2020-v1_NANOAOD DoubleMuon_Run2016D-02Apr2020-v1_NANOAOD DoubleMuon_Run2016E-02Apr2020-v1_NANOAOD DoubleMuon_Run2016F-02Apr2020-v1_NANOAOD DoubleMuon_Run2016G-02Apr2020-v1_NANOAOD DoubleMuon_Run2016H-02Apr2020-v1_NANOAOD",
    "muon_eg_data" : "MuonEG_Run2016B-02Apr2020_ver2-v1_NANOAOD MuonEG_Run2016C-02Apr2020-v1_NANOAOD MuonEG_Run2016D-02Apr2020-v1_NANOAOD MuonEG_Run2016E-02Apr2020-v2_NANOAOD MuonEG_Run2016F-02Apr2020-v1_NANOAOD MuonEG_Run2016G-02Apr2020-v1_NANOAOD MuonEG_Run2016H-02Apr2020-v1_NANOAOD",
}

xsecs = {}
xsecs["tw_dilep"] = 0.01123;
xsecs["ttjets"] = 831.762;
xsecs["tt1lep"] = 364.35;
xsecs["tt2lep"] = 87.31;
xsecs["tg"] = 2.967;
xsecs["wg"] = 405.271;
xsecs["wjets"] = 61334.9;
xsecs["wwg"] = 0.2147;
xsecs["ww"] = 0.16975;
xsecs["www"] = 0.2086;
xsecs["wwz"] = 0.1651;
xsecs["wzg"] = 0.04123;
xsecs["wz"] = 4.4297;
xsecs["wzz"] = 0.05565;
xsecs["qqww"] = 0.05390;
xsecs["zg"] = 405.271;
xsecs["zz"] = 1.256;
xsecs["zzz"] = 0.01398;
xsecs["ggh"] = 0.01181;
xsecs["ttg_dilep"] = 0.632;
xsecs["ttg_1lep"] = 0.77;
xsecs["tthh"] = 0.000757;
xsecs["tttj"] = 0.000474;
xsecs["tttt"] = 0.01197;
xsecs["tttw"] = 0.000788;
xsecs["ttwh"] = 0.001582;
xsecs["ttw"] = 0.2043;
xsecs["ttww"] = 0.01150;
xsecs["ttwz"] = 0.003884;
xsecs["ttzh"] = 0.001535;
xsecs["ttz_m10"] = 0.2529;
xsecs["ttz_m1-10"] = 0.0493;
xsecs["ttzz"] = 0.001982;
xsecs["dyjets_m10-50"] = 18610;
xsecs["dyjet_m50"] = 6020.85;
xsecs["vh_nobb"] = 2.1360;
xsecs["tzq"] = 0.0758;
xsecs["tth_nobb"] = 0.2710;
xsecs["signal_hct_atop"] = 83.88144;
xsecs["signal_hut_atop"] = 83.88144;
xsecs["signal_hct_top"] = 83.88144;
xsecs["signal_hut_top"] = 83.88144;

def get_weight(signal_name, year, version):
    samples_by_year = {2016:samples_2016, 2017:samples_2017, 2018:samples_2017}
    xsec = xsecs[signal_name]
    samples = samples_by_year[year]
    path_to_effective_events = os.path.expandvars("$TWHOME/production/n_events/{0}_{1}_n_events.txt".format(samples[signal_name], version))
    effective_events_file = open(path_to_effective_events, 'r')
    (nevents, effective_events) = effective_events_file.readlines()
    effective_events_file.close()
    effective_events = int(effective_events)
    return xsec / effective_events
    
    
    
#print(get_weight("signal_hct_atop", 2018, "fcnc_v6_SRonly_5may2021"))