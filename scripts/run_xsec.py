'''
Get x-secs. Needs a working cmsenv, like:
cd /home/users/dspitzba/TOP/CMSSW_10_6_19/src/; cmsenv; cd -;

'''


import imp, os, sys
import subprocess, shutil

## default cmsRun cfg file
defaultCFG = """
import FWCore.ParameterSet.Config as cms
process = cms.Process("GenXSec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring('{FILEPATH}') )
process.dummy = cms.EDAnalyzer("GenXSecAnalyzer", genFilterInfoTag = cms.InputTag("genFilterEfficiencyProducer") )
process.p = cms.Path(process.dummy)"""

cfgFile     = 'xsecCfg.py'
identifier  = "After filter: final cross section ="


from Tools.helpers import dasWrapper

def get_mini_file(nano):
    try:
        mini = dasWrapper(nano, 'parent')[0]
    except IndexError:
        print ("No parent found")
        return None
    mini_file = dasWrapper(mini, 'file')[0]
    return mini_file

samples = [
    #'/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM', ##FIXME Filter not taken into account!!
    #'/ttHJetTobb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM', ##FIXME Filter not taken into account!!
    #'/WZTo3LNu_mllmin01_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM', ## 5.859e+01
    #'/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM', ## 4.664
    #'/WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM', ## 4.664
    #'/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM', ## 5.212
    #'/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM', # 0.2178
    #'/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/SSWW_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
    #'/QCD_Pt_15to30_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_30to50_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_50to80_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_80to120_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_120to170_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #"/TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",  # 10.06
    #"/TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct_TuneCP5-MadGraph5-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_tauDecays_102X_mc2017_realistic_v8-v1/NANOAODSIM",  # 15.06
    #"/TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct_TuneCP5-MadGraph5-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_tauDecays_102X_upgrade2018_realistic_v21-v1/NANOAODSIM",  # 15.01
    #"/TT_FCNC-aTtoHJ_Tleptonic_HToWWZZtautau_eta_hct-MadGraph5-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",  # 16.25
    #"/ZGTo2LG_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8_ext1-v1/NANOAODSIM",
    #"/ZGToLLG_01J_5f_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",
    #"/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
    #"/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM",
    #'/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
    #'/SSWW_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
    #'/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv2-Pilot_106X_mc2017_realistic_v8-v3/NANOAODSIM',
    #'/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
    #'/store/mc/PhaseIISummer17wmLHEGENOnly/TTWJetsToLNu_TuneCUETP8M1_14TeV-amcatnloFXFX-madspin-pythia8/GEN/93X_upgrade2023_realistic_v5_ext1-v3/10000/0029E818-8552-E811-87D1-1866DAEA6D08.root', # 0.2378
    #'/store/mc/PhaseIISummer17wmLHEGENOnly/TTZToLLNuNu_M-10_TuneCUETP8M1_14TeV-amcatnlo-pythia8/GEN/93X_upgrade2023_realistic_v5-v1/80000/8A4401A0-5672-E811-8903-0CC47A84788A.root', # 0.2949
    #'/store/mc/RunIIAutumn18MiniAOD/ggTomonoH_bb_sn_0p35_tn_1p0_mXd_10_MH3_1600_MH4_350_MH2_1600_MHC_1600_TuneCP3_13TeV_madgraph-pythia/MINIAODSIM/102X_upgrade2018_realistic_v15-v2/100000/461BFB17-6F80-134C-B2B9-AD46C2B59EB6.root',
    #'/store/mc/RunIISummer20UL18MiniAODv2/THQ_ctcvcp_4f_Hincl_TuneCP5_13TeV_madgraph_pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/80000/3C272FC5-CFCA-594E-8B9A-FC1877E43C47.root',
    #'/ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM',
    #'/THW_5f_Hincl_13TeV_madgraph_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
    #'/THQ_4f_Hincl_13TeV_madgraph_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
    #'/THQ_ctcvcp_4f_Hincl_TuneCP5_13TeV_madgraph_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM',
    #'/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM',
    '/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM',
    ]
    
print ("## Will process the following samples: %s"%(",".join( f for f in samples ) ))

results = []

for nano in samples:
    print (nano)
    if nano.startswith('root://') or nano.startswith('/store/'):
        # 
        f_in = nano
    else:
        f_in = get_mini_file(nano)
    if f_in is None: continue
    replaceString = {'FILEPATH': f_in}
    cmsCfgString = defaultCFG.format( **replaceString )
    
    cmsRunCfg = open(cfgFile, 'w')
    cmsRunCfg.write(cmsCfgString)
    cmsRunCfg.close()

    print ("Working on Dataset:", nano)
    print (" - file: %s"%f_in)

    
    p = subprocess.Popen(['cmsRun', cfgFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stderr.readlines()
    for line in output:
        #print line
        if line.startswith(identifier): result = line

    xsec, unc = float(result.split(identifier)[1].split('+-')[0]), float(result.split(identifier)[1].split('+-')[1].replace('pb',''))
    
    results.append((nano.strip('/').split('/')[0], xsec))

    #os.remove(cfgFile)
    
print ("Found the following x-secs:")
print ("{:80}{:10}".format("Name", "x-sec (pb)"))
for res in results:
    print ("{:80}{:10}".format(res[0], res[1]))
