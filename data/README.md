# README

This is a collection of data and config files.

Searching for a MiniAOD sample on DAS:
```
dasgoclient -query="/QCD_Pt-*Mu*/RunIIAutumn18*/MINIAODSIM"
```
Watch out for special names (like flat PU, Val etc - those are usually special PPD samples that we don't want to use in an analysis).

To get the correspoding NanoAOD child of a MiniAOD:
```
dasgoclient -query="child dataset=/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
```

The NanoAOD sample goes into a `samplesX.txt` file, together with the x-sec (in pb).
Navigate to ../postProcessing/ and run `ipython -i getSampleInformation.py -- --name samplesX` to get the corresponding `samplesX.yaml` file that contains the sum of weights etc.
If you want to use the NanoAOD sample directly, also add it to the corresponding category in `nano_mapping.yaml`


## Ultra-Legecay samples

[Summary](https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun2LegacyAnalysis)

- 2016 preVFP = APV, runs B-F, 19.5/fb
- 2016 postVFP, runs G-H, 16.8/fb
- 2017 41.48/fb
- 2018 59.83/fb

This is a table for 2016.
```
/DoubleMuon/Run2016B-ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016C-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016D-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016E-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016F-HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016F-UL2016_MiniAODv1_NanoAODv2-v2/NANOAOD
/DoubleMuon/Run2016G-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
/DoubleMuon/Run2016H-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD
```

Lumis are given [here](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM)
