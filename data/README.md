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
If you want to use the NanoAOD sample directly, also add it to the corresponding category in `nano_mapping.yaml`.
