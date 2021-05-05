# Ultra Legacy 2018 PSets

Inspired by: `TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8`

Whole chain in [MCM](https://cms-pdmv.cern.ch/mcm/chained_requests?prepid=TOP-chain_RunIISummer20UL18wmLHEGEN_flowRunIISummer20UL18SIM_flowRunIISummer20UL18DIGIPremix_flowRunIISummer20UL18HLT_flowRunIISummer20UL18RECO_flowRunIISummer20UL18MiniAODv2-00049&page=0&shown=15)

Everything in `CMSSW_10_6_19_patch3`?

## LHE

[gridpack->LHE](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL18wmLHEGEN-00071)

```
export SCRAM_ARCH=slc7_amd64_gcc700
scram p CMSSW CMSSW_10_6_19_patch3
cd CMSSW_10_6_19_patch3/src
eval `scram runtime -sh`

curl -s -k https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_fragment/TOP-RunIISummer20UL18wmLHEGEN-00071 --retry 3 --create-dirs -o Configuration/GenProduction/python/TOP-RunIISummer20UL18wmLHEGEN-00071-fragment.py

scram b -j 8

cmsDriver.py Configuration/GenProduction/python/tW_scattering.py --python_filename gen_cfg.py --eventcontent RAWSIM,LHE --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN,LHE --fileout file:output_gen.root --conditions 106X_upgrade2018_realistic_v4 --beamspot Realistic25ns13TeVEarly2018Collision --customise_commands process.source.numberEventsInLuminosityBlock="cms.untracked.uint32(161)" --step LHE,GEN --geometry DB:Extended --era Run2_2018 --no_exec --mc -n 10
```

## SIM

Same release? Officially `CMSSW_10_6_17_patch1`

[SIM](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL18SIM-00119)

This is the bottleneck per event (~1ev/min).

```
cmsDriver.py  --python_filename sim_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM --fileout file:output_sim.root --conditions 106X_upgrade2018_realistic_v11_L1v1 --beamspot Realistic25ns13TeVEarly2018Collision --step SIM --geometry DB:Extended --filein file:output_gen.root --era Run2_2018 --runUnscheduled --no_exec --mc -n 10
```

## Premix

Same release? Officially `CMSSW_10_6_17_patch1`

[Premix](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL18DIGIPremix-00119)

This is usually the setup bottleneck because of the premix dbs query, but was suprisingly fast.

```
cmsDriver.py  --python_filename premix_cfg.py --eventcontent PREMIXRAW --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-DIGI --fileout file:output_premix.root --pileup_input "dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL18_106X_upgrade2018_realistic_v11_L1v1-v2/PREMIX" --conditions 106X_upgrade2018_realistic_v11_L1v1 --step DIGI,DATAMIX,L1,DIGI2RAW --procModifiers premix_stage2 --geometry DB:Extended --filein file:output_sim.root --datamix PreMix --era Run2_2018 --runUnscheduled --no_exec --mc -n 10
```

## HLT

Has to be run in `CMSSW_10_2_16_UL` <- WTF is this? Who thought this was a good idea??

[HLT](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL18HLT-00119)

```
cmsDriver.py  --python_filename hlt_cfg.py --eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-RAW --fileout file:output_hlt.root --conditions 102X_upgrade2018_realistic_v15 --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True)' --step HLT:2018v32 --geometry DB:Extended --filein file:output_premix.root --era Run2_2018 --no_exec --mc -n 10
```

## RECO

Release: `CMSSW_10_6_17_patch1`

[Reco](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL18RECO-00119)


## Combine RECO-MAOD

[Taken from here](https://twiki.cern.ch/twiki/bin/view/CMS/PdmVLegacy2018Analysis)

```
cmsDriver.py step2 --filein file:output_hlt.root --fileout file:output_maod.root --mc --eventcontent MINIAODSIM --datatier MINIAODSIM --runUnscheduled --conditions 106X_upgrade2018_realistic_v15_L1v1 --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT --nThreads 8 --geometry DB:Extended --era Run2_2018 --python_filename maod_cfg.py -n 10 --no_exec
```

## NanoAOD

```
cmsDriver.py step3 --filein file:output_maod.root --fileout file:output_nano.root --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 106X_upgrade2018_realistic_v15_L1v1 --step NANO --nThreads 8 --era Run2_2018 --python_filename nano_cfg.py -n 10 --no_exec
```
