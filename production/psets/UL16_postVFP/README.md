# UL16 post-VFP

[PdmV](https://twiki.cern.ch/twiki/bin/view/CMS/PdmVLegacy2016postVFPAnalysis)

## GEN

```
cmsDriver.py Configuration/GenProduction/python/tW_scattering.py --fileout file:output_gen.root --mc --eventcontent RAWSIM,LHE --datatier GEN,LHE --conditions 106X_mcRun2_asymptotic_v13 --beamspot Realistic25ns13TeV2016Collision --step LHE,GEN --geometry DB:Extended --era Run2_2016 --python_filename gen_cfg.py -n 10 --no_exec
```

## SIM

```
cmsDriver.py --filein file:output_gen.root --fileout file:output_sim.root --mc --eventcontent RAWSIM --runUnscheduled --datatier GEN-SIM --conditions 106X_mcRun2_asymptotic_v13 --beamspot Realistic25ns13TeV2016Collision --step SIM --nThreads 8 --geometry DB:Extended --era Run2_2016 --python_filename sim_cfg.py -n 10 --no_exec
```

## Premix

```
cmsDriver.py --filein file:output_sim.root --fileout file:output_premix.root  --pileup_input "dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL16_106X_mcRun2_asymptotic_v13-v1/PREMIX" --mc --eventcontent PREMIXRAW --runUnscheduled --datatier GEN-SIM-DIGI --conditions 106X_mcRun2_asymptotic_v13 --step DIGI,DATAMIX,L1,DIGI2RAW --procModifiers premix_stage2 --geometry DB:Extended --datamix PreMix --era Run2_2016 --python_filename premix_cfg.py -n 10 --no_exec
```

## HLT

Needs `CMSSW_8_0_33_UL` (OMG)
Scram arch: `slc7_amd64_gcc530`

```
cmsDriver.py --filein file:output_premix.root --fileout file:output_hlt.root --mc --eventcontent RAWSIM --outputCommand "keep *_mix_*_*,keep *_genPUProtons_*_*" --datatier GEN-SIM-RAW --inputCommands "keep *","drop *_*_BMTF_*","drop *PixelFEDChannel*_*_*_*" --conditions 80X_mcRun2_asymptotic_2016_TrancheIV_v6 --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True)' --step HLT:25ns15e33_v4 --nThreads 8 --geometry DB:Extended --era Run2_2016 --python_filename hlt_cfg.py -n 10 --no_exec
```

## Raw->MAOD

```
cmsDriver.py --filein file:output_hlt.root --fileout file:miniAOD.root --mc --eventcontent MINIAODSIM --datatier MINIAODSIM --runUnscheduled --conditions 106X_mcRun2_asymptotic_v13 --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT --nThreads 8 --geometry DB:Extended --era Run2_2016 --python_filename maod_cfg.py -n 10 --no_exec
```

## NANOAOD

```
cmsDriver.py --filein file:miniAOD.root --fileout file:nanoAOD.root --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 106X_mcRun2_asymptotic_v13 --step NANO --era Run2_2016 --python_filename nano_cfg.py -n 10 --no_exec
```

## Raw->MAODv2

In CMSSW_10_6_20
from [McM](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL16MiniAODv2-00112)

``` shell
cmsDriver.py --python_filename maodv2_cfg.py --filein file:output_hlt.root --fileout file:miniAOD.root --mc --eventcontent MINIAODSIM --datatier MINIAODSIM --runUnscheduled --conditions 106X_mcRun2_asymptotic_v17 --procModifiers run2_miniAOD_UL --geometry DB:Extended --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT --nThreads 8 --era Run2_2016 -n 10 --no_exec
```

## NanoAOD v9

In CMSSW_10_6_27
from [McM](https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/TOP-RunIISummer20UL16NanoAODv9-00113)

``` shell
cmsDriver.py step3 --mc --filein file:miniAOD.root --fileout file:nanoAOD.root --conditions 106X_mcRun2_asymptotic_v17 --step NANO --era Run2_2016,run2_nanoAOD_106Xv2 --eventcontent NANOAODSIM --datatier NANOAODSIM --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)));process.MessageLogger.cerr.FwkReport.reportEvery=100" --nThreads 8 --python_filename nanov9_cfg.py -n -1 --no_exec
```
