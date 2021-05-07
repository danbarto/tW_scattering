# UL for 2017

Loosly following [PdmV](https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVLegacy2017Analysis)

## Gridpack -> LHE -> GEN

```
export SCRAM_ARCH=slc7_amd64_gcc700
scram p CMSSW CMSSW_10_6_20
cd CMSSW_10_6_20/src
eval `scram runtime -sh`

curl -s -k https://raw.githubusercontent.com/danbarto/tW_scattering/master/production/psets/tW_scattering.py --retry 3 --create-dirs -o Configuration/GenProduction/python/tW_scattering.py

scram b -j 8
```

```
cmsDriver.py Configuration/GenProduction/python/tW_scattering.py --fileout file:output_gen.root --mc --eventcontent RAWSIM,LHE --datatier GEN,LHE --conditions 106X_mc2017_realistic_v8 --beamspot Realistic25ns13TeVEarly2017Collision --step LHE,GEN --geometry DB:Extended --era Run2_2017 --python_filename gen_cfg.py -n 10 --no_exec
```

## SIM

```
cmsDriver.py --python_filename sim_cfg.py --filein file:output_gen.root --fileout file:output_sim.root --mc --eventcontent RAWSIM --runUnscheduled --datatier GEN-SIM --conditions 106X_mc2017_realistic_v8 --beamspot Realistic25ns13TeVEarly2017Collision --step SIM --geometry DB:Extended --era Run2_2017 --nThreads 8 -n 10 --no_exec
```

Could be run with 8 threads? Use `--nThreads 8`

## Premix

```
cmsDriver.py --python_filename premix_cfg.py --filein file:output_sim.root --fileout file:output_premix.root  --pileup_input "dbs:/Neutrino_E-10_gun/RunIISummer20ULPrePremix-UL17_106X_mc2017_realistic_v6-v3/PREMIX" --mc --eventcontent PREMIXRAW --runUnscheduled --datatier GEN-SIM-DIGI --conditions 106X_mc2017_realistic_v6 --step DIGI,DATAMIX,L1,DIGI2RAW --procModifiers premix_stage2 --geometry DB:Extended --datamix PreMix --era Run2_2017 --nThreads 8 -n 10 --no_exec
```

## HLT

Needs `CMSSW_9_4_14_UL_patch1`.
```
cmsDriver.py --python_filename hlt_cfg.py --filein file:output_premix.root --fileout file:output_hlt.root --mc --eventcontent RAWSIM --datatier GEN-SIM-RAW --conditions 94X_mc2017_realistic_v15 --customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True)' --step HLT:2e34v40 --geometry DB:Extended --era Run2_2017 --nThreads 8 -n 10 --no_exec
```

## Reco -> MAOD

```
cmsDriver.py --python_filename maod_cfg.py --filein file:output_hlt.root --fileout file:miniAOD.root --mc --eventcontent MINIAODSIM --datatier MINIAODSIM --runUnscheduled --conditions 106X_mc2017_realistic_v8 --step RAW2DIGI,L1Reco,RECO,RECOSIM,PAT --nThreads 8 --geometry DB:Extended --era Run2_2017 -n 10 --no_exec
```

## NanoAOD

```
cmsDriver.py --python_filename nano_cfg.py --filein file:miniAOD.root --fileout file:nanoAOD.root --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 106X_mc2017_realistic_v8 --step NANO --era Run2_2017 -n 10 --no_exec
```

