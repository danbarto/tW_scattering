#!/bin/bash

NEVENTS=$1
RANDOM_SEED=$2
OUTDIR=$3


export SCRAM_ARCH=slc6_amd64_gcc700
source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ -r CMSSW_10_2_7/src ] ; then 
 echo release CMSSW_10_2_7 already exists
else
scram p CMSSW CMSSW_10_2_7
fi
cd CMSSW_10_2_7/src
eval `scram runtime -sh`

cd ../../
mkdir -p CMSSW_10_2_7/src/Configuration/GenProduction/python/
cp tW_scattering.py CMSSW_10_2_7/src/Configuration/GenProduction/python/
cd CMSSW_10_2_7/src

scram b
cd ../../

seed=$(date +%s)

### run the LHE-GEN-SIM step

cmsDriver.py Configuration/GenProduction/python/tW_scattering.py --fileout file:tW_scattering_GS_${RANDOM_SEED}.root --mc --eventcontent RAWSIM,LHE --datatier GEN-SIM,LHE --conditions 102X_upgrade2018_realistic_v11 --beamspot Realistic25ns13TeVEarly2018Collision --step LHE,GEN,SIM --nThreads 1 --geometry DB:Extended --era Run2_2018 --python_filename tW_scattering_${RANDOM_SEED}_cfg.py --customise Configuration/DataProcessing/Utils.addMonitoring --customise_commands process.RandomNumberGeneratorService.externalLHEProducer.initialSeed=$RANDOM_SEED -n $NEVENTS || exit $? ; 

### run step1

cmsDriver.py step1 --filein file:tW_scattering_GS_${RANDOM_SEED}.root --fileout file:tW_scattering_step1_${RANDOM_SEED}.root  --pileup_input /store/mc/RunIISummer17PrePremix/Neutrino_E-10_gun/GEN-SIM-DIGI-RAW/PUAutumn18_102X_upgrade2018_realistic_v15-v1/00021/533FDEB2-6193-7549-BDA4-B9E633941C41.root --mc --eventcontent PREMIXRAW --datatier GEN-SIM-RAW --conditions 102X_upgrade2018_realistic_v15 --step DIGI,DATAMIX,L1,DIGI2RAW,HLT:@relval2018 --procModifiers premix_stage2 --nThreads 1 --datamix PreMix --era Run2_2018 --python_filename tW_scattering_1_${RANDOM_SEED}_cfg.py --customise Configuration/DataProcessing/Utils.addMonitoring -n $NEVENTS

### run step2
cmsDriver.py step2 --filein file:tW_scattering_step1_${RANDOM_SEED}.root --fileout file:tW_scattering_DRPremix_${RANDOM_SEED}.root --mc --eventcontent AODSIM --runUnscheduled --datatier AODSIM --conditions 102X_upgrade2018_realistic_v15 --step RAW2DIGI,L1Reco,RECO,RECOSIM,EI --procModifiers premix_stage2 --nThreads 1 --era Run2_2018 --python_filename tW_scattering_2_${RANDOM_SEED}_cfg.py --customise Configuration/DataProcessing/Utils.addMonitoring -n $NEVENTS

### run miniAOD step

cmsDriver.py step1 --filein file:tW_scattering_DRPremix_${RANDOM_SEED}.root --fileout file:tW_scattering_miniAOD_${RANDOM_SEED}.root --mc --eventcontent MINIAODSIM --runUnscheduled --datatier MINIAODSIM --conditions 102X_upgrade2018_realistic_v15 --step PAT --nThreads 1 --geometry DB:Extended --era Run2_2018 --python_filename tW_scattering_MAOD_${RANDOM_SEED}_cfg.py --customise Configuration/DataProcessing/Utils.addMonitoring -n $NEVENTS

### setup and run the nanoAOD step

export SCRAM_ARCH=slc6_amd64_gcc700
source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ -r CMSSW_10_2_15/src ] ; then
 echo release CMSSW_10_2_15 already exists
else
scram p CMSSW CMSSW_10_2_15
fi
cd CMSSW_10_2_15/src
eval `scram runtime -sh`

scram b
cd ../../

cmsDriver.py myNanoProdMc2018 --filein file:tW_scattering_miniAOD_${RANDOM_SEED}.root --fileout file:tW_scattering_nanoAOD_${RANDOM_SEED}.root -s NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM  --conditions 102X_upgrade2018_realistic_v19 --era Run2_2018,run2_nanoAOD_102Xv1 --python_filename tW_scattering_NAOD_${RANDOM_SEED}_cfg.py --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)))" -n $NEVENTS

### delete intermediate root files
rm -f tW_scattering_GS_${RANDOM_SEED}.root
rm -f tW_scattering_GS_${RANDOM_SEED}_inLHE.root
rm -f tW_scattering_step1_${RANDOM_SEED}.root
rm -f tW_scattering_DRPremix_${RANDOM_SEED}.root

### delete intermediate config files
rm -f tW_scattering_${RANDOM_SEED}_cfg.py
rm -f tW_scattering_1_${RANDOM_SEED}_cfg.py
rm -f tW_scattering_2_${RANDOM_SEED}_cfg.py
rm -f tW_scattering_MAOD_${RANDOM_SEED}_cfg.py
rm -f tW_scattering_NAOD_${RANDOM_SEED}_cfg.py

### copy outputs

gfal-copy -p -f -t 4200 tW_scattering_miniAOD_${RANDOM_SEED}.root gsiftp://gftp.t2.ucsd.edu/${OUTDIR}/miniAOD/tW_scattering_miniAOD_${RANDOM_SEED}.root --checksum ADLER32
gfal-copy -p -f -t 4200 tW_scattering_nanoAOD_${RANDOM_SEED}.root gsiftp://gftp.t2.ucsd.edu/${OUTDIR}/nanoAOD/tW_scattering_nanoAOD_${RANDOM_SEED}.root --checksum ADLER32


