# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/tW_scattering.py --python_filename topW_EFT_cfg.py --eventcontent NANOAODGEN --customise Configuration/DataProcessing/Utils.addMonitoring --datatier NANOAOD --fileout file:topW_EFT.root --conditions auto:mc --beamspot Realistic25ns13TeVEarly2018Collision --step LHE,GEN,NANOGEN --geometry DB:Extended --era Run2_2018 --no_exec --mc -n 100
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process('NANOGEN',Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2018Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('PhysicsTools.NanoAOD.nanogen_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

maxN = 10

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxN)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('Configuration/GenProduction/python/tW_scattering.py nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.NANOAODGENoutput = cms.OutputModule("NanoAODOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAOD'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:output.root'),
    outputCommands = process.NANOAODGENEventContent.outputCommands
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring(
            'pythia8CommonSettings', 
            'pythia8CP5Settings', 
            'pythia8aMCatNLOSettings', 
            'processParameters'
        ),
        processParameters = cms.vstring('TimeShower:nPartonsInBorn = 0'),
        pythia8CP5Settings = cms.vstring(
            'Tune:pp 14', 
            'Tune:ee 7', 
            'MultipartonInteractions:ecmPow=0.03344', 
            'MultipartonInteractions:bProfile=2', 
            'MultipartonInteractions:pT0Ref=1.41', 
            'MultipartonInteractions:coreRadius=0.7634', 
            'MultipartonInteractions:coreFraction=0.63', 
            'ColourReconnection:range=5.176', 
            'SigmaTotal:zeroAXB=off', 
            'SpaceShower:alphaSorder=2', 
            'SpaceShower:alphaSvalue=0.118', 
            'SigmaProcess:alphaSvalue=0.118', 
            'SigmaProcess:alphaSorder=2', 
            'MultipartonInteractions:alphaSvalue=0.118', 
            'MultipartonInteractions:alphaSorder=2', 
            'TimeShower:alphaSorder=2', 
            'TimeShower:alphaSvalue=0.118', 
            'SigmaTotal:mode = 0', 
            'SigmaTotal:sigmaEl = 21.89', 
            'SigmaTotal:sigmaTot = 100.309', 
            'PDF:pSet=LHAPDF6:NNPDF31_nnlo_as_0118'
        ),
        pythia8CommonSettings = cms.vstring(
            'Tune:preferLHAPDF = 2', 
            'Main:timesAllowErrors = 10000', 
            'Check:epTolErr = 0.01', 
            'Beams:setProductionScalesFromLHEF = off', 
            'SLHA:keepSM = on', 
            'SLHA:minMassSM = 1000.', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tau0Max = 10', 
            'ParticleDecays:allowPhotonRadiation = on'
        ),
        pythia8aMCatNLOSettings = cms.vstring(
            'SpaceShower:pTmaxMatch = 1', 
            'SpaceShower:pTmaxFudge = 1', 
            'SpaceShower:MEcorrections = off', 
            'TimeShower:pTmaxMatch = 1', 
            'TimeShower:pTmaxFudge = 1', 
            'TimeShower:MEcorrections = off', 
            'TimeShower:globalRecoil = on', 
            'TimeShower:limitPTmaxGlobal = on', 
            'TimeShower:nMaxGlobalRecoil = 1', 
            'TimeShower:globalRecoilMode = 2', 
            'TimeShower:nMaxGlobalBranch = 1', 
            'TimeShower:weightGluonToQuark = 1'
        )
    ),
    comEnergy = cms.double(13000.0),
    filterEfficiency = cms.untracked.double(1.0),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)


process.externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    #args = cms.vstring('/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks//TTWJetsToLNuEWK_5f_EFT_myNLO_cpq3_4_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz'),
    #args = cms.vstring('/home/users/dspitzba/TTW/MG_test/genproductions/bin/MadGraph5_aMCatNLO/WORKING/TTWJetsToLNuEWK_5f_EFT_myNLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'),
    #args = cms.vstring('/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_NLO_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz'), # SM with madspin
    #args = cms.vstring('/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsEWK_5f_NLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'), # SM without madspin
    #args = cms.vstring('/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'), 
    #args = cms.vstring('/home/users/dspitzba/TTW/MG_test/genproductions/bin/MadGraph5_aMCatNLO/TTZJetsToLL_5f_NLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'), 
    args = cms.vstring('/home/users/dspitzba/TTW/MG_SMEFTNLO103/genproductions/bin/MadGraph5_aMCatNLO/TTll_5f_LO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz'), 
    nEvents = cms.untracked.uint32(maxN),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.lhe_step = cms.Path(process.externalLHEProducer)
process.generation_step = cms.Path(process.pgen)
process.nanoAOD_step = cms.Path(process.nanogenSequence)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.NANOAODGENoutput_step = cms.EndPath(process.NANOAODGENoutput)

# Schedule definition
process.schedule = cms.Schedule(process.lhe_step,process.generation_step,process.genfiltersummary_step,process.nanoAOD_step,process.endjob_step,process.NANOAODGENoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
	if path in ['lhe_step']: continue
	getattr(process,path).insert(0, process.ProductionFilterSequence)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.NanoAOD.nanogen_cff
from PhysicsTools.NanoAOD.nanogen_cff import customizeNanoGEN 

#call to customisation function customizeNanoGEN imported from PhysicsTools.NanoAOD.nanogen_cff
process = customizeNanoGEN(process)

###################
# Storing weights #
###################

## postfix _nlo for NLO samples, nothing for LO samples

named_weights = [
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_6p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_3p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_6p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_3p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_3p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_6p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_3p",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_3p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_3p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_0p_cpQM_6p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_3p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_0p_cpt_6p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p",
    "ctZ_3p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_3p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
    "ctZ_6p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p",
]

process.genWeightsTable.namedWeightIDs = named_weights
process.genWeightsTable.namedWeightLabels = named_weights

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)


process.options.numberOfThreads=cms.untracked.uint32(12)
# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
