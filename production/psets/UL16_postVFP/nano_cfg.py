# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: --filein file:miniAOD.root --fileout file:nanoAOD.root --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 106X_mcRun2_asymptotic_v13 --step NANO --era Run2_2016 --python_filename nano_cfg.py -n 10 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016

process = cms.Process('NANO',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:miniAOD.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('--filein nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAODSIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:nanoAOD.root'),
    outputCommands = process.NANOAODSIMEventContent.outputCommands
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_mcRun2_asymptotic_v13', '')

# Path and EndPath definitions
process.nanoAOD_step = cms.Path(process.nanoSequenceMC)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.nanoAOD_step,process.endjob_step,process.NANOAODSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.NanoAOD.nano_cff
from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeMC 

#call to customisation function nanoAOD_customizeMC imported from PhysicsTools.NanoAOD.nano_cff
process = nanoAOD_customizeMC(process)

named_weights = [
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_6p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_3p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_6p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_3p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_3p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_0p_cpQ3_6p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_3p_nlo",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_3p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_3p_cpQ3_3p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_0p_cpQM_6p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p_nlo",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p_nlo",
    "ctZ_0p_cpt_3p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_3p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_0p_cpt_6p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_3p_nlo",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_3p_ctp_0p_nlo",
    "ctZ_3p_cpt_0p_cpQM_0p_cpQ3_3p_ctW_0p_ctp_0p_nlo",
    "ctZ_3p_cpt_0p_cpQM_3p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_3p_cpt_3p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
    "ctZ_6p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo",
]

process.genWeightsTable.namedWeightIDs = named_weights
process.genWeightsTable.namedWeightLabels = named_weights

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
