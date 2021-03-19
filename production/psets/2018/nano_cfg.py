# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: myNanoProdMc_NanoGEN -s NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --no_exec --conditions auto:run2_mc
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('NANO')

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
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:GEN_102X.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('myNanoProdMc_NanoGEN nevts:1'),
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
    fileName = cms.untracked.string('output.root'),
    outputCommands = process.NANOAODSIMEventContent.outputCommands
)

# Additional output definition

## gen stuff
#genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
#    genEvent = cms.InputTag("generator"),
#    genLumiInfoHeader = cms.InputTag("generator"),
#    lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
#    preferredPDFs = cms.VPSet( # see https://lhapdf.hepforge.org/pdfsets.html
#        cms.PSet( name = cms.string("PDF4LHC15_nnlo_30_pdfas"), lhaid = cms.uint32(91400) ),
#        cms.PSet( name = cms.string("NNPDF31_nnlo_hessian_pdfas"), lhaid = cms.uint32(306000) ),
#        cms.PSet( name = cms.string("NNPDF30_nlo_as_0118"), lhaid = cms.uint32(260000) ), # for some 92X samples. Note that the nominal weight, 260000, is not included in the LHE ...
#        cms.PSet( name = cms.string("NNPDF30_lo_as_0130"), lhaid = cms.uint32(262000) ), # some MLM 80X samples have only this (e.g. /store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/02A210D6-F5C3-E611-B570-008CFA197BD4.root )
#        cms.PSet( name = cms.string("NNPDF30_nlo_nf_4_pdfas"), lhaid = cms.uint32(292000) ), # some FXFX 80X samples have only this (e.g. WWTo1L1Nu2Q, WWTo4Q)
#        cms.PSet( name = cms.string("NNPDF30_nlo_nf_5_pdfas"), lhaid = cms.uint32(292200) ), # some FXFX 80X samples have only this (e.g. DYJetsToLL_Pt, WJetsToLNu_Pt, DYJetsToNuNu_Pt)
#    ),
#    namedWeightIDs = cms.vstring(),
#    namedWeightLabels = cms.vstring(),
#    lheWeightPrecision = cms.int32(14),
#    maxPdfWeights = cms.uint32(150),
#    debug = cms.untracked.bool(False),
#)
#lheInfoTable = cms.EDProducer("LHETablesProducer",
#    lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
#    precision = cms.int32(14),
#    storeLHEParticles = cms.bool(True)
#)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Hacking at nanoSequenceMC from PhysicsTools.NanoAOD.nano_cff
process.jetMC.remove(process.jetMCTable)
process.nanogenSequence = cms.Sequence(
    process.genParticleSequence+
    process.genParticles2HepMC+
    process.particleLevel+
    process.genJetTable+
    process.patJetPartons+
    process.genJetFlavourAssociation+
    process.genJetFlavourTable+
    process.genJetAK8Table+
    process.genJetAK8FlavourAssociation+
    process.genJetAK8FlavourTable+
    process.tauGenJets+
    process.tauGenJetsSelectorAllHadrons+
    process.genVisTaus+
    process.genVisTauTable+
    process.genTable+
    process.genWeightsTable+
    process.genParticleTables+
    #process.particleLevelTables+
    process.lheInfoTable
)
process.finalGenParticles.src = cms.InputTag("genParticles")
process.genParticles2HepMC.genParticles  = cms.InputTag("genParticles")
process.patJetPartons.particles = cms.InputTag("genParticles")

process.genJetTable.src = cms.InputTag("ak4GenJetsNoNu")
process.genJetFlavourAssociation.jets = process.genJetTable.src
process.genJetFlavourTable.src = process.genJetTable.src
process.genJetFlavourTable.jetFlavourInfos = cms.InputTag("genJetFlavourAssociation")
process.genJetAK8Table.src = cms.InputTag("ak8GenJetsNoNu")
process.genJetAK8FlavourAssociation.jets = process.genJetAK8Table.src
process.genJetAK8FlavourTable.src = process.genJetAK8Table.src
process.tauGenJets.GenParticles = cms.InputTag("genParticles")
process.genVisTaus.srcGenParticles = cms.InputTag("genParticles")

# Path and EndPath definitions
process.nanoAOD_step = cms.Path(process.nanogenSequence)
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

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
