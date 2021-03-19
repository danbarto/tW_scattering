#! /usr/bin/env python
# Original Author: Izaak Neutelings (Februari, 2020)
# Description: Standalone to dump gen particle information
# Instructions: Install nanoAOD-tools and run
#   python dumpLHE.py -n 10
# Sources:
#   https://github.com/cms-nanoAOD/nanoAOD-tools#nanoaod-tools
#   https://github.com/cms-nanoAOD/nanoAOD-tools/tree/master/python/postprocessing/examples
#   https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/genparticles_cff.py
#   https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/plugins/LHETablesProducer.cc
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i', '--infiles', dest='infiles', action='store', type=str, default=None)
parser.add_argument('-n', '--max',     dest='maxEvts', action='store', type=int, default=20)
args = parser.parse_args()

## the following needs: https://github.com/scikit-hep/particle
from particle import Particle

# SETTINGS
outdir    = '.'
maxEvts   = args.maxEvts if args.maxEvts>0 else 9999999999
branchsel = None

# INPUT FILES
#  dasgoclient --query="dataset=/DYJetsToLL_M-50*/*18NanoAODv5*/NANOAOD*"
#  dasgoclient --query="dataset=/DYJetsToLL_M-50_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18NanoAODv5-PUFall18Fast_Nano1June2019_lhe_102X_upgrade2018_realistic_v19-v1/NANOAODSIM file" | head -n10
director  = "root://cms-xrd-global.cern.ch/" #"root://xrootd-cms.infn.it/"
infiles   = [
  #director+'/store/mc/RunIIAutumn18NanoAODv5/DYJetsToLL_M-50_TuneCP2_13TeV-madgraphMLM-pythia8/NANOAODSIM/PUFall18Fast_Nano1June2019_lhe_102X_upgrade2018_realistic_v19-v1/250000/9A3D4107-5366-C243-915A-F4426F464D2F.root',
  #'/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/tW_scattering_nanoAOD_100.root'
  #director + '/store/mc/RunIIFall17NanoAODv7/WminusH_HToBB_WToLNu_M125_13TeV_powheg_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/70000/AC066AE4-C6E2-C245-9F85-D017D83507EB.root'
  #'/hadoop/cms/store/user/mbryson/WH_hadronic/WH_had_750_1/test/WH_hadronic_nanoAOD_500.root'
  'root://xrootd.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv6/W1JetsToLNu_NuPt-200_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/250000/3917F723-A4EF-AF42-9CA9-D5358FF660CB.root'
  #'/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/tW_scattering_nanoAOD_500.root'
]
if args.infiles:
  infiles = [args.infiles]

# HAS BIT
def hasBit(value,bit):
  """Check if i'th bit is set to 1, i.e. binary of 2^(i-1),
  from the right to the left, starting from position i=0."""
  # https://cms-nanoaod-integration.web.cern.ch/integration/master-102X/mc102X_doc.html#GenPart
  # Gen status flags, stored bitwise, are:
  #    0: isPrompt,                          8: fromHardProcess,
  #    1: isDecayedLeptonHadron,             9: isHardProcessTauDecayProduct,
  #    2: isTauDecayProduct,                10: isDirectHardProcessTauDecayProduct,
  #    3: isPromptTauDecayProduct,          11: fromHardProcessBeforeFSR,
  #    4: isDirectTauDecayProduct,          12: isFirstCopy,
  #    5: isDirectPromptTauDecayProduct,    13: isLastCopy,
  #    6: isDirectHadronDecayProduct,       14: isLastCopyBeforeFSR
  #    7: isHardProcess,
  ###return bin(value)[-bit-1]=='1'
  ###return format(value,'b').zfill(bit+1)[-bit-1]=='1'
  return (value & (1 << bit))>0

# DUMPER MODULE
class LHEDumper(Module):
  
  def __init__(self):
    self.nleptons = 0
    self.nevents  = 0
 
  def hasAncestor(self, p, ancestorPdg, genParts):
    motherIdx = p.genPartIdxMother
    while motherIdx>0:
      if (abs(genParts[motherIdx].pdgId) == ancestorPdg): return True
      motherIdx = genParts[motherIdx].genPartIdxMother
    return False
 
  def analyze(self,event):
    """Dump LHE information for each gen particle in given event."""
    print "%s event %s %s"%('-'*10,event.event,'-'*50)
    self.nevents += 1
    leptonic = False
    particles = Collection(event,'GenPart')
    #particles = Collection(event,'LHEPart')
    print " \033[4m%7s %8s %10s %8s %8s %10s %8s %8s %8s %9s %10s %11s %11s \033[0m"%(
      "index","pdgId","particle","moth","mothid", "moth part", "dR","pt","status","prompt","last copy", "hard scatter", "W ancestor")
    for i, particle in enumerate(particles):
      mothidx  = particle.genPartIdxMother
      if 0<=mothidx<len(particles):
        moth    = particles[mothidx]
        mothpid = moth.pdgId
        mothdR  = min(10,particle.DeltaR(moth)) #particle.p4().DeltaR(moth.p4())
      else:
        mothpid = 0
        mothdR  = -1
      prompt    = hasBit(particle.statusFlags,0)
      lastcopy  = hasBit(particle.statusFlags,13)
      hardprocess = hasBit(particle.statusFlags,7)
      hasWancestor = (self.hasAncestor( particle, 24, particles) and abs(particle.pdgId)<5)
      try:
          particleName =  Particle.from_pdgid(int(particle.pdgId)).name
      except:
          particleName = str(particle.pdgId)
      try:
        motherName = Particle.from_pdgid(int(mothpid)).name if mothpid != 0 else 'initial'
      except:
          particleName = str(particle.pdgId)
      print " %7d %8d %10s %8d %8d %10s %8.2f %8.2f %8d %9s %10s %11s %11s"%(
        i,particle.pdgId,particleName,mothidx,mothpid,motherName,mothdR,particle.pt,particle.status,prompt,lastcopy,hardprocess, hasWancestor)
      if abs(particle.pdgId) in [11,13,15]:
        leptonic = True
    if leptonic:
      self.nleptons += 1
    
  def endJob(self):
    print '-'*70
    if self.nevents>0:
      print "  %-10s %4d / %-4d (%.1f%%)"%('leptonic:',self.nleptons,self.nevents,100.0*self.nleptons/self.nevents)
    print "%s done %s"%('-'*10,'-'*54)
  
# PROCESS NANOAOD
#filterEvent = 'event==606||event==352'
#filterEvent = 'event==1'
filterEvent = '(1)'
processor = PostProcessor(outdir,infiles,noOut=True,cut=filterEvent,modules=[LHEDumper()],maxEntries=maxEvts)
processor.run()
