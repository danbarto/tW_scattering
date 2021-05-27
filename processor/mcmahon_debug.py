from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
import pandas as pd
import numpy as np

# the below command will change to .from_root in coffea v0.7.0
# events = NanoEventsFactory.from_root('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_v2/nanoSkim_1.root', schemaclass=NanoAODSchema).events()

# events = NanoEventsFactory.from_root('root://xcache-redirector.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv7/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/70000/DE335891-829A-B943-99BE-E5A179F5F3EB.root', schemaclass=NanoAODSchema).events()
events = NanoEventsFactory.from_root('/hadoop/cms/store/user/ksalyer/FCNC_NanoSkim/fcnc_v3/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM_fcnc_v3/output_12.root', schemaclass=NanoAODSchema).events()
#
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *
from Tools.helpers import mt
from Tools.SS_selection import SS_selection
from Tools.fake_rate import fake_rate

#electron     = Collections(events, "Electron", "tightSSTTH").get()

## now do whatever you would have done in the processor

# we can use a very loose preselection to filter the events. nothing is done with this presel, though
presel = ak.num(events.Jet)>=0

ev = events[presel]
# load the config - probably not needed anymore
cfg = loadConfig()

muon         = Collections(ev, "Muon", "tightFCNC").get()
fakeablemuon = Collections(ev, "Muon", "fakeableFCNC").get()  

tight_muon_gen_prompt        = Collections(ev, "Muon", "tightFCNCGenPrompt", year=2016).get()
tight_muon_gen_nonprompt     = Collections(ev, "Muon", "tightFCNCGenNonprompt").get()
tight_electron_gen_prompt    = Collections(ev, "Electron", "tightFCNCGenPrompt").get()
tight_electron_gen_nonprompt = Collections(ev, "Electron", "tightFCNCGenNonprompt").get()
#nonprompt
loose_muon_gen_nonprompt     = Collections(ev, "Muon", "fakeableFCNCGenNonprompt").get()
loose_electron_gen_nonprompt = Collections(ev, "Electron", "fakeableFCNCGenNonprompt").get()

electron         = Collections(ev, "Electron", "tightFCNC").get()
fakeableelectron = Collections(ev, "Electron", "fakeableFCNC").get()

jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
#get loose leptons that are explicitly not tight
muon_orthogonality_param = ((ak.num(loose_muon_gen_nonprompt)==1) & (ak.num(tight_muon_gen_nonprompt)==0) | 
                            (ak.num(loose_muon_gen_nonprompt)==2) & (ak.num(tight_muon_gen_nonprompt)==1) )

electron_orthogonality_param = ((ak.num(loose_electron_gen_nonprompt)==1) & (ak.num(tight_electron_gen_nonprompt)==0) | 
                                (ak.num(loose_electron_gen_nonprompt)==2) & (ak.num(tight_electron_gen_nonprompt)==1) )

#clean jets :
# we want at least two jets that are outside of the lepton jets by deltaR > 0.4
jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
jet_sel = (ak.num(jets[~( match(jets, tight_muon_gen_prompt       , deltaRCut=0.4) | 
                          match(jets, tight_muon_gen_nonprompt    , deltaRCut=0.4) | 
                          match(jets, tight_electron_gen_prompt   , deltaRCut=0.4) | 
                          match(jets, tight_electron_gen_nonprompt, deltaRCut=0.4) | 
                         (match(jets, loose_muon_gen_nonprompt       , deltaRCut=0.4) & muon_orthogonality_param) | 
                         (match(jets, loose_electron_gen_nonprompt   , deltaRCut=0.4) & electron_orthogonality_param))])>=2)

dilepton = cross(muon, electron)
SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

two_lepton_sel = ( ak.num(tight_muon_gen_prompt)     + ak.num(tight_electron_gen_prompt)    + 
                   ak.num(tight_muon_gen_nonprompt)  + ak.num(tight_electron_gen_nonprompt) + 
                  (ak.num(loose_muon_gen_nonprompt)     - ak.num(tight_muon_gen_nonprompt))       +    #muon L!T counts
                  (ak.num(loose_electron_gen_nonprompt) - ak.num(tight_electron_gen_nonprompt)))  == 2 #electron L!T counts
 
num_leptons = ( ak.num(tight_muon_gen_prompt)        + ak.num(tight_electron_gen_prompt)     + 
                ak.num(tight_muon_gen_nonprompt)     + ak.num(tight_electron_gen_nonprompt)  + 
               (ak.num(loose_muon_gen_nonprompt)     - ak.num(tight_muon_gen_nonprompt))     +
               (ak.num(loose_electron_gen_nonprompt) - ak.num(tight_electron_gen_nonprompt))) 

#TT selection is two tight leptons, where one is a gen-level prompt, and the other is a gen-level nonprompt, so we should
#account for all of the possible lepton combinations below:
TT_selection = (SS_selection(tight_electron_gen_prompt, tight_muon_gen_nonprompt)     |
                SS_selection(tight_electron_gen_nonprompt, tight_muon_gen_prompt)     |
                SS_selection(tight_electron_gen_prompt, tight_electron_gen_nonprompt) | 
                SS_selection(tight_muon_gen_nonprompt, tight_muon_gen_prompt)         ) & two_lepton_sel & jet_sel
#SS_selection gives us all events that have a same sign pair of leptons coming from the provided two object collections

#TL selection is one tight lepton that is a gen-level prompt, and one loose (and NOT tight) lepton that is a gen-level nonprompt.
#The orthogonality_param is a hacky way to ensure that we are only looking at 2 lepton events that have a tight not loose lepton in the event
TL_selection = ((SS_selection(tight_electron_gen_prompt, loose_muon_gen_nonprompt)     & muon_orthogonality_param)     |
                (SS_selection(tight_muon_gen_prompt, loose_muon_gen_nonprompt)         & muon_orthogonality_param)     |
                (SS_selection(tight_electron_gen_prompt, loose_electron_gen_nonprompt) & electron_orthogonality_param) |
                (SS_selection(tight_muon_gen_prompt, loose_electron_gen_nonprompt)     & electron_orthogonality_param) ) & two_lepton_sel & jet_sel

"""Now We are making the different selections for the different regions. As a reminder, our SR is one tight gen-level prompt and one tight gen-level nonprompt, and our CR is
one tight gen-level prompt and one loose NOT tight gen-level nonprompt"""
#EE SR (Tight gen-level prompt e + Tight gen-level nonprompt e)
EE_SR_sel = SS_selection(tight_electron_gen_prompt, tight_electron_gen_nonprompt) & two_lepton_sel & jet_sel
#EE CR (Tight gen-level prompt e + L!T gen-level nonprompt e)
EE_CR_sel = (SS_selection(tight_electron_gen_prompt, loose_electron_gen_nonprompt) & electron_orthogonality_param) & two_lepton_sel & jet_sel

#MM SR (Tight gen-level prompt mu + Tight gen-level nonprompt mu)
MM_SR_sel = SS_selection(tight_muon_gen_nonprompt, tight_muon_gen_prompt)  & two_lepton_sel & jet_sel
#MM CR (Tight gen-level prompt mu + L!T gen-level nonprompt mu)
MM_CR_sel = (SS_selection(tight_muon_gen_prompt, loose_muon_gen_nonprompt) & muon_orthogonality_param) & two_lepton_sel & jet_sel

#EM SR (Tight gen-level prompt e + Tight gen-level nonprompt mu)
EM_SR_sel = SS_selection(tight_electron_gen_prompt, tight_muon_gen_nonprompt) & two_lepton_sel & jet_sel
#EM_CR (Tight gen-level prompt e + L!T gen-level nonprompt mu)
EM_CR_sel = (SS_selection(tight_electron_gen_prompt, loose_muon_gen_nonprompt) & muon_orthogonality_param) & two_lepton_sel & jet_sel

#ME SR (Tight gen-level prompt mu + Tight gen-level nonprompt e)
breakpoint()
ME_SR_sel = SS_selection(tight_electron_gen_nonprompt, tight_muon_gen_prompt) & two_lepton_sel & jet_sel
#ME CR (Tight gen-level prompt mu + L!T gen-level nonprompt e)
ME_CR_sel = (SS_selection(tight_muon_gen_prompt, loose_electron_gen_nonprompt) & electron_orthogonality_param) & two_lepton_sel & jet_sel

debug_sel = ME_SR_sel
print("length of debug selection: {}".format(len(debug_sel[debug_sel])))
breakpoint()

electron_2018 = fake_rate("../data/fake_rate/FR_electron_2018.p")
electron_2016 = fake_rate("../data/fake_rate/FR_electron_2016.p")
muon_2018 = fake_rate("../data/fake_rate/FR_muon_2018.p")
muon_2016 = fake_rate("../data/fake_rate/FR_muon_2016.p")

weight_muon = muon_2018.FR_weight(loose_muon_gen_nonprompt)
weight_electron = electron_2018.FR_weight(loose_electron_gen_nonprompt)




print(tight_muon_gen_nonprompt[debug_sel].eta)
