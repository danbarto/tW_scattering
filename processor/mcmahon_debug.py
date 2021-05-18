from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
import pandas as pd
import numpy as np

# the below command will change to .from_root in coffea v0.7.0
# events = NanoEventsFactory.from_root('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_v2/nanoSkim_1.root', schemaclass=NanoAODSchema).events()

# events = NanoEventsFactory.from_root('root://xcache-redirector.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv7/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/70000/DE335891-829A-B943-99BE-E5A179F5F3EB.root', schemaclass=NanoAODSchema).events()
events = NanoEventsFactory.from_root('/hadoop/cms/store/user/ksalyer/FCNC_NanoSkim/fcnc_v3/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM_fcnc_v3/output_2.root', schemaclass=NanoAODSchema).events()
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

#electron     = Collections(events, "Electron", "tightSSTTH").get()

## now do whatever you would have done in the processor

# we can use a very loose preselection to filter the events. nothing is done with this presel, though
presel = ak.num(events.Jet)>=0

ev = events[presel]
# load the config - probably not needed anymore
cfg = loadConfig()

muon         = Collections(ev, "Muon", "tightFCNC").get()
fakeablemuon = Collections(ev, "Muon", "fakeableFCNC").get()  

tight_muon_gen_prompt        = Collections(ev, "Muon", "tightFCNCGenPrompt").get()
tight_muon_gen_nonprompt     = Collections(ev, "Muon", "tightFCNCGenNonprompt").get()
tight_electron_gen_prompt    = Collections(ev, "Electron", "tightFCNCGenPrompt").get()
tight_electron_gen_nonprompt = Collections(ev, "Electron", "tightFCNCGenNonprompt").get()

loose_muon_gen_prompt     = Collections(ev, "Muon", "fakeableFCNCGenPrompt").get()
loose_electron_gen_prompt = Collections(ev, "Electron", "fakeableFCNCGenPrompt").get()

electron         = Collections(ev, "Electron", "tightFCNC").get()
fakeableelectron = Collections(ev, "Electron", "fakeableFCNC").get()

##Jets
Jets = events.Jet

breakpoint()
Te_Tu_Selection = SS_selection(tight_electron_gen_prompt, tight_muon_gen_prompt)


## MET -> can switch to puppi MET
met_pt  = ev.MET.pt
met_phi = ev.MET.phi

#get loose leptons that are explicitly not tight
loose_muon_gen_prompt_orthogonal = loose_muon_gen_prompt[(ak.num(loose_muon_gen_prompt)==1) & (ak.num(tight_muon_gen_prompt)==0) | 
                                                         (ak.num(loose_muon_gen_prompt)==2) & (ak.num(tight_muon_gen_prompt)==1) ]

loose_electron_gen_prompt_orthogonal = loose_muon_gen_prompt[(ak.num(loose_electron_gen_prompt)==1) & (ak.num(tight_electron_gen_prompt)==0) | 
                                                             (ak.num(loose_electron_gen_prompt)==2) & (ak.num(tight_electron_gen_prompt)==1) ]


#clean jets :
# we want at least two jets that are outside of the lepton jets by deltaR > 0.4
jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
jet_sel = (ak.num(jets[~(match(jets, tight_muon_gen_prompt, deltaRCut=0.4) | 
                         match(jets, tight_muon_gen_nonprompt, deltaRCut=0.4) | 
                         match(jets, tight_electron_gen_prompt, deltaRCut=0.4) | 
                         match(jets, tight_electron_gen_nonprompt, deltaRCut=0.4) | 
                         match(jets, loose_muon_gen_prompt_orthogonal, deltaRCut=0.4) | 
                         match(jets, loose_electron_gen_prompt_orthogonal, deltaRCut=0.4))])>=2)

dilepton = cross(muon, electron)
SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

two_lepton_sel = (ak.num(tight_muon_gen_prompt) + ak.num(tight_electron_gen_prompt) + ak.num(tight_muon_gen_nonprompt) + ak.num(tight_electron_gen_nonprompt)) == 2
Te_Te_sel = (ak.num(tight_electron_gen_prompt) == 1)  & (ak.num(tight_electron_gen_nonprompt) == 1) & two_lepton_sel & jet_sel
Tu_Tu_sel = (ak.num(tight_muon_gen_prompt) == 1)      & (ak.num(tight_muon_gen_nonprompt) == 1)     & two_lepton_sel & jet_sel
Te_Lu_sel = (ak.num(tight_electron_gen_prompt) == 1)  & (ak.num(loose_muon_gen_prompt) == 1)        & two_lepton_sel & jet_sel
Tu_Le_sel = (ak.num(tight_muon_gen_prompt) == 1)      & (ak.num(loose_electron_gen_prompt) == 1)    & two_lepton_sel & jet_sel

#modify this to clean the jets for the measurement region
#& (ak.num(jets[~match(jets, fakeablemuon, deltaRCut=1.0)])>=1)


pt_muon_Tu_Tu  = ak.to_numpy(ak.flatten(fakeablemuon[Tu_Tu_sel].conePt))
eta_muon_Tu_Tu = ak.to_numpy(ak.flatten(fakeablemuon[Tu_Tu_sel].eta))

pt_electron_Te_Te  = ak.to_numpy(ak.flatten(fakeableelectron[Te_Te_sel].conePt))
eta_electron_Te_Te = ak.to_numpy(ak.flatten(fakeableelectron[Te_Te_sel].eta))

pt_muon_Tu_Le = ak.to_numpy(ak.flatten(fakeablemuon[Tu_Le_sel].conePt))
eta_muon_Tu_Le = ak.to_numpy(ak.flatten(fakeablemuon[Tu_Le_sel].eta))

#SS conePt
# I_1 = 0.11; I_2 = 0.74; I_3 = 6.8
# floor_miniIso = (ev.Muon.miniPFRelIso_all - I_1) * ((ev.Muon.miniPFRelIso_all - I_1) > 0) #equivalent to max(0, ev.Muon.miniPFRelIso_all - I_1)
# PF_unflatten = ak.from_regular(ev.Muon.miniPFRelIso_all[:,:,np.newaxis])
# max_miniIso = ak.max(ak.concatenate([PF_unflatten - I_1, ak.zeros_like(PF_unflatten)], axis=2), axis=2) #equivalent to max(0, ev.Muon.miniPFRelIso_all - I_1)
# muon_pt_unflatten = ak.from_regular(ev.Muon.pt[:,:,np.newaxis])
# jet_pt_unflatten = ak.from_regular(ev.Muon.matched_jet.pt[:,:,np.newaxis])
# max_pt = ak.max(ak.concatenate([muon_pt_unflatten, jet_pt_unflatten * I_2], axis=2), axis=2) #max(ev.Muon.pt, ev.Muon.matched_jet.pt * I_2)
# conePt = (ev.Muon.pt*(1 + max_miniIso)) * (ev.Muon.jetPtRelv2 > I_3) + (max_pt * ~(ev.Muon.jetPtRelv2 > I_3))

# mask_close = (ak.fill_none(ev.Muon.delta_r(ev.Muon.matched_jet),99)<0.4)*1
# mask_far = ~(ak.fill_none(ev.Muon.delta_r(ev.Muon.matched_jet),99)<0.4)*1

# jetRelIsoV2 = ev.Muon.jetRelIso*mask_close + ev.Muon.pfRelIso03_all*mask_far  # default to 0 if no match