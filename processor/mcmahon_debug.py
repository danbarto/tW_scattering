from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea import processor, hist
import pandas as pd
import numpy as np

# the below command will change to .from_root in coffea v0.7.0
# events = NanoEventsFactory.from_root('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_v2/nanoSkim_1.root', schemaclass=NanoAODSchema).events()

# events = NanoEventsFactory.from_root('root://xcache-redirector.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv7/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/70000/DE335891-829A-B943-99BE-E5A179F5F3EB.root', schemaclass=NanoAODSchema).events()
events = NanoEventsFactory.from_root('/hadoop/cms/store/user/ksalyer/FCNC_NanoSkim/fcnc_v3/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8_RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1_NANOAODSIM_fcnc_v3/output_40.root', schemaclass=NanoAODSchema).events()
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
from processor.default_accumulators import desired_output, add_processes_to_output, dataset_axis, pt_axis, eta_axis

#electron     = Collections(events, "Electron", "tightSSTTH").get()

## now do whatever you would have done in the processor

def SS_fill_weighted(output, mumu_sel, ee_sel, mue_sel, emu_sel, mu_weights=None, e_weights=None, **kwargs):
    if len(kwargs.keys())==3: #dataset, axis_1, axis_2
        vals_1 = np.array([])
        vals_2 = np.array([])
        weights = np.array([])
        for sel in [mumu_sel, emu_sel]:
            vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
            vals_2 = np.concatenate((vals_2, list(kwargs.values())[2][sel]))
            if mu_weights == None:
                tmp_mu_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                weights = np.concatenate((weights, tmp_mu_weights))
            else:
                weights = np.concatenate((weights, mu_weights[sel]))
        for sel in [ee_sel, mue_sel]:
            vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
            vals_2 = np.concatenate((vals_2, list(kwargs.values())[2][sel]))
            if e_weights == None:
                tmp_e_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                weights = np.concatenate((weights, tmp_e_weights))
            else:
                weights = np.concatenate((weights, e_weights[sel]))
        return_dict = kwargs
        return_dict[list(kwargs.keys())[1]] = vals_1
        return_dict[list(kwargs.keys())[2]] = vals_2

    elif len(kwargs.keys())==2: #dataset, axis_1
        vals_1 = np.array([])
        weights = np.array([])
        for sel in [mumu_sel, emu_sel]:
            vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
            if mu_weights == None:
                tmp_mu_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                weights = np.concatenate((weights, tmp_mu_weights))
            else:
                weights = np.concatenate((weights, mu_weights[sel]))
        for sel in [ee_sel, mue_sel]:
            vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
            if e_weights == None:
                tmp_e_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                weights = np.concatenate((weights, tmp_e_weights))
            else:
                weights = np.concatenate((weights, e_weights[sel]))
        return_dict = kwargs
        return_dict[list(kwargs.keys())[1]] = vals_1

    #fill the histogram
    output.fill(**return_dict, weight=weights)


presel = ak.num(events.Jet)>=0

ev = events[presel]
output = processor.dict_accumulator(desired_output).identity()
##Jets
Jets = events.Jet

## MET -> can switch to puppi MET
met_pt  = ev.MET.pt
met_phi = ev.MET.phi

 ### For FCNC, we want electron -> tightTTH
ele_t = Collections(ev, "Electron", "tightFCNC", year=2016).get()
ele_t_p = ele_t[((ele_t.genPartFlav==1) | (ele_t.genPartFlav==15))]
ele_t_np = ele_t[((ele_t.genPartFlav!=1) & (ele_t.genPartFlav!=15))]

ele_l = Collections(ev, "Electron", "fakeableFCNC", year=2016).get()
ele_l_p = ele_l[((ele_l.genPartFlav==1) | (ele_l.genPartFlav==15))]
ele_l_np = ele_l[((ele_l.genPartFlav!=1) & (ele_l.genPartFlav!=15))]

mu_t         = Collections(ev, "Muon", "tightFCNC", year=2016).get()
mu_t_p = mu_t[((mu_t.genPartFlav==1) | (mu_t.genPartFlav==15))]
mu_t_np = mu_t[((mu_t.genPartFlav!=1) & (mu_t.genPartFlav!=15))]

mu_l = Collections(ev, "Muon", "fakeableFCNC", year=2016).get()
mu_l_p = mu_l[((mu_l.genPartFlav==1) | (mu_l.genPartFlav==15))]
mu_l_np = mu_l[((mu_l.genPartFlav!=1) & (mu_l.genPartFlav!=15))]

#clean jets :
# we want at least two jets that are outside of the lepton jets by deltaR > 0.4
jets = getJets(ev, maxEta=2.4, minPt=40, pt_var='pt')
jet_sel = (ak.num(jets[~(match(jets, ele_l, deltaRCut=0.4) | match(jets, mu_l, deltaRCut=0.4))])>=2)

"""Now We are making the different selections for the different regions. As a reminder, our SR is one tight gen-level prompt and one tight gen-level nonprompt, and our CR is
one tight gen-level prompt and one loose NOT tight gen-level nonprompt"""

mumu_SR = ak.concatenate([mu_t_p, mu_t_np], axis=1)
mumu_SR_SS = (ak.sum(mumu_SR.charge, axis=1)!=0)
mumu_SR_sel = (ak.num(mu_t_p)==1) & (ak.num(mu_t_np)==1) & (ak.num(mu_l)==2) & jet_sel & mumu_SR_SS & (ak.num(mumu_SR[mumu_SR.pt>20])>1) & (ak.num(ele_l)==0)

mumu_CR = ak.concatenate([mu_t_p, mu_l_np], axis=1)
mumu_CR_SS = (ak.sum(mumu_CR.charge, axis=1)!=0)
mumu_CR_sel = (ak.num(mu_t_p)==1) & (ak.num(mu_l_np)==1) & (ak.num(mu_l)==2) & jet_sel & mumu_CR_SS & (ak.num(mumu_CR[mumu_CR.pt>20])>1) & (ak.num(ele_l)==0)

ee_SR = ak.concatenate([ele_t_p, ele_t_np], axis=1)
ee_SR_SS = (ak.sum(ee_SR.charge, axis=1)!=0)
ee_SR_sel = (ak.num(ele_t_p)==1) & (ak.num(ele_t_np)==1) & (ak.num(ele_l)==2) & jet_sel & ee_SR_SS & (ak.num(ee_SR[ee_SR.pt>20])>1) & (ak.num(mu_l)==0)

ee_CR = ak.concatenate([ele_t_p, ele_l_np], axis=1)
ee_CR_SS = (ak.sum(ee_CR.charge, axis=1)!=0)
ee_CR_sel = (ak.num(ele_t_p)==1) & (ak.num(ele_l_np)==1) & (ak.num(ele_l)==2) & jet_sel & ee_CR_SS & (ak.num(ee_CR[ee_CR.pt>20])>1) & (ak.num(mu_l)==0)

mue_SR = ak.concatenate([mu_t_p, ele_t_np], axis=1)
mue_SR_SS = (ak.sum(mue_SR.charge, axis=1)!=0)
mue_SR_sel = (ak.num(mu_t_p)==1) & (ak.num(ele_t_np)==1) & (ak.num(ele_l)==1) & jet_sel & mue_SR_SS & (ak.num(mue_SR[mue_SR.pt>20])>1) & (ak.num(mu_l)==1)

mue_CR = ak.concatenate([mu_t_p, ele_l_np], axis=1)
mue_CR_SS = (ak.sum(mue_CR.charge, axis=1)!=0)
mue_CR_sel = (ak.num(mu_t_p)==1) & (ak.num(ele_l_np)==1) & (ak.num(ele_l)==1) & jet_sel & mue_CR_SS & (ak.num(mue_CR[mue_CR.pt>20])>1) & (ak.num(mu_l)==1)

emu_SR = ak.concatenate([ele_t_p, mu_t_np], axis=1)
emu_SR_SS = (ak.sum(emu_SR.charge, axis=1)!=0)
emu_SR_sel = (ak.num(ele_t_p)==1) & (ak.num(mu_t_np)==1) & (ak.num(mu_l)==1) & jet_sel & emu_SR_SS & (ak.num(emu_SR[emu_SR.pt>20])>1) & (ak.num(ele_l)==1)

emu_CR = ak.concatenate([ele_t_p, mu_l_np], axis=1)
emu_CR_SS = (ak.sum(emu_CR.charge, axis=1)!=0)
emu_CR_sel = (ak.num(ele_t_p)==1) & (ak.num(mu_l_np)==1) & (ak.num(mu_l)==1) & jet_sel & emu_CR_SS & (ak.num(emu_CR[emu_CR.pt>20])>1) & (ak.num(ele_l)==1)

#combine all selections for generic CR and SR
CR_sel = mumu_CR_sel | ee_CR_sel | mue_CR_sel | emu_CR_sel
SR_sel = mumu_SR_sel | ee_SR_sel | mue_SR_sel | emu_SR_sel

electron_2018 = fake_rate("../data/fake_rate/FR_electron_2018.p")
electron_2017 = fake_rate("../data/fake_rate/FR_electron_2017.p")
electron_2016 = fake_rate("../data/fake_rate/FR_electron_2016.p")
muon_2018 = fake_rate("../data/fake_rate/FR_muon_2018.p")
muon_2017 = fake_rate("../data/fake_rate/FR_muon_2017.p")
muon_2016 = fake_rate("../data/fake_rate/FR_muon_2016.p")

weight_muon = muon_2016.FR_weight(mu_l_np)
weight_electron = electron_2016.FR_weight(ele_l_np)
breakpoint()
#fill combined histograms now (basic definitions are in default_accumulators.py)
SS_fill_weighted(output["MET"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset="debug",  pt=ev.MET.pt, phi=ev.MET.phi)
print(ak.max(ak.concatenate([ev.Muon.pt, ev.Electron.pt], axis=2), axis=2))