import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd
from yahist import Hist1D, Hist2D

# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *
from Tools.helpers import mt
from Tools.fake_rate import fake_rate
from Tools.SS_selection import SS_selection

class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}, debug=False):
        self.variations = variations
        self.year = year
        self.debug = debug
        self.btagSF = btag_scalefactor(year)
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator
    
    def SS_fill_weighted(self, output, mumu_sel, ee_sel, mue_sel, emu_sel, mu_weights=None, e_weights=None, **kwargs):
        if len(kwargs.keys())==3: #dataset, axis_1, axis_2
            vals_1 = np.array([])
            vals_2 = np.array([])
            weights = np.array([])
            for sel in [mumu_sel, emu_sel]:
                vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
                vals_2 = np.concatenate((vals_2, list(kwargs.values())[2][sel]))
                if type(mu_weights) != ak.highlevel.Array:
                    tmp_mu_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                    weights = np.concatenate((weights, tmp_mu_weights))
                else:
                    weights = np.concatenate((weights, mu_weights[sel]))
            for sel in [ee_sel, mue_sel]:
                vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
                vals_2 = np.concatenate((vals_2, list(kwargs.values())[2][sel]))
                if type(e_weights) != ak.highlevel.Array:
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
                if type(mu_weights) != ak.highlevel.Array:
                    tmp_mu_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                    weights = np.concatenate((weights, tmp_mu_weights))
                else:
                    weights = np.concatenate((weights, mu_weights[sel]))
            for sel in [ee_sel, mue_sel]:
                vals_1 = np.concatenate((vals_1, list(kwargs.values())[1][sel]))
                if type(e_weights) != ak.highlevel.Array:
                    tmp_e_weights = np.ones_like(kwargs[list(kwargs.keys())[1]][sel])
                    weights = np.concatenate((weights, tmp_e_weights))
                else:
                    weights = np.concatenate((weights, e_weights[sel]))
            return_dict = kwargs
            return_dict[list(kwargs.keys())[1]] = vals_1
        
        #fill the histogram
        output.fill(**return_dict, weight=weights)

    def fill_pt_individual(self, output, dataset, pt, tag, dilep_selections, weight_muon, weight_electron):
        output["MM_{}_CR_weighted".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["mumu_CR"]], weight = weight_muon[dilep_selections["mumu_CR"]])
        output["MM_{}_SR".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["mumu_SR"]])
        output["EE_{}_CR_weighted".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["ee_CR"]], weight = weight_electron[dilep_selections["ee_CR"]])
        output["EE_{}_SR".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["ee_SR"]])
        output["EM_{}_CR_weighted".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["emu_CR"]], weight = weight_muon[dilep_selections["emu_CR"]])
        output["EM_{}_SR".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["emu_SR"]])
        output["ME_{}_CR_weighted".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["mue_CR"]], weight = weight_electron[dilep_selections["mue_CR"]])
        output["ME_{}_SR".format(tag)].fill(dataset=dataset, pt=pt[dilep_selections["mue_SR"]])
        
    def fill_ht_individual(self, output, dataset, ht, tag, dilep_selections, weight_muon, weight_electron):
        output["MM_{}_CR_weighted".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["mumu_CR"]], weight = weight_muon[dilep_selections["mumu_CR"]])
        output["MM_{}_SR".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["mumu_SR"]])
        output["EE_{}_CR_weighted".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["ee_CR"]], weight = weight_electron[dilep_selections["ee_CR"]])
        output["EE_{}_SR".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["ee_SR"]])
        output["EM_{}_CR_weighted".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["emu_CR"]], weight = weight_muon[dilep_selections["emu_CR"]])
        output["EM_{}_SR".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["emu_SR"]])
        output["ME_{}_CR_weighted".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["mue_CR"]], weight = weight_electron[dilep_selections["mue_CR"]])
        output["ME_{}_SR".format(tag)].fill(dataset=dataset, ht=ht[dilep_selections["mue_SR"]])

    def fill_multiplicity_individual(self, output, dataset, mult, tag, dilep_selections, weight_muon, weight_electron):
        output["MM_{}_CR_weighted".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["mumu_CR"]], weight = weight_muon[dilep_selections["mumu_CR"]])
        output["MM_{}_SR".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["mumu_SR"]])
        output["EE_{}_CR_weighted".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["ee_CR"]], weight = weight_electron[dilep_selections["ee_CR"]])
        output["EE_{}_SR".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["ee_SR"]])
        output["EM_{}_CR_weighted".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["emu_CR"]], weight = weight_muon[dilep_selections["emu_CR"]])
        output["EM_{}_SR".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["emu_SR"]])
        output["ME_{}_CR_weighted".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["mue_CR"]], weight = weight_electron[dilep_selections["mue_CR"]])
        output["ME_{}_SR".format(tag)].fill(dataset=dataset, multiplicity=mult[dilep_selections["mue_SR"]])
        

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        events = events[ak.num(events.Jet)>0] #corrects for rare case where there isn't a single jet in event
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=0
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ##Jets
        Jets = events.Jet
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi
    
         ### For FCNC, we want electron -> tightTTH
        ele_t = Collections(ev, "Electron", "tightFCNC", year=self.year).get()
        ele_t_p = ele_t[((ele_t.genPartFlav==1) | (ele_t.genPartFlav==15))]
        ele_t_np = ele_t[((ele_t.genPartFlav!=1) & (ele_t.genPartFlav!=15))]

        ele_l = Collections(ev, "Electron", "fakeableFCNC", year=self.year).get()
        ele_l_p = ele_l[((ele_l.genPartFlav==1) | (ele_l.genPartFlav==15))]
        ele_l_np = ele_l[((ele_l.genPartFlav!=1) & (ele_l.genPartFlav!=15))]
        
        mu_t    = Collections(ev, "Muon", "tightFCNC", year=self.year).get()
        mu_t_p  = mu_t[((mu_t.genPartFlav==1) | (mu_t.genPartFlav==15))]
        mu_t_np = mu_t[((mu_t.genPartFlav!=1) & (mu_t.genPartFlav!=15))]

        mu_l = Collections(ev, "Muon", "fakeableFCNC", year=self.year).get()
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
        
        dilep_selections = {"mumu_SR":mumu_SR_sel, "mumu_CR": mumu_CR_sel, "ee_SR":ee_SR_sel, "ee_CR":ee_CR_sel, "mue_SR":mue_SR_sel, "mue_CR": mue_CR_sel, "emu_SR":emu_SR_sel, "emu_CR":emu_CR_sel}
        
        #combine all selections for generic CR and SR
        CR_sel = mumu_CR_sel | ee_CR_sel | mue_CR_sel | emu_CR_sel
        SR_sel = mumu_SR_sel | ee_SR_sel | mue_SR_sel | emu_SR_sel

        electron_2018 = fake_rate("../data/fake_rate/FR_electron_2018.p")
        electron_2017 = fake_rate("../data/fake_rate/FR_electron_2017.p")
        electron_2016 = fake_rate("../data/fake_rate/FR_electron_2016.p")
        muon_2018 = fake_rate("../data/fake_rate/FR_muon_2018.p")
        muon_2017 = fake_rate("../data/fake_rate/FR_muon_2017.p")
        muon_2016 = fake_rate("../data/fake_rate/FR_muon_2016.p")
        
        if self.year==2018:
            weight_muon = muon_2018.FR_weight(mu_l_np)
            weight_electron = electron_2018.FR_weight(ele_l_np)
            
        elif self.year==2017:
            weight_muon = muon_2017.FR_weight(mu_l_np)
            weight_electron = electron_2017.FR_weight(ele_l_np)
            
        elif self.year==2016:
            weight_muon = muon_2016.FR_weight(mu_l_np)
            weight_electron = electron_2016.FR_weight(ele_l_np)
            
        #separate by different combinations of two-lepton events
        output['EE_CR'].fill(dataset = dataset, weight = np.sum(ee_CR_sel[ee_CR_sel]))
        output['EE_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_electron[ee_CR_sel]))) 
        output['EE_SR'].fill(dataset = dataset, weight = np.sum(ee_SR_sel[ee_SR_sel]))

        output['MM_CR'].fill(dataset = dataset, weight = np.sum(mumu_CR_sel[mumu_CR_sel])) 
        output['MM_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_muon[mumu_CR_sel]))) 
        output['MM_SR'].fill(dataset = dataset, weight = np.sum(mumu_SR_sel[mumu_SR_sel]))

        output['EM_CR'].fill(dataset = dataset, weight = np.sum(emu_CR_sel[emu_CR_sel])) 
        output['EM_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_muon[emu_CR_sel]))) 
        output['EM_SR'].fill(dataset = dataset, weight = np.sum(emu_SR_sel[emu_SR_sel]))
        
        output['ME_CR'].fill(dataset = dataset, weight = np.sum(mue_CR_sel[mue_CR_sel])) 
        output['ME_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_electron[mue_CR_sel]))) 
        output['ME_SR'].fill(dataset = dataset, weight = np.sum(mue_SR_sel[mue_SR_sel]))
        
        #fill combined histograms now (basic definitions are in default_accumulators.py)
        self.SS_fill_weighted(output["MET_CR"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, dataset=dataset, pt=ev.MET.pt)
        self.SS_fill_weighted(output["MET_CR_weighted"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, mu_weights = weight_muon, e_weights = weight_electron, dataset=dataset, pt=ev.MET.pt)
        self.SS_fill_weighted(output["MET_SR"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset=dataset, pt=ev.MET.pt)
        self.fill_pt_individual(output, dataset, ev.MET.pt, "MET", dilep_selections, weight_muon, weight_electron)
        #leading lepton pt
        LeadLep_pt = ak.max(ak.concatenate([ev.Muon.pt, ev.Electron.pt], axis=1), axis=1) 
        #sum of all regions
        self.SS_fill_weighted(output["pt_LeadLep_CR"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, dataset=dataset, pt=LeadLep_pt)
        self.SS_fill_weighted(output["pt_LeadLep_CR_weighted"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, mu_weights = weight_muon, e_weights = weight_electron, dataset=dataset, pt=LeadLep_pt)
        self.SS_fill_weighted(output["pt_LeadLep_SR"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset=dataset, pt=LeadLep_pt)
        self.fill_pt_individual(output, dataset, LeadLep_pt, "pt_LeadLep", dilep_selections, weight_muon, weight_electron)
        
        #njets
        njets = ak.num(jets, axis=1)
        self.SS_fill_weighted(output["njets_CR"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, dataset=dataset, multiplicity=njets)
        self.SS_fill_weighted(output["njets_CR_weighted"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, mu_weights = weight_muon, e_weights = weight_electron, dataset=dataset, multiplicity=njets)
        self.SS_fill_weighted(output["njets_SR"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset=dataset, multiplicity=njets)
        self.fill_multiplicity_individual(output, dataset, njets, "njets", dilep_selections, weight_muon, weight_electron)
        
        #btags
        btag = ak.num(getBTagsDeepFlavB(jets, year=self.year))
        self.SS_fill_weighted(output["N_b_CR"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, dataset=dataset, multiplicity=btag)
        self.SS_fill_weighted(output["N_b_CR_weighted"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, mu_weights = weight_muon, e_weights = weight_electron, dataset=dataset, multiplicity=btag)
        self.SS_fill_weighted(output["N_b_SR"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset=dataset, multiplicity=btag)
        self.fill_multiplicity_individual(output, dataset, btag, "N_b", dilep_selections, weight_muon, weight_electron)
        
        #HT
        ht = ak.sum(jets.pt, axis=1)
        self.SS_fill_weighted(output["HT_CR"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, dataset=dataset, ht=ht)
        self.SS_fill_weighted(output["HT_CR_weighted"], mumu_CR_sel, ee_CR_sel, mue_CR_sel, emu_CR_sel, mu_weights = weight_muon, e_weights = weight_electron, dataset=dataset, ht=ht)
        self.SS_fill_weighted(output["HT_SR"], mumu_SR_sel, ee_SR_sel, mue_SR_sel, emu_SR_sel, dataset=dataset, ht=ht)
        self.fill_ht_individual(output, dataset, ht, "HT", dilep_selections, weight_muon, weight_electron)
        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.default_accumulators import desired_output, add_processes_to_output, dataset_axis, pt_axis, eta_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset

    overwrite = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'nano_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    fileset = make_fileset(['QCD'], samples, redirector=redirector_ucsd, small=True)
    
    meta = get_sample_meta(fileset, samples)

    add_processes_to_output(fileset, desired_output)
    
    desired_output.update({
        "single_mu_fakeable": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis),
        "single_mu": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis)
    })

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            nano_analysis(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()
