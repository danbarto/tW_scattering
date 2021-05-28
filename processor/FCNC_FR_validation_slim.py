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

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        events = events[ak.num(events.Jet)>0] #corrects for rare case where there isn't a single jet in event
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=0
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        # cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ### For FCNC, we want electron -> tightTTH
        ele_t = Collections(ev, "Electron", "tightFCNC", year=self.year).get()
        ele_t_p = ele_t[((ele_t.genPartFlav==1) | (ele_t.genPartFlav==15))]
        ele_t_np = ele_t[((ele_t.genPartFlav!=1) & (ele_t.genPartFlav!=15))]

        ele_l = Collections(ev, "Electron", "fakeableFCNC", year=self.year).get()
        ele_l_p = ele_l[((ele_l.genPartFlav==1) | (ele_l.genPartFlav==15))]
        ele_l_np = ele_l[((ele_l.genPartFlav!=1) & (ele_l.genPartFlav!=15))]
        
        mu_t         = Collections(ev, "Muon", "tightFCNC", year=self.year).get()
        mu_t_p = mu_t[((mu_t.genPartFlav==1) | (mu_t.genPartFlav==15))]
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
        
        output['MM_CR'].fill(dataset = dataset, weight = np.sum(mumu_CR_sel[mumu_CR_sel])) 
        output['MM_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_muon[mumu_CR_sel]))) 
        output['MM_SR'].fill(dataset = dataset, weight = np.sum(mumu_SR_sel[mumu_SR_sel]))

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
