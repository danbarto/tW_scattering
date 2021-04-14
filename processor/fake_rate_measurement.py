import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *

class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
        
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
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ## Muons
        #muon     = ev.Muon
        
        ## Electrons
        electron         = Collections(ev, "Electron", "tightSSTTH").get()
        fakeableelectron = Collections(ev, "Electron", "fakeableSSTTH").get()
        #vetoelectron     = Collections(ev, "Electron", "vetoTTH").get() # "loose" electrons
        
        muon         = Collections(ev, "Muon", "tightSSTTH").get()
        fakeablemuon = Collections(ev, "Muon", "fakeableSSTTH").get()
        #vetomuon     = Collections(ev, "Muon", "vetoTTH").get()    # "loose" muons
        
        ##Jets
        Jets = events.Jet
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)

#         output['lead_lep'].fill(
#             dataset = dataset,
#             pt  = ak.to_numpy(ak.flatten(leading_lepton[baseline].pt)),
#             eta = ak.to_numpy(ak.flatten(leading_lepton[baseline].eta)),
#             phi = ak.to_numpy(ak.flatten(leading_lepton[baseline].phi)),
#             weight = weight.weight()[baseline]
#         )
        
        muon_selection = ((ak.num(fakeablemuon)==1) ^ (ak.num(muon)==1))
        output['single_mu_fakeable'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fakeablemuon[(ak.num(fakeablemuon)==1) & (ak.num(muon)==0) & (ak.num(Jets[~match(Jets, fakeablemuon, deltaRCut=0.7)])>=1)].conePt)),
            eta = ak.to_numpy(ak.flatten(fakeablemuon[(ak.num(fakeablemuon)==1) & (ak.num(muon)==0) & (ak.num(Jets[~match(Jets, fakeablemuon, deltaRCut=0.7)])>=1)].eta))
        )
        output['single_mu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(muon[(ak.num(fakeablemuon)==0) & (ak.num(muon)==1) & (ak.num(Jets[~match(Jets, muon, deltaRCut=0.7)])>=1)].conePt)),
            eta = ak.to_numpy(ak.flatten(muon[(ak.num(fakeablemuon)==0) & (ak.num(muon)==1) & (ak.num(Jets[~match(Jets, muon, deltaRCut=0.7)])>=1)].eta))
        )
        output['single_e_fakeable'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fakeableelectron[(ak.num(fakeableelectron)==1) & (ak.num(electron)==0) & (ak.num(Jets[~match(Jets, fakeableelectron, deltaRCut=0.7)])>=1)].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(fakeableelectron[(ak.num(fakeableelectron)==1) & (ak.num(electron)==0) & (ak.num(Jets[~match(Jets, fakeableelectron, deltaRCut=0.7)])>=1)].etaSC)))
        )
        output['single_e'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(electron[(ak.num(fakeableelectron)==0) & (ak.num(electron)==1) & (ak.num(Jets[~match(Jets, electron, deltaRCut=0.7)])>=1)].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(electron[(ak.num(fakeableelectron)==0) & (ak.num(electron)==1) & (ak.num(Jets[~match(Jets, electron, deltaRCut=0.7)])>=1)].etaSC)))
        )
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

    add_processes_to_output(fileset, desired_output)
    
    desired_output.update({
        "single_mu_fakeable": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis),
        "single_mu": hist.Hist("Counts", dataset_axis, pt_axis, eta_axis)
    })

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
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
