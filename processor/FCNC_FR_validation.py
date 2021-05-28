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
        electron         = Collections(ev, "Electron", "tightFCNC").get()
        fakeableelectron = Collections(ev, "Electron", "fakeableFCNC").get()
        
        muon         = Collections(ev, "Muon", "tightFCNC").get()
        fakeablemuon = Collections(ev, "Muon", "fakeableFCNC").get()
        
        #validation cuts are split up based on gen-level information
        tight_muon_gen_prompt        = Collections(ev, "Muon", "tightFCNCGenPrompt", year=self.year).get()
        tight_muon_gen_nonprompt     = Collections(ev, "Muon", "tightFCNCGenNonprompt", year=self.year).get()
        tight_electron_gen_prompt    = Collections(ev, "Electron", "tightFCNCGenPrompt", year=self.year).get()
        tight_electron_gen_nonprompt = Collections(ev, "Electron", "tightFCNCGenNonprompt", year=self.year).get()

        loose_muon_gen_nonprompt     = Collections(ev, "Muon", "fakeableFCNCGenNonprompt", year=self.year).get()
        loose_electron_gen_nonprompt = Collections(ev, "Electron", "fakeableFCNCGenNonprompt", year=self.year).get()
        
        ##Jets
        Jets = events.Jet
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi
    
        lepton   = fakeablemuon   #ak.concatenate([fakeablemuon, fakeableelectron], axis=1)
        mt_lep_met = mt(lepton.pt, lepton.phi, ev.MET.pt, ev.MET.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)
        
        selection = PackedSelection()
        selection.add('MET<20',   (ev.MET.pt < 20))
        selection.add('mt<20',     min_mt_lep_met < 20)
        selection_reqs = ['MET<20', 'mt<20'] 
        fcnc_reqs_d = { sel: True for sel in selection_reqs}
        fcnc_selection = selection.require(**fcnc_reqs_d)
        
        # define the weight
        #weight = Weights( len(ev) )

        jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
        #get loose leptons that are explicitly not tight
        muon_orthogonality_param = ((ak.num(loose_muon_gen_nonprompt)==1) & (ak.num(tight_muon_gen_nonprompt)==0) | 
                                    (ak.num(loose_muon_gen_nonprompt)==2) & (ak.num(tight_muon_gen_nonprompt)==1) )

        electron_orthogonality_param = ((ak.num(loose_electron_gen_nonprompt)==1) & (ak.num(tight_electron_gen_nonprompt)==0) | 
                                        (ak.num(loose_electron_gen_nonprompt)==2) & (ak.num(tight_electron_gen_nonprompt)==1) )

        #clean jets :
        # we want at least two jets that are outside of the lepton jets by deltaR > 0.4
        jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
        jet_sel = (ak.num(jets[~(match(jets, fakeableelectron, deltaRCut=0.4)|match(jets, fakeablemuon, deltaRCut=0.4))])>=2)

        #jet_sel = (ak.num(jets[~( match(jets, tight_muon_gen_prompt       , deltaRCut=0.4) | 
        #                          match(jets, tight_muon_gen_nonprompt    , deltaRCut=0.4) | 
        #                          match(jets, tight_electron_gen_prompt   , deltaRCut=0.4) | 
        #                          match(jets, tight_electron_gen_nonprompt, deltaRCut=0.4) | 
        #                         (match(jets, loose_muon_gen_nonprompt       , deltaRCut=0.4) & muon_orthogonality_param) | 
        #                         (match(jets, loose_electron_gen_nonprompt   , deltaRCut=0.4) & electron_orthogonality_param))])>=2)

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

        test_muon = ak.concatenate([tight_muon_gen_nonprompt, tight_muon_gen_prompt], axis=1)
        test_muon_SS = (ak.sum(test_muon.charge, axis=1)!=0)
        #lead_muon_pt = ak.max(test_muon.pt,axis=1)
        #MM_SR_sel = (ak.num(tight_muon_gen_nonprompt)==1) & (ak.num(tight_muon_gen_prompt)==1) & (ak.num(fakeablemuon)==2) & (ak.num(fakeableelectron)==0) & jet_sel & test_muon_SS
        MM_SR_sel = (ak.num(tight_muon_gen_nonprompt)==1) & (ak.num(tight_muon_gen_prompt)==1) & (ak.num(fakeablemuon)==2) & jet_sel & test_muon_SS & (ak.num(test_muon[test_muon.pt>25])>0) & (ak.num(fakeableelectron)==0)
        #MM_SR_sel = SS_selection(tight_muon_gen_nonprompt, tight_muon_gen_prompt)  & two_lepton_sel & jet_sel
        #MM CR (Tight gen-level prompt mu + L!T gen-level nonprompt mu)
        SR_muon = ak.concatenate([tight_muon_gen_prompt, loose_muon_gen_nonprompt], axis=1)
        SR_muon_SS = (ak.sum(SR_muon.charge, axis=1)!=0)
        MM_CR_sel = (ak.num(tight_muon_gen_prompt)==1) & (ak.num(loose_muon_gen_nonprompt)==1) & (ak.num(fakeablemuon)==2) & jet_sel & SR_muon_SS & (ak.num(SR_muon[SR_muon.pt>25])>0) & (ak.num(fakeableelectron)==0)
        #MM_CR_sel = (SS_selection(tight_muon_gen_prompt, loose_muon_gen_nonprompt) & muon_orthogonality_param) & two_lepton_sel & jet_sel & (ak.num(test_muon[test_muon.pt>25])>0) & (ak.num(fakeableelectron)==0)
        
        #EM SR (Tight gen-level prompt e + Tight gen-level nonprompt mu)
        test_em = ak.concatenate([tight_electron_gen_prompt, tight_muon_gen_nonprompt], axis=1)
        test_em_SS = (ak.sum(test_em.charge, axis=1)!=0)
        EM_SR_sel = (ak.num(tight_electron_gen_prompt)==1) & (ak.num(tight_muon_gen_nonprompt)==1) & (ak.num(fakeablemuon)==1) & jet_sel & test_em_SS & (ak.num(test_em[test_em.pt>25])>0) & (ak.num(fakeableelectron)==1)
        #EM_SR_sel = SS_selection(tight_electron_gen_prompt, tight_muon_gen_nonprompt) & two_lepton_sel & jet_sel
        #EM_CR (Tight gen-level prompt e + L!T gen-level nonprompt mu)
        EM_CR_sel = (SS_selection(tight_electron_gen_prompt, loose_muon_gen_nonprompt) & muon_orthogonality_param) & two_lepton_sel & jet_sel
        
        #ME SR (Tight gen-level prompt mu + Tight gen-level nonprompt e)
        ME_SR_sel = SS_selection(tight_electron_gen_nonprompt, tight_muon_gen_prompt) & two_lepton_sel & jet_sel
        #ME CR (Tight gen-level prompt mu + L!T gen-level nonprompt e)
        ME_CR_sel = (SS_selection(tight_muon_gen_prompt, loose_electron_gen_nonprompt) & electron_orthogonality_param) & two_lepton_sel & jet_sel
        
        debug_sel = (SS_selection(tight_muon_gen_nonprompt, tight_muon_gen_prompt) | SS_selection(tight_electron_gen_prompt, tight_muon_gen_nonprompt)) & two_lepton_sel & jet_sel

        electron_2018 = fake_rate("../data/fake_rate/FR_electron_2018.p")
        electron_2017 = fake_rate("../data/fake_rate/FR_electron_2017.p")
        electron_2016 = fake_rate("../data/fake_rate/FR_electron_2016.p")
        muon_2018 = fake_rate("../data/fake_rate/FR_muon_2018.p")
        muon_2017 = fake_rate("../data/fake_rate/FR_muon_2017.p")
        muon_2016 = fake_rate("../data/fake_rate/FR_muon_2016.p")
        
        if self.year==2018:
            weight_muon = muon_2018.FR_weight(loose_muon_gen_nonprompt)
            weight_electron = electron_2018.FR_weight(loose_electron_gen_nonprompt)
            
        elif self.year==2017:
            weight_muon = muon_2017.FR_weight(loose_muon_gen_nonprompt)
            weight_electron = electron_2017.FR_weight(loose_electron_gen_nonprompt)
            
        elif self.year==2016:
            weight_muon = muon_2016.FR_weight(loose_muon_gen_nonprompt)
            weight_electron = electron_2016.FR_weight(loose_electron_gen_nonprompt)
        
        output['EE_CR'].fill(dataset = dataset, weight = np.sum(EE_CR_sel[EE_CR_sel])) #np.sum(np.ones_like(ak.to_numpy(ak.flatten(loose_electron_gen_nonprompt[EE_CR_sel]))))
        output['EE_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_electron[EE_CR_sel]))) #np.sum(np.ones_like(ak.to_numpy(ak.flatten(loose_electron_gen_nonprompt[EE_CR_sel]))) * ak.to_numpy(weight_electron[EE_CR_sel]))
        output['EE_SR'].fill(dataset = dataset, weight = np.sum(EE_SR_sel[EE_SR_sel]))

        output['MM_CR'].fill(dataset = dataset, weight = np.sum(MM_CR_sel[MM_CR_sel])) 
        output['MM_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_muon[MM_CR_sel]))) 
        output['MM_SR'].fill(dataset = dataset, weight = np.sum(MM_SR_sel[MM_SR_sel]))

        output['EM_CR'].fill(dataset = dataset, weight = np.sum(EM_CR_sel[EM_CR_sel])) 
        output['EM_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_muon[EM_CR_sel]))) 
        output['EM_SR'].fill(dataset = dataset, weight = np.sum(EM_SR_sel[EM_SR_sel]))
        
        output['ME_CR'].fill(dataset = dataset, weight = np.sum(ME_CR_sel[ME_CR_sel])) 
        output['ME_CR_weighted'].fill(dataset = dataset, weight = np.sum(ak.to_numpy(weight_electron[ME_CR_sel]))) 
        output['ME_SR'].fill(dataset = dataset, weight = np.sum(ME_SR_sel[ME_SR_sel]))

        
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
