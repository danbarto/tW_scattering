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

class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, debug=False):
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
        tight_muon_gen_prompt        = Collections(ev, "Muon", "tightFCNCGenPrompt").get()
        tight_muon_gen_nonprompt     = Collections(ev, "Muon", "tightFCNCGenNonprompt").get()
        tight_electron_gen_prompt    = Collections(ev, "Electron", "tightFCNCGenPrompt").get()
        tight_electron_gen_nonprompt = Collections(ev, "Electron", "tightFCNCGenNonprompt").get()

        loose_muon_gen_prompt     = Collections(ev, "Muon", "fakeableFCNCGenPrompt").get()
        loose_electron_gen_prompt = Collections(ev, "Electron", "fakeableFCNCGenPrompt").get()
        
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
        #selection.add('MET<19',        (ev.MET.pt<19) )
        selection_reqs = ['MET<20', 'mt<20']#, 'MET<19']
        fcnc_reqs_d = { sel: True for sel in selection_reqs}
        fcnc_selection = selection.require(**fcnc_reqs_d)
        
        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)

        jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
        #get loose leptons that are explicitly not tight
        muon_orthogonality_param = ((ak.num(loose_muon_gen_prompt)==1) & (ak.num(tight_muon_gen_prompt)==0) | 
                                    (ak.num(loose_muon_gen_prompt)==2) & (ak.num(tight_muon_gen_prompt)==1) )

        electron_orthogonality_param = ((ak.num(loose_electron_gen_prompt)==1) & (ak.num(tight_electron_gen_prompt)==0) | 
                                        (ak.num(loose_electron_gen_prompt)==2) & (ak.num(tight_electron_gen_prompt)==1) )

        loose_muon_gen_prompt_orthogonal = loose_muon_gen_prompt[muon_orthogonality_param]
        loose_electron_gen_prompt_orthogonal = loose_muon_gen_prompt[electron_orthogonality_param]


        #clean jets :
        # we want at least two jets that are outside of the lepton jets by deltaR > 0.4
        jets = getJets(ev, maxEta=2.4, minPt=25, pt_var='pt')
        jet_sel = (ak.num(jets[~( match(jets, tight_muon_gen_prompt       , deltaRCut=0.4) | 
                                  match(jets, tight_muon_gen_nonprompt    , deltaRCut=0.4) | 
                                  match(jets, tight_electron_gen_prompt   , deltaRCut=0.4) | 
                                  match(jets, tight_electron_gen_nonprompt, deltaRCut=0.4) | 
                                 (match(jets, loose_muon_gen_prompt       , deltaRCut=0.4) & muon_orthogonality_param) | 
                                 (match(jets, loose_electron_gen_prompt   , deltaRCut=0.4) & electron_orthogonality_param))])>=2)

        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

        two_lepton_sel = ( ak.num(tight_muon_gen_prompt)     + ak.num(tight_electron_gen_prompt)    + 
                           ak.num(tight_muon_gen_nonprompt)  + ak.num(tight_electron_gen_nonprompt) + 
                          (ak.num(loose_muon_gen_prompt)     - ak.num(tight_muon_gen_prompt))       +    #muon L!T counts
                          (ak.num(loose_electron_gen_prompt) - ak.num(tight_electron_gen_prompt)))  == 2 #electron L!T counts

        #TT selection is two tight leptons, where one is a gen-level prompt, and the other is a gen-level nonprompt, so we should
        #account for all of the possible lepton combinations below:
        TT_selection = (SS_selection(tight_electron_gen_prompt, tight_muon_gen_nonprompt)     |
                        SS_selection(tight_electron_gen_nonprompt, tight_muon_gen_prompt)     |
                        SS_selection(tight_electron_gen_prompt, tight_electron_gen_nonprompt) | 
                        SS_selection(tight_muon_gen_prompt, tight_muon_gen_nonprompt)         ) & two_lepton_sel & jet_sel
        #SS_selection gives us all events that have a same sign pair of leptons coming from the provided two object collections

        #TL selection is one tight lepton that is a gen-level prompt, and one loose (and NOT tight) lepton that is a gen-level nonprompt.
        #The orthogonality_param is a hacky way to ensure that we are only looking at 2 lepton events that have a tight not loose lepton in the event
        TL_selection = ((SS_selection(tight_electron_gen_prompt, loose_muon_gen_prompt)     & muon_orthogonality_param)     |
                        (SS_selection(tight_muon_gen_prompt, loose_muon_gen_prompt)         & muon_orthogonality_param)     |
                        (SS_selection(tight_electron_gen_prompt, loose_electron_gen_prompt) & electron_orthogonality_param) |
                        (SS_selection(tight_muon_gen_prompt, loose_electron_gen_prompt)     & electron_orthogonality_param) ) & two_lepton_sel & jet_sel

        muon_FR = fake_rate("../data/fake_rates.p")
        
        output['TT_tight_mu_prompt'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(tight_mu_gen_prompt[TT_selection].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(tight_mu_gen_prompt[TT_selection].eta)))
        )
        output['TT_tight_mu_nonprompt'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(tight_mu_gen_nonprompt[TT_selection].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(tight_mu_gen_nonprompt[TT_selection].eta)))
        )
        output['TL_tight_mu_prompt'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(tight_mu_gen_prompt[TT_selection].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(tight_mu_gen_prompt[TT_selection].eta)))
        )
        output['TL_tight_mu_nonprompt'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(tight_mu_gen_nonprompt[TT_selection].conePt)),
            eta = np.abs(ak.to_numpy(ak.flatten(tight_mu_gen_nonprompt[TT_selection].eta)))
        )

        
        if self.debug:
            #create pandas dataframe for debugging
            passed_events = ev[tight_muon_sel]
            passed_muons = muon[tight_muon_sel]
            event_p = ak.to_pandas(passed_events[["event"]])
            event_p["MET_PT"] = passed_events["MET"]["pt"]
            event_p["mt"] = min_mt_lep_met[tight_muon_sel]
            event_p["num_tight_mu"] = ak.to_numpy(ak.num(muon)[tight_muon_sel])
            event_p["num_loose_mu"] = ak.num(fakeablemuon)[tight_muon_sel]
            muon_p = ak.to_pandas(ak.flatten(passed_muons)[["pt", "conePt", "eta", "dz", "dxy", "ptErrRel", "miniPFRelIso_all", "jetRelIsoV2", "jetRelIso", "jetPtRelv2"]])
            #convert to numpy array for the output
            events_array = pd.concat([muon_p, event_p], axis=1)

            events_to_add = [6886009]
            for e in events_to_add:
                tmp_event = ev[ev.event==e]
                added_event = ak.to_pandas(tmp_event[["event"]])
                added_event["MET_PT"] = tmp_event["MET"]["pt"]
                added_event["mt"] = min_mt_lep_met[ev.event==e]
                added_event["num_tight_mu"] = ak.to_numpy(ak.num(muon)[ev.event==e])
                added_event["num_loose_mu"] = ak.to_numpy(ak.num(fakeablemuon)[ev.event==e])
                add_muon = ak.to_pandas(ak.flatten(muon[ev.event==e])[["pt", "conePt", "eta", "dz", "dxy", "ptErrRel", "miniPFRelIso_all", "jetRelIsoV2", "jetRelIso", "jetPtRelv2"]])
                add_concat = pd.concat([add_muon, added_event], axis=1)
                events_array = pd.concat([events_array, add_concat], axis=0)

            output['muons_df'] += processor.column_accumulator(events_array.to_numpy())
        
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
