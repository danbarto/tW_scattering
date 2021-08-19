try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiData, LumiMask, LumiList

import numpy as np

# this is all very bad practice
from Tools.objects import Collections, choose, cross, match
from Tools.basic_objects import getJets
from Tools.config_helpers import loadConfig
from Tools.helpers import build_weight_like
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM
from Tools.triggers import getFilters, getTriggers

from memory_profiler import profile

class dielectron_mass(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>0
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        mu_t     = Collections(ev, "Muon", "tightSSTTH", year=year).get()

        triggers  = getTriggers(ev, year=self.year, dataset=dataset)

        selection = PackedSelection()
        selection.add('MET_pt',      (ev.MET.pt>0) )
        selection.add('singlemuon',  ((ak.num(mu_t[mu_t.pt>35])>0)) )
        selection.add('trig_1m',     (ev.HLT.IsoMu24) )
        selection.add('trigger',     (triggers) )
        
        bl_reqs = ['MET_pt', 'singlemuon', 'trigger']
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)

        run_ = ak.to_numpy(ev.run)
        lumi_ = ak.to_numpy(ev.luminosityBlock)
        event_ = ak.to_numpy(ev.event)

        output['%s_run'%dataset] += processor.column_accumulator(run_[BL])
        output['%s_lumi'%dataset] += processor.column_accumulator(lumi_[BL])
        output['%s_event'%dataset] += processor.column_accumulator(event_[BL])

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping
    from processor.default_accumulators import dataset_axis, multiplicity_axis, desired_output

    from processor.meta_processor import get_sample_meta

    #from pympler import muppy, summary

    year = 2018

    fileset_orig = make_fileset(['Data'], get_samples('samples_UL18.yaml'), year='UL2018', redirector=redirector_ucsd, small=False)

    fileset = {}
    # Checks can't be done with 2018B because lumi sections are missing in SingleMuon PD...
    # 2018C looks good.
    fileset['MuonEG'] = fileset_orig['/MuonEG/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD']
    fileset['SingleMuon'] = fileset_orig['/SingleMuon/Run2018C-UL2018_MiniAODv1_NanoAODv2-v2/NANOAOD']
    fileset['DoubleMuon'] = fileset_orig['/DoubleMuon/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD']
    #fileset['EGamma'] = fileset_orig['/EGamma/Run2018B-UL2018_MiniAODv1_NanoAODv2-v2/NANOAOD']

    exe_args = {
        'workers': 8,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
        "skipbadfiles": True,
    }
    exe = processor.futures_executor
    #exe = processor.iterative_executor

    from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis
    for rle in ['run', 'lumi', 'event']:
        desired_output.update({
                'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'SingleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
        })

    output = processor.run_uproot_job(
            fileset,
            "Events",
            dielectron_mass(year=year, variations=[], accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
    )    


    em = zip_run_lumi_event(output, 'MuonEG')
    m  = zip_run_lumi_event(output, 'SingleMuon')
    mm = zip_run_lumi_event(output, 'DoubleMuon')

    print ("Total events from MuonEG:", len(em))
    print ("Total events from SingleMuon:", len(m))
    print ("Total events from DoubleMuon:", len(mm))

    em_mm = np.intersect1d(em, mm)
    print ("Overlap MuonEG/DoubleMuon:", len(em_mm))

    m_mm = np.intersect1d(m, mm)
    print ("Overlap SingleMuon/DoubleMuon:", len(m_mm))

    em_m = np.intersect1d(em, m)
    print ("Overlap MuonEG/SingleMuon:", len(em_m))

    # All single muon triggered events should be in there.
    # If there's an overlap of the difference with MuonEG or DoubleMuon we have a problem.
    # We should probably still require two tight muons with pt>25/20 just to make sure.
