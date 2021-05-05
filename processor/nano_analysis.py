
try:
    import awkward1 as ak
except ImportError:
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
from Tools.helpers import build_weight_like
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
        muon     = ev.Muon
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tight").get()

        gen_matched_electron = electron[((electron.genPartIdx >= 0) & (abs(electron.pdgId)==11))]
        gen_matched_electron = gen_matched_electron[abs(gen_matched_electron.matched_gen.pdgId)==11]

        is_flipped = (gen_matched_electron.matched_gen.pdgId*(-1) == gen_matched_electron.pdgId)

        ## Merge electrons and muons - this should work better now in ak1
        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

        lepton   = ak.concatenate([muon, electron], axis=1)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        
        selection = PackedSelection()
        selection.add('dilep',         dilep )
        selection.add('filter',        (filters) )
        
        bl_reqs = ['dilep', 'filter']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)

        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[baseline], weight=weight.weight()[baseline])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(muon)[baseline], weight=weight.weight()[baseline])

        output['lead_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[baseline].eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[baseline].phi)),
            weight = weight.weight()[baseline]
        )

        output["gen_matched_electron"].fill(
            dataset = dataset,
            pt  = ak.flatten(gen_matched_electron.pt),
            eta = abs(ak.flatten(gen_matched_electron.eta)),
            weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron)>0), gen_matched_electron.pt),
            #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        )

        output["flipped_electron"].fill(
            dataset = dataset,
            pt  = ak.flatten(gen_matched_electron[is_flipped].pt),
            eta = abs(ak.flatten(gen_matched_electron[is_flipped].eta)),
            weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron[is_flipped])>0), gen_matched_electron[is_flipped].pt),
            #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        )
        
        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.default_accumulators import desired_output, add_processes_to_output, add_files_to_output, dataset_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping

    from processor.meta_processor import get_sample_meta

    overwrite = False
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'nano_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    #fileset = make_fileset(['TTW', 'TTZ'], samples, redirector=redirector_ucsd, small=True, n_max=5)  # small, max 5 files per sample
    #fileset = make_fileset(['DY'], samples, redirector=redirector_ucsd, small=True, n_max=10)
    fileset = make_fileset(['top'], samples, redirector=redirector_fnal, small=False)

    add_processes_to_output(fileset, desired_output)

    pt_axis_coarse  = hist.Bin("pt",            r"$p_{T}$ (GeV)", [15,40,60,80,100,200,300])
    eta_axis_coarse = hist.Bin("eta",           r"$\eta$", [0,0.8,1.479,2.5])
    
    desired_output.update({
        "gen_matched_electron": hist.Hist("Counts", dataset_axis, pt_axis_coarse, eta_axis_coarse),
        "flipped_electron": hist.Hist("Counts", dataset_axis, pt_axis_coarse, eta_axis_coarse),
    })

    meta = get_sample_meta(fileset, samples)

    exe_args = {
        'workers': 12,
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
            chunksize=500000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    # load the functions to make a nice plot from the output histograms
    # and the scale_and_merge function that scales the individual histograms
    # to match the physical cross section
    
    from plots.helpers import makePlot, scale_and_merge
    
    # define a few axes that we can use to rebin our output histograms
    N_bins_red     = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    
    # define nicer labels and colors
    
    my_labels = {
        nano_mapping['TTW'][0]: 'ttW',
        nano_mapping['TTZ'][0]: 'ttZ',
        nano_mapping['DY'][0]: 'DY',
        nano_mapping['top'][0]: 't/tt+jets',
    }
    
    my_colors = {
        nano_mapping['TTW'][0]: '#8AC926',
        nano_mapping['TTZ'][0]: '#FFCA3A',
        nano_mapping['DY'][0]: '#6A4C93',
        nano_mapping['top'][0]: '#1982C4',
    }

    # take the N_ele histogram out of the output, apply the x-secs from samples to the samples in fileset
    # then merge the histograms into the categories defined in nano_mapping

    print ("Total events in output histogram N_ele: %.2f"%output['N_ele'].sum('dataset').sum('multiplicity').values(overflow='all')[()])
    
    my_hists = {}
    #my_hists['N_ele'] = scale_and_merge(output['N_ele'], samples, fileset, nano_mapping)
    my_hists['N_ele'] = scale_and_merge(output['N_ele'], meta, fileset, nano_mapping)
    print ("Total scaled events in merged histogram N_ele: %.2f"%my_hists['N_ele'].sum('dataset').sum('multiplicity').values(overflow='all')[()])
    
    # Now make a nice plot of the electron multiplicity.
    # You can have a look at all the "magic" (and hard coded monstrosities) that happens in makePlot
    # in plots/helpers.py
    
    makePlot(my_hists, 'N_ele', 'multiplicity',
             data=[],
             bins=N_bins_red, log=True, normalize=False, axis_label=r'$N_{electron}$',
             new_colors=my_colors, new_labels=my_labels,
             #order=[nano_mapping['DY'][0], nano_mapping['TTZ'][0]],
             save=os.path.expandvars(cfg['meta']['plots'])+'/nano_analysis/N_ele_test.png'
            )
