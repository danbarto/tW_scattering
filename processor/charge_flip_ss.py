try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection


# this is all very bad practice
from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.helpers import build_weight_like
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *

from Tools.chargeFlip import charge_flip

class charge_flip_ss(processor.ProcessorABC):
    def __init__(self, year=2018, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        #self.leptonSF = LeptonSF(year=year)
        
        self.charge_flip_ratio = charge_flip(os.path.expandvars('$TWHOME/data/chargeflip/chargeflipfull2016.pkl.gz'))
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=2
        
        ev = events[presel]
        dataset = ev.metadata['dataset']

        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tight").get()
        electron = electron[(electron.pt > 20) & (abs(electron.eta) < 2.4)]

        electron = electron[( (electron.genPartIdx >= 0) & (abs(electron.matched_gen.pdgId)==11) )] #from here on all leptons are gen-matched
        
        
        ##Muons
        muon     = Collections(ev, "Muon", "tight").get()
        muon = muon[(muon.pt > 20) & (abs(muon.eta) < 2.4)]
        
        muon = muon[( (muon.genPartIdx >= 0) & (abs(muon.matched_gen.pdgId)==13) )]
        
        
        ##Leptons

        lepton   = ak.concatenate([muon, electron], axis=1)
        SSlepton = (ak.sum(lepton.charge, axis=1) != 0) & (ak.num(lepton)==2)
        OSlepton = (ak.sum(lepton.charge, axis=1) == 0) & (ak.num(lepton)==2)
        
        
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        
        
        
        #jets
        jet       = getJets(ev, minPt=40, maxEta=2.4, pt_var='pt')
        jet       = jet[ak.argsort(jet.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] 
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        # setting up the various weights
        weight = Weights( len(ev) )
        weight2 = Weights( len(ev))
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            weight2.add("weight", ev.genWeight)
            
        weight2.add("charge flip", self.charge_flip_ratio.flip_weight(electron))
                                   
                      
        #selections    
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        ss = (SSlepton)
        os = (OSlepton)
        jet_all = (ak.num(jet) >= 2)
        
        
        selection = PackedSelection()
        selection.add('filter',      (filters) )
        selection.add('ss',          ss )
        selection.add('os',          os )
        selection.add('jet',         jet_all )
        
        bl_reqs = ['filter', 'jet']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)
        
        s_reqs = bl_reqs + ['ss']
        s_reqs_d = { sel: True for sel in s_reqs }
        ss_sel = selection.require(**s_reqs_d)
        
        o_reqs = bl_reqs + ['os']
        o_reqs_d = {sel: True for sel in o_reqs }
        os_sel = selection.require(**o_reqs_d)
   
        
        #outputs
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[baseline], weight=weight.weight()[baseline])
        
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(lepton)[ss_sel], weight=weight.weight()[ss_sel])
        
        output['N_ele2'].fill(dataset=dataset, multiplicity=ak.num(lepton)[os_sel], weight=weight2.weight()[os_sel])
        
        output["electron"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[ss_sel].pt)),
            eta = abs(ak.to_numpy(ak.flatten(leading_lepton[ss_sel].eta))),
            phi = ak.to_numpy(ak.flatten(leading_lepton[ss_sel].phi)),
            weight = weight.weight()[ss_sel]
        )
        
        output["electron2"].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[os_sel].pt)),
            eta = abs(ak.to_numpy(ak.flatten(leading_lepton[os_sel].eta))),
            phi = ak.to_numpy(ak.flatten(leading_lepton[os_sel].phi)),
            weight = weight2.weight()[os_sel]
        )

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from klepto.archives import dir_archive
    from processor.default_accumulators import desired_output, add_processes_to_output, add_files_to_output, dataset_axis, multiplicity_axis, pt_axis, eta_axis, phi_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping

    from processor.meta_processor import get_sample_meta
    overwrite = True
    local = True
   
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'charge_flip_check'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    samples = get_samples()

    #fileset = make_fileset(['TTW', 'TTZ'], samples, redirector=redirector_ucsd, small=True, n_max=5)  # small, max 5 files per sample
    #fileset = make_fileset(['DY'], samples, redirector=redirector_ucsd, small=True, n_max=10)
    #fileset = make_fileset(['top'], samples, redirector=redirector_ucsd, small=True)

    # Use just one local file for debugging.
    # Copied with: `xrdcp root://xcache-redirector.t2.ucsd.edu:2040//store/mc/RunIIAutumn18NanoAODv7/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/60000/022107FA-F567-1B44-B139-A18ADC996FCF.root .`
    fileset = {'/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': ['022107FA-F567-1B44-B139-A18ADC996FCF.root']}

    pt_axis_coarse  = hist.Bin("pt",            r"$p_{T}$ (GeV)", [15,40,60,80,100,200,300])
    eta_axis_coarse = hist.Bin("eta",           r"$\eta$", [0,0.8,1.479,2.5])

    desired_output.update({
        "N_ele2": hist.Hist("Counts", dataset_axis, multiplicity_axis),
        "electron2": hist.Hist("Counts", dataset_axis, pt_axis_coarse, eta_axis_coarse, phi_axis),
    })

    add_processes_to_output(fileset, desired_output)

    meta = get_sample_meta(fileset, samples)
   
    if local:

        exe_args = {
            'workers': 16,
            'function_args': {'flatten': False},
             "schema": NanoAODSchema,
        }
        exe = processor.futures_executor
   
    else:
        from Tools.helpers import get_scheduler_address
        from dask.distributed import Client, progress

        scheduler_address = get_scheduler_address()
        c = Client(scheduler_address)

        exe_args = {
            'client': c,
            'function_args': {'flatten': False},
            "schema": NanoAODSchema,
        }
        exe = processor.dask_executor
   
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            charge_flip_ss(year=year, variations=[], accumulator=desired_output),
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
