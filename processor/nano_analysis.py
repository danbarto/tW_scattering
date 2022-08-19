import datetime

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist, util
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


class nano_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        #self.btagSF = btag_scalefactor(year)
        
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
        muon = muon[muon.pt > 20]
        muon = muon[abs(muon.eta) < 2.4]
        
        ## Electrons
        electron     = ev.Electron  # Collections(ev, "Electron", "tight").get()
        electron = electron[electron.pt > 20]
        electron = electron[abs(electron.eta) < 2.5] 

        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7 )
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

        central   = jet[(abs(jet.eta)<2.4)]
        btag      = getBTagsDeepFlavB(jet)
        light     = getBTagsDeepFlavB(jet, invert=True)
        light_central = light[(abs(light.eta)<2.5)]
        fwd       = getFwdJet(light)

        n_ele = ak.num(electron, axis=1)
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
        
        dilepton_mass = (leading_lepton+trailing_lepton).mass

        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        lt = met_pt + ak.sum(lepton.pt, axis=1)

        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # generator weight
            weight.add("weight", ev.genWeight)
            
        #filters   = getFilters(ev, year=self.year, dataset=dataset)
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        
        selection = PackedSelection()
        selection.add('dilep',         dilep )
        #selection.add('filter',        (filters) )
        
        bl_reqs = ['dilep']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)

        output['lead_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[baseline].eta)),
            #phi = ak.to_numpy(ak.flatten(leading_lepton[baseline].phi)),
            weight = weight.weight()[baseline],
            n_ele = n_ele[baseline],
            systematic = 'central',
        )

        output['m_ll'].fill(
            dataset = dataset,
            mass = ak.flatten(dilepton_mass[baseline]),
            n_ele = n_ele[baseline],
            weight = weight.weight()[baseline],
            systematic = 'central',
        )

        output['N_jet'].fill(
            dataset=dataset,
            systematic = 'central',
            n_ele = n_ele[baseline],
            multiplicity=ak.num(jet)[baseline],
            weight=weight.weight()[baseline],
        )

        output['N_fwd'].fill(
            dataset=dataset,
            systematic = 'central',
            n_ele = n_ele[baseline],
            multiplicity=ak.num(fwd)[baseline],
            weight=weight.weight()[baseline],
        )

        output['MET'].fill(
            dataset = dataset,
            systematic = 'central',
            n_ele = n_ele[baseline],
            pt  = met_pt[baseline],
            phi  = met_phi[baseline],
            weight = weight.weight()[baseline]
        )

        output['LT'].fill(
            dataset = dataset,
            systematic = 'central',
            n_ele = n_ele[baseline],
            ht  = lt[baseline],
            weight = weight.weight()[baseline]
        )




        ## This is just for charge flip.
        ## FIXME: can this just be deleted?
        #output["gen_matched_electron"].fill(
        #    dataset = dataset,
        #    pt  = ak.flatten(gen_matched_electron.pt),
        #    eta = abs(ak.flatten(gen_matched_electron.eta)),
        #    weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron)>0), gen_matched_electron.pt),
        #    #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        #)

        #output["flipped_electron"].fill(
        #    dataset = dataset,
        #    pt  = ak.flatten(gen_matched_electron[is_flipped].pt),
        #    eta = abs(ak.flatten(gen_matched_electron[is_flipped].eta)),
        #    weight = build_weight_like(weight.weight(), (ak.num(gen_matched_electron[is_flipped])>0), gen_matched_electron[is_flipped].pt),
        #    #weight = ak.flatten(weight.weight() * ak.ones_like(gen_matched_electron.pt)),
        #)
        
        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from processor.default_accumulators import desired_output, add_processes_to_output, add_files_to_output, dataset_axis

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping

    from processor.meta_processor import get_sample_meta

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=None, help="Rerun or try using existing results??")
    argParser.add_argument('--keep', action='store_true', default=None, help="Keep/use existing results??")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on a DASK cluster?")
    argParser.add_argument('--iterative', action='store_true', default=None, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2022', help="Which year to run on?")
    argParser.add_argument('--sample', action='store', default='all', )
    args = argParser.parse_args()

    iterative   = args.iterative
    overwrite   = args.rerun
    small       = args.small
    verysmall   = args.verysmall

    if verysmall:
        small = True

    year        = int(args.year[0:4])
    ul          = "UL%s"%(args.year[2:]) if year<2022 else "Run3_%s"%(args.year[2:])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    # load the config and the cache
    cfg = loadConfig()
    
    samples = get_samples(f"samples_{ul}.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    #fileset = make_fileset(['top', 'DY', 'diboson'], samples, year=year, redirector=redirector_ucsd, small=small)

    if args.sample == 'MCall':
        sample_list = ['DY', 'top', 'diboson']
        #sample_list = ['DY', 'topW', 'top', 'TTW', 'TTZ', 'XG', 'rare', 'diboson']
    elif args.sample == 'data':
        if year == 2018 or year == 2022:
            sample_list = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        else:
            sample_list = ['DoubleMuon', 'MuonEG', 'DoubleEG', 'SingleMuon', 'SingleElectron']
    else:
        sample_list = [args.sample]

    for sample in sample_list:
        # NOTE we could also rescale processes here?
        #
        print (f"Working on samples: {sample}")
        reweight = {}
        renorm   = {}
        for dataset in mapping[ul][sample]:
            if samples[dataset]['reweight'] == 1:
                reweight[dataset] = 1
                renorm[dataset] = 1
            else:
                # Currently only supporting a single reweight.
                weight, index = samples[dataset]['reweight'].split(',')
                index = int(index)
                renorm[dataset] = samples[dataset]['sumWeight']/samples[dataset][weight][index]  # NOTE: needs to be divided out
                reweight[dataset] = (weight, index)



        fileset = make_fileset([sample], samples, year=ul, redirector=redirector_ucsd, small=small)
        meta = get_sample_meta(fileset, samples)

        # NOTE need to update the sumweight values in the samples directory because we run directly from the grid
        # and some files might not be available / readable
        for dataset in meta:
            samples[dataset]['sumWeight'] = meta[dataset]['sumWeight']

        add_processes_to_output(fileset, desired_output)

        pt_axis_coarse  = hist.Bin("pt",            r"$p_{T}$ (GeV)", [15,40,60,80,100,200,300])
        eta_axis_coarse = hist.Bin("eta",           r"$\eta$", [0,0.8,1.479,2.5])


        from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, one_axis, mass_axis
        from processor.default_accumulators import systematic_axis, eft_axis, charge_axis, n_ele_axis, eta_axis, delta_eta_axis, pt_axis, delta_phi_axis
        desired_output.update({
            "lead_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "trail_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "lead_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "sublead_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "fwd_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "m_ll": hist.Hist("Counts", dataset_axis, systematic_axis, mass_axis, n_ele_axis),
            "LT": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, ht_axis),
            "N_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_fwd": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),

           })

        ##desired_output.update({
        ##    "gen_matched_electron": hist.Hist("Counts", dataset_axis, pt_axis_coarse, eta_axis_coarse),
        ##    "flipped_electron": hist.Hist("Counts", dataset_axis, pt_axis_coarse, eta_axis_coarse),
        ##})


        if local:# and not profile:
            exe = processor.FuturesExecutor(workers=10)

        elif iterative:
            exe = processor.IterativeExecutor()

        else:
            from Tools.helpers import get_scheduler_address
            from dask.distributed import Client, progress

            scheduler_address = get_scheduler_address()
            c = Client(scheduler_address)

            exe = processor.DaskExecutor(client=c, status=True, retries=3)

        runner = processor.Runner(
            exe,
            #retries=3,
            schema=NanoAODSchema,
            chunksize=500000,
            maxchunks=None,
            skipbadfiles=True,
           )

        # define the cache name
        cache_name = f'nano_analysis_{sample}_{year}{era}'

        # find an old existing output
        output = get_latest_output(cache_name, cfg)

        if overwrite or output is None:
            print ("I'm running now")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_name += f'_{timestamp}.coffea'
            #if small: cache_name += '_small'
            cache = os.path.join(os.path.expandvars(cfg['caches']['base']), cache_name)

            output = runner(
                fileset,
                treename="Events",
                processor_instance=nano_analysis(
                    year=year,
                    accumulator=desired_output,
                ),
            )

            util.save(output, cache)

        if not local:
            # clean up the DASK workers. this partially frees up memory on the workers
            c.cancel(output)
            # NOTE: this really restarts the cluster, but is the only fully effective
            # way of deallocating all the accumulated memory...
            c.restart()


    output = get_merged_output("nano_analysis", samples=samples, year=str(year))
    plot_dir    = os.path.join(os.path.expandvars(cfg['meta']['plots']), str(year), 'OS/v0.7.0_v2/')

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    from plots.helpers import makePlot

    # load the functions to make a nice plot from the output histograms
    # and the scale_and_merge function that scales the individual histograms
    # to match the physical cross section

    from Tools.config_helpers import get_merged_output

    lumi = 0.5

#    data = ['SingleMuon', 'DoubleMuon', 'MuonEG', 'EGamma']
    data = []
    order = ['diboson', 'DY', 'top']

    datasets = data + order

    all_processes = [ x[0] for x in output['lead_lep'].values().keys() ]
    signals = []
    omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)    
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    ht_bins_red = hist.Bin('ht', r'$p_{T}\ (GeV)$', 7,100,800)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)

    my_labels = {
        'topW_lep': 'top-W scat.',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'rare': 'rare',
        'top': r'$t\bar{t}$',
        'XG': 'XG',  # this is bare XG
        'DY': 'Drell-Yan',  # this is bare XG
    }

    my_colors = {
        'topW_lep': '#FF595E',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'rare': '#EE82EE',
        'top': '#1982C4',
        'XG': '#5bc0de',
        'DY': '#6A4C93',
    }
    for log in True, False:

        plot_dir_temp = plot_dir + "/log/" if log else plot_dir + "/lin/"

        makePlot(output, 'lead_lep', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=False, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'lead_lep_pt')
                 )

        makePlot(output, 'm_ll', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=False, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='ee',  
                 save=os.path.expandvars(plot_dir_temp+'m_ll_ee')
                 )
        makePlot(output, 'm_ll', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=False, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='em',
                 save=os.path.expandvars(plot_dir_temp+'m_ll_emu')
                 )
        makePlot(output, 'm_ll', 'mass',
                 data=data,
                 bins=mass_bins, log=log, normalize=False, axis_label=r'$M_{\ell\ell}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 channel='mm',
                 save=os.path.expandvars(plot_dir_temp+'m_ll_mumu')
                 )
        makePlot(output, 'N_jet', 'multiplicity',
                 data=data,
                 bins=N_bins, log=log, normalize=False, axis_label=r'$N_{jet}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'N_jet'),
                 )
        makePlot(output, 'N_fwd', 'multiplicity',
                 data=data,
                 bins=N_bins_red, log=log, normalize=False, axis_label=r'$N_{fwd\ jets}$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['pt_jesTotalUp'], downHists=['pt_jesTotalDown'],
                 save=os.path.expandvars(plot_dir_temp+'N_fwd'),
                 )

        makePlot(output, 'MET', 'pt',
                 data=data,
                 bins=pt_bins, log=log, normalize=False, axis_label=r'$p_{T}^{miss}$ (GeV)',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 #upHists=['jes_up'], downHists=['jes_down'],
                 save=os.path.expandvars(plot_dir_temp+'MET_pt'),
                 )

        makePlot(output, 'MET', 'phi',
                 data=data,
                 bins=None, log=log, normalize=False, axis_label=r'$\phi(p_{T}^{miss})$',
                 new_colors=my_colors, new_labels=my_labels,
                 order=order,
                 omit=omit,
                 signals=signals,
                 lumi=lumi,
                 save=os.path.expandvars(plot_dir_temp+'MET_phi'),
                 )
        makePlot(output, 'LT', 'ht',
                 data=data,
                 bins=ht_bins_red, log=log, normalize=False, axis_label=r'$L_T\ (GeV)$',
                 new_colors=my_colors, new_labels=my_labels, lumi=lumi,
                 order=order,
                 signals=signals,
                 save=os.path.expandvars(plot_dir_temp+'LT'),
                )
        
