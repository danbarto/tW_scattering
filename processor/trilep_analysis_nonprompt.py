import awkward1 as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import *
from Tools.btag_scalefactors import *
#from Tools.lepton_scalefactors import *
from Tools.selections import Selection

class trilep_analysis(processor.ProcessorABC):
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
        
        # use a very loose preselection to filter the events
        presel = ak.num(events.Jet)>2
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        
        ## Generated leptons
        gen_lep = ev.GenL
        leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
        trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ## Muons
        muon     = Collections(ev, "Muon", "tightTTH").get()
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightTTH").get()
        vetoelectron = Collections(ev, "Electron", "vetoTTH").get()
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        ## Merge electrons and muons - this should work better now in ak1
        dilepton = cross(muon, electron)

        dimuon = choose(muon,2)
        OS_dimuon = dimuon[(dimuon['0'].charge*dimuon['1'].charge < 0)]

        dielectron = choose(electron)
        OS_dielectron = dielectron[(dielectron['0'].charge*dielectron['1'].charge < 0)]

        OS_dimuon_bestZmumu = OS_dimuon[ak.singletons(ak.argmin(abs(OS_dimuon.mass-91.2), axis=1))]
        OS_dielectron_bestZee = OS_dielectron[ak.singletons(ak.argmin(abs(OS_dielectron.mass-91.2), axis=1))]
        OS_dilepton_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.mass, OS_dielectron_bestZee.mass], axis=1), 1, clip=True), -1)

        lepton   = ak.concatenate([muon, electron], axis=1)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var='pt_nom')
        jet       = jet[ak.argsort(jet.pt_nom, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        central   = jet[(abs(jet.eta)<2.4)]
        btag      = getBTagsDeepFlavB(jet, year=self.year) # should study working point for DeepJet
        light     = getBTagsDeepFlavB(jet, year=self.year, invert=True)
        fwd       = getFwdJet(light)
        fwd_noPU  = getFwdJet(light, puId=False)
        
        ## forward jets
        j_fwd = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator
        
        jf          = cross(j_fwd, jet)
        mjf         = (jf['0']+jf['1']).mass
        j_fwd2      = jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'] # this is the jet that forms the largest invariant mass with j_fwd
        delta_eta   = abs(j_fwd2.eta - j_fwd.eta)

        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        ## other variables
        ht = ak.sum(jet.pt, axis=1)
        st = met_pt + ht + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)
        
        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            ## lepton SFs
            #weight.add("lepton", self.leptonSF.get(electron, muon))
        
        cutflow     = Cutflow(output, ev, weight=weight)

        sel = Selection(
            dataset = dataset,
            events = ev,
            year = self.year,
            ele = electron,
            ele_veto = vetoelectron,
            mu = muon,
            mu_veto = vetomuon,
            jet_all = jet,
            jet_central = central,
            jet_btag = btag,
            jet_fwd = fwd,
            met = ev.MET,
        )

        BL = sel.trilep_baseline(cutflow=cutflow)
        
        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight.weight()[BL])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight.weight()[BL])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight.weight()[BL])
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL], weight=weight.weight()[BL])
        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight.weight()[BL])
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])
        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL], weight=weight.weight()[BL])
        output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight.weight()[BL])
        output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight.weight()[BL])
        output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight.weight()[BL])
        output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight.weight()[BL])
        output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight.weight()[BL])
        output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight.weight()[BL])
        
        # make a plot of the dilepton mass, but without applying the cut on the dilepton mass itself (N-1 plot)
        output['dilep_mass'].fill(dataset=dataset, mass=ak.flatten(OS_dilepton_mass[sel.trilep_baseline(omit=['offZ'])]), weight=weight.weight()[sel.trilep_baseline(omit=['offZ'])])

        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[BL].pt,
            phi  = ev.MET[BL].phi,
            weight = weight.weight()[BL]
        )

        output['lead_gen_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_gen_lep[BL].pt)),
            eta = ak.to_numpy(ak.flatten(leading_gen_lep[BL].eta)),
            phi = ak.to_numpy(ak.flatten(leading_gen_lep[BL].phi)),
            weight = weight.weight()[BL]
        )

        output['trail_gen_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['lead_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['trail_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
            eta = ak.flatten(jet.eta[:, 0:1][BL]),
            phi = ak.flatten(jet.phi[:, 0:1][BL]),
            weight = weight.weight()[BL]
        )
        
        output['j2'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 1:2][BL].pt_nom),
            eta = ak.flatten(jet[:, 1:2][BL].eta),
            phi = ak.flatten(jet[:, 1:2][BL].phi),
            weight = weight.weight()[BL]
        )
        
        #output['j3'].fill(
        #    dataset = dataset,
        #    pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
        #    eta = ak.flatten(jet[:, 2:3][BL].eta),
        #    phi = ak.flatten(jet[:, 2:3][BL].phi),
        #    weight = weight.weight()[BL]
        #)
        
        
        output['fwd_jet'].fill(
            dataset = dataset,
            pt  = ak.flatten(j_fwd[BL].pt),
            eta = ak.flatten(j_fwd[BL].eta),
            phi = ak.flatten(j_fwd[BL].phi),
            weight = weight.weight()[BL]
        )
            
        output['high_p_fwd_p'].fill(dataset=dataset, p = ak.flatten(j_fwd[BL].p), weight = weight.weight()[BL])
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.default_accumulators import *

    overwrite = False
    year = 2018
    small = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'trilep_analysis'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    
    fileset = {
        'topW_v3': fileset_2018['topW_v3'],
        'TTW': fileset_2018['TTW'],
        'TTZ': fileset_2018['TTZ'],
        'TTH': fileset_2018['TTH'],
        'diboson': fileset_2018['diboson'],
        'ttbar': fileset_2018['top2l'], # like 20 events (10x signal)
        'DY': fileset_2018['DY'], # like 20 events (10x signal)
    }

    fileset = make_small(fileset, small)
    
    add_processes_to_output(fileset, desired_output)

    # add some histograms that we defined in the processor
    # everything else is taken the default_accumulators.py
    from processor.default_accumulators import mass_axis, dataset_axis
    desired_output.update({
        "dilep_mass": hist.Hist("Counts", dataset_axis, mass_axis),
    })

    histograms = sorted(list(desired_output.keys()))

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
            trilep_analysis(year=year, variations=variations, accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()


    lines = ['entry']
    lines += [
            'filter',
            'lepveto',
            'trilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'offZ',
            'MET>50',
            'N_jet>2',
            'N_central>1',
            'N_btag>0',
            'N_fwd>0',
        ]

    from Tools.helpers import getCutFlowTable
    df = getCutFlowTable(output, processes=list(fileset.keys()), lines=lines, significantFigures=4, signal='topW_v3')

