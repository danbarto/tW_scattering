import awkward1 as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import *
from Tools.config_helpers import *
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.lepton_scalefactors import *

class SS_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)
        
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
        muon     = Collections(ev, "Muon", "tightSSTTH").get()
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightSSTTH").get()
        vetoelectron = Collections(ev, "Electron", "vetoTTH").get()
        dielectron   = choose(electron, 2)
        SSelectron   = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        ## Merge electrons and muons - this should work better now in ak1
        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

        lepton   = ak.concatenate([muon, electron], axis=1)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        n_nonprompt = getNonPromptFromFlavour(electron) + getNonPromptFromFlavour(muon)
        n_chargeflip = getChargeFlips(electron, ev.GenPart) + getChargeFlips(muon, ev.GenPart)

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
        
        ## event selectors
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        pos_charge = ((ak.sum(electron.pdgId, axis=1) + ak.sum(muon.pdgId, axis=1))<0)
        neg_charge = ((ak.sum(electron.pdgId, axis=1) + ak.sum(muon.pdgId, axis=1))>0)
        lep0pt    = ((ak.num(electron[(electron.pt>25)]) + ak.num(muon[(muon.pt>25)]))>0)
        lep0pt_40 = ((ak.num(electron[(electron.pt>40)]) + ak.num(muon[(muon.pt>40)]))>0)
        lep0pt_100 = ((ak.num(electron[(electron.pt>100)]) + ak.num(muon[(muon.pt>100)]))>0)
        lep1pt    = ((ak.num(electron[(electron.pt>20)]) + ak.num(muon[(muon.pt>20)]))>1)
        lep1pt_30 = ((ak.num(electron[(electron.pt>30)]) + ak.num(muon[(muon.pt>30)]))>1)
        lepveto   = ((ak.num(vetoelectron) + ak.num(vetomuon))==2)
        
        # define the weight
        weight = Weights( len(ev) )
        
        #mult = 1
        #if dataset=='inclusive': mult = 0.0478/47.448
        #if dataset=='plus': mult = 0.0036/7.205

        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        
        selection = PackedSelection()
        selection.add('lepveto',       lepveto)
        selection.add('dilep',         dilep )
        selection.add('filter',        (filters) )
        selection.add('p_T(lep0)>25',  lep0pt )
        selection.add('p_T(lep0)>40',  lep0pt_40 )
        selection.add('p_T(lep1)>20',  lep1pt )
        selection.add('p_T(lep1)>30',  lep1pt_30 )
        selection.add('SS',            ( SSlepton | SSelectron | SSmuon) )
        selection.add('pos',           ( pos_charge ) )
        selection.add('neg',           ( neg_charge ) )
        selection.add('N_jet>3',       (ak.num(jet)>=4) )
        selection.add('N_jet>4',       (ak.num(jet)>=5) )
        selection.add('N_central>2',   (ak.num(central)>=3) )
        selection.add('N_central>3',   (ak.num(central)>=4) )
        selection.add('N_btag>0',      (ak.num(btag)>=1) )
        selection.add('MET>50',        (ev.MET.pt>50) )
        selection.add('ST',            (st>600) )
        selection.add('N_fwd>0',       (ak.num(fwd)>=1 ))
        selection.add('delta_eta',     (ak.any(delta_eta>2, axis=1) ) )
        selection.add('fwd_p>500',     (ak.any(j_fwd.p>500, axis=1) ) )
        
        ss_reqs = ['lepveto', 'dilep', 'SS', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20', 'N_jet>3', 'N_central>2', 'N_btag>0']
        bl_reqs = ss_reqs + ['N_fwd>0', 'N_jet>4', 'N_central>3', 'ST', 'MET>50', 'delta_eta']
        sr_reqs = bl_reqs + ['fwd_p>500', 'p_T(lep0)>40', 'p_T(lep1)>30']

        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_selection = selection.require(**ss_reqs_d)
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)
        sr_reqs_d = { sel: True for sel in sr_reqs }
        SR = selection.require(**sr_reqs_d)

        cutflow     = Cutflow(output, ev, weight=weight)
        cutflow_reqs_d = {}
        for req in sr_reqs:
            cutflow_reqs_d.update({req: True})
            cutflow.addRow( req, selection.require(**cutflow_reqs_d) )
        
        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[ss_selection].npvs, weight=weight.weight()[ss_selection])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[ss_selection].npvsGood, weight=weight.weight()[ss_selection])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[ss_selection], weight=weight.weight()[ss_selection])
        output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight.weight()[BL])
        output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight.weight()[BL])
        output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight.weight()[BL])
        output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight.weight()[BL])
        output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight.weight()[BL])
        output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight.weight()[BL])
        output['chargeFlip_vs_nonprompt'].fill(dataset=dataset, n1=n_chargeflip[ss_selection], n2=n_nonprompt[ss_selection], n_ele=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        
        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[ss_selection].pt,
            phi  = ev.MET[ss_selection].phi,
            weight = weight.weight()[ss_selection]
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
        
        output['j3'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
            eta = ak.flatten(jet[:, 2:3][BL].eta),
            phi = ak.flatten(jet[:, 2:3][BL].phi),
            weight = weight.weight()[BL]
        )
        
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.std_acumulators import *

    overwrite = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'SS_analysis'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    histograms = sorted(list(desired_output.keys()))
    
    year = 2018
    
    fileset = {
        'topW_v2': fileset_2018['topW_v2'],
        'TTW': fileset_2018['TTW'],
        'TTZ': fileset_2018['TTZ'],
        'TTH': fileset_2018['TTH'],
        'ttbar': fileset_2018['ttbar'],
    }
    
    
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
            SS_analysis(year=year, variations=variations, accumulator=desired_output),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()


