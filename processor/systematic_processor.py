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

class systematicsAnalyzer(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], acumulator={}):
        self.variations = variations
        
        # we can use a large number of bins and rebin later
        self.year = year
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)

        self._accumulator = processor.dict_accumulator( acumulator )
        

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
        
        ## Muons
        muon     = Collections(ev, "Muon", "tight").get()
        vetomuon = Collections(ev, "Muon", "veto").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        OSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)<0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tight").get()
        vetoelectron = Collections(ev, "Electron", "veto").get()
        dielectron   = choose(electron, 2)
        SSelectron   = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
        OSelectron   = ak.any((dielectron['0'].charge * dielectron['1'].charge)<0, axis=1)
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        ## Merge electrons and muons - this should work better now in ak1
        lepton   = ak.concatenate([muon, electron], axis=1)
        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)
        OSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)<0, axis=1)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7)
        jet       = jet[(jet.pt>25) & (jet.jetId>1)]
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        ## event selectors
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        triggers  = getTriggers(ev, year=self.year, dataset=dataset)
        
        dilep     = ((ak.num(electron)==1) & (ak.num(muon)==1))
        lep0pt    = ((ak.num(electron[(electron.pt>25)]) + ak.num(muon[(muon.pt>25)]))>0)
        lep1pt    = ((ak.num(electron[(electron.pt>20)]) + ak.num(muon[(muon.pt>20)]))>1)
        lepveto   = ((ak.num(vetoelectron) + ak.num(vetomuon))==2)
        
        # define the weight
        weight = Weights( len(ev) )
        
        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            
            ## PU weight - not in the babies...
            #weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            #weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        
        selection = PackedSelection()
        selection.add('lepveto',       lepveto)
        selection.add('dilep',         dilep )
        selection.add('trigger',       (triggers) )
        selection.add('filter',        (filters) )
        selection.add('p_T(lep0)>25',  lep0pt )
        selection.add('p_T(lep1)>20',  lep1pt )
        selection.add('OS',            OSlepton )
        selection.add('N_jet>2',       (ak.num(jet)>=3) )
        selection.add('MET>30',        (ev.MET.pt>30) )


        os_reqs = ['lepveto', 'dilep', 'trigger', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20', 'OS']
        bl_reqs = os_reqs + ['N_jet>2', 'MET>30']

        os_reqs_d = { sel: True for sel in os_reqs }
        os_selection = selection.require(**os_reqs_d)
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)

        cutflow     = Cutflow(output, ev, weight=weight)
        cutflow_reqs_d = {}
        for req in bl_reqs:
            cutflow_reqs_d.update({req: True})
            cutflow.addRow( req, selection.require(**cutflow_reqs_d) )

        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[os_selection].npvs, weight=weight.weight()[os_selection])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[os_selection].npvsGood, weight=weight.weight()[os_selection])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[os_selection], weight=weight.weight()[os_selection])
        
        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[os_selection].pt,
            phi  = ev.MET[os_selection].phi,
            weight = weight.weight()[os_selection]
        )
        
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet.pt[:, 0:1][BL]),
            eta = ak.flatten(jet.eta[:, 0:1][BL]),
            phi = ak.flatten(jet.phi[:, 0:1][BL]),
            weight = weight.weight()[BL]
        )

        # Now, take care of systematic unceratinties
        if not dataset=='MuonEG':
            alljets = getJets(ev, minPt=0, maxEta=4.7)
            alljets = alljets[(alljets.jetId>1)]
            for var in self.variations:
                # get the collections that change with the variations
                jet_var = getPtEtaPhi(alljets, pt_var=var)
                jet_var = jet_var[(jet_var.pt>25)]
                jet_var = jet_var[~match(jet_var, muon, deltaRCut=0.4)] # remove jets that overlap with muons
                jet_var = jet_var[~match(jet_var, electron, deltaRCut=0.4)] # remove jets that overlap with electrons


                # get the modified selection -> more difficult
                selection.add('N_jet>2_'+var, (ak.num(jet_var.pt)>3)) # something needs to be improved with getPtEtaPhi function
                selection.add('MET>30_'+var, (getattr(ev.MET, var)>30) )

                bl_reqs = os_reqs + ['N_jet>2_'+var, 'MET>30_'+var]
                bl_reqs_d = { sel: True for sel in bl_reqs }
                BL = selection.require(**bl_reqs_d)

                # the OS selection remains unchanged
                output['N_jet_'+var].fill(dataset=dataset, multiplicity=ak.num(jet_var)[os_selection], weight=weight.weight()[os_selection])

                # We don't need to redo all plots with variations. E.g., just add uncertainties to the jet plots.
                output['j1_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(jet_var.pt[:, 0:1][BL]),
                    eta = ak.flatten(jet_var.eta[:, 0:1][BL]),
                    phi = ak.flatten(jet_var.phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
                
        
        return output

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':

    year = 2018
    
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.std_acumulators import *


    fileset = {
        'tW_scattering': fileset_2018['tW_scattering'],
        'topW_v2': fileset_2018['topW_v2'],
    }
    
    
    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
    }
    exe = processor.futures_executor
    
    output = processor.run_uproot_job(
        fileset,
        "Events",
        systematicsAnalyzer(year=year, variations=variations, acumulator=desired_output),
        exe,
        exe_args,
        chunksize=250000,
    )


    from Tools.helpers import getCutFlowTable

    processes = ['tW_scattering', 'topW_v2']
    lines = ['entry']
    lines += ['lepveto', 'dilep', 'trigger', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20',  'OS', 'N_btag=2', 'N_jet>2', 'N_fwd>0', 'MET>30']
    df = getCutFlowTable(output, processes=processes, lines=lines, significantFigures=4)
