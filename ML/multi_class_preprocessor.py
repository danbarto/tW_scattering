import awkward as ak

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

import sys
import warnings


out_dict = {
    'total':          processor.defaultdict_accumulator(int),
    'presel':         processor.defaultdict_accumulator(int),
    'sel':            processor.defaultdict_accumulator(int),
    'topW_v3':        processor.defaultdict_accumulator(int),
    'topW_v2':        processor.defaultdict_accumulator(int),
    'TTW':            processor.defaultdict_accumulator(int),
    'TTZ':            processor.defaultdict_accumulator(int),
    'TTH':            processor.defaultdict_accumulator(int),
    'ttbar':          processor.defaultdict_accumulator(int),
}

variables = [
    'mjj_max',
    'delta_eta_jj',
    'met',
    'ht',
    'st',
    'n_jet',
    'n_fwd',
    'n_central',
    'n_tau',
    'n_track',
    'fwd_jet_p',
    'fwd_jet_pt',
    'fwd_jet_eta',
    'lead_jet_pt',
    'sublead_jet_pt',
    'lead_lep_pt',
    'lead_lep_eta',
    'sublead_lep_pt',
    'sublead_lep_eta',
    'dilepton_mass',
    'dilepton_pt',
    'label',
]

for var in variables:
    out_dict.update({var: processor.column_accumulator(np.zeros(shape=(0,)))})


class ML_preprocessor(processor.ProcessorABC):
    '''
    e.g. deltaR of leptons, min deltaR of lepton and jet

    '''
    def __init__(self, year=2018):
        
        self.year = year
        
        self._accumulator = processor.dict_accumulator( out_dict )
        


    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()
        
        output['total']['all'] += len(events)
        # use a very loose preselection to filter the events
        presel = ak.num(events.Jet)>2
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # load the config - probably not needed anymore
        cfg = loadConfig()
        
        ## Muons
        muon     = Collections(ev, "Muon", "tightTTH").get()
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightTTH").get()
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
        
        dilepton_mass = (leading_lepton+trailing_lepton).mass
        dilepton_pt = (leading_lepton+trailing_lepton).pt

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
        
        tau       = getTaus(ev)
        track     = getIsoTracks(ev)
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
        lep0pt    = ((ak.num(electron[(electron.pt>25)]) + ak.num(muon[(muon.pt>25)]))>0)
        lep1pt    = ((ak.num(electron[(electron.pt>20)]) + ak.num(muon[(muon.pt>20)]))>1)
        lepveto   = ((ak.num(vetoelectron) + ak.num(vetomuon))==2)
        
        
        selection = PackedSelection()
        selection.add('lepveto',       lepveto)
        selection.add('dilep',         dilep )
        selection.add('filter',        (filters) )
        selection.add('p_T(lep0)>25',  lep0pt )
        selection.add('p_T(lep1)>20',  lep1pt )
        selection.add('SS',            ( SSlepton | SSelectron | SSmuon) )
        selection.add('N_jet>3',       (ak.num(jet)>=4) )
        selection.add('N_central>2',   (ak.num(central)>=3) )
        selection.add('N_btag>0',      (ak.num(btag)>=1) )
        selection.add('N_fwd>0',       (ak.num(fwd)>=1 ))
        
        ss_reqs = ['lepveto', 'dilep', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20', 'SS']
        bl_reqs = ss_reqs + ['N_jet>3', 'N_central>2', 'N_btag>0', 'N_fwd>0']

        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_selection = selection.require(**ss_reqs_d)
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)

        weight = Weights( len(ev) )
        weight.add("weight", ev.weight)

        cutflow     = Cutflow(output, ev, weight=weight)
        cutflow_reqs_d = {}
        for req in bl_reqs:
            cutflow_reqs_d.update({req: True})
            cutflow.addRow( req, selection.require(**cutflow_reqs_d) )

        labels = {'topW_v3': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'ttbar': 4}
        if dataset in labels:
            label_mult = labels[dataset]
        else:
            label_mult = 5
        label = np.ones(len(ev[BL])) * label_mult

        output["lead_lep_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(leading_lepton[BL].pt, axis=1)))
        output["sublead_lep_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(trailing_lepton[BL].pt, axis=1)))
        output["lead_lep_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(leading_lepton[BL].eta, axis=1)))
        output["sublead_lep_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(trailing_lepton[BL].eta, axis=1)))

        output["lead_jet_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 0:1][BL].pt, axis=1)))
        output["sublead_jet_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 1:2][BL].pt, axis=1)))

        output["fwd_jet_p"] += processor.column_accumulator(ak.to_numpy(ak.flatten(j_fwd[BL].p, axis=1)))
        output["fwd_jet_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(j_fwd[BL].pt, axis=1)))
        output["fwd_jet_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(j_fwd[BL].eta, axis=1)))

        output["mjj_max"] += processor.column_accumulator(ak.to_numpy(ak.max(mjf[BL], axis=1)))
        output["delta_eta_jj"] += processor.column_accumulator(ak.to_numpy(ak.flatten(delta_eta[BL], axis=1)))

        output["met"] += processor.column_accumulator(ak.to_numpy(met_pt[BL]))
        output["ht"] += processor.column_accumulator(ak.to_numpy(ht[BL]))
        output["st"] += processor.column_accumulator(ak.to_numpy(st[BL]))
        output["n_jet"] += processor.column_accumulator(ak.to_numpy(ak.num(jet[BL])))
        output["n_fwd"] += processor.column_accumulator(ak.to_numpy(ak.num(fwd[BL])))
        output["n_central"] += processor.column_accumulator(ak.to_numpy(ak.num(central[BL])))
        output["n_tau"] += processor.column_accumulator(ak.to_numpy(ak.num(tau[BL])))
        output["n_track"] += processor.column_accumulator(ak.to_numpy(ak.num(track[BL])))
        
        output["dilepton_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(dilepton_pt[BL], axis=1)))
        output["dilepton_mass"] += processor.column_accumulator(ak.to_numpy(ak.flatten(dilepton_mass[BL], axis=1)))

        output["label"] += processor.column_accumulator(label)
        
        output["presel"]["all"] += len(ev[ss_selection])
        output["sel"]["all"] += len(ev[BL])

        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    import glob
    from klepto.archives import dir_archive
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    import pandas as pd

    overwrite = True
    year = 2018
    
    # load the config and the cache
    cfg = loadConfig()
    
    fileset = {
        #'topW_v2': fileset_2018['topW_v2'],
        'topW_v3': fileset_2018['topW_v3'], # 6x larger stats
        #'topW_v3': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWplusJetsToLNuEWK_5f_NLO_v2_RunIIAutumn18_NANO_v4/*_1.root'), # 6x larger stats
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
    
    print ("I'm running now")
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    output = processor.run_uproot_job(
        fileset,
        "Events",
        ML_preprocessor(year = year),
        exe,
        exe_args,
        chunksize=250000,
    )
    
    df_dict = {}
    for var in variables:
        df_dict.update({var: output[var].value})

    df_out = pd.DataFrame( df_dict )

    if overwrite:
        os.remove(os.path.expandvars('$TWHOME/ML/data/multiclass_input.h5'))

    df_out.to_hdf('data/multiclass_input.h5', key='df', format='table', mode='a', append=True)


