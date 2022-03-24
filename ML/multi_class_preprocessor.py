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
from Tools.ttH_lepton_scalefactors import *
from Tools.helpers import mt, get_four_vec, pad_and_flatten

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
    'rare':           processor.defaultdict_accumulator(int),
    'ttbar':          processor.defaultdict_accumulator(int),
    'ttbar1l_MG':     processor.defaultdict_accumulator(int),
}

variables = [
    'mjj_max',
    'delta_eta_jj',
    'met',
    'ht',
    'st',
    'n_jet',
    'n_btag',
    'n_fwd',
    'n_central',
    'n_tau',
    'n_track',
    'n_lep_tight',
    'n_lep',

    'fwd_jet_p',
    'fwd_jet_pt',
    'fwd_jet_eta',
    'fwd_jet_phi',
    'fwd_jet_energy',
    'fwd_jet_px',
    'fwd_jet_py',
    'fwd_jet_pz',

    'lead_jet_pt',
    'lead_jet_eta',
    'lead_jet_phi',

    'sublead_jet_pt',
    'sublead_jet_eta',
    'sublead_jet_phi',

    'lead_btag_pt',
    'lead_btag_eta',
    'lead_btag_phi',
    'lead_btag_energy',
    'lead_btag_px',
    'lead_btag_py',
    'lead_btag_pz',

    'sublead_btag_pt',
    'sublead_btag_eta',
    'sublead_btag_phi',
    'sublead_btag_energy',
    'sublead_btag_px',
    'sublead_btag_py',
    'sublead_btag_pz',

    'lead_lep_pt',
    'lead_lep_eta',
    'lead_lep_phi',
    'lead_lep_charge',
    'lead_lep_energy',
    'lead_lep_px',
    'lead_lep_py',
    'lead_lep_pz',

    'sublead_lep_pt',
    'sublead_lep_eta',
    'sublead_lep_phi',
    'sublead_lep_charge',
    'sublead_lep_energy',
    'sublead_lep_px',
    'sublead_lep_py',
    'sublead_lep_pz',

    'dilepton_mass',
    'dilepton_pt',
    'min_bl_dR',
    'min_mt_lep_met',
    'label',
    'label_cat',
    'weight',
]

for i in range(6):
    for j in ['pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz']:
        variables.append('j%s_%s'%(i, j))

for var in variables:
    out_dict.update({var: processor.column_accumulator(np.zeros(shape=(0,)))})


class ML_preprocessor(processor.ProcessorABC):
    '''
    e.g. deltaR of leptons, min deltaR of lepton and jet

    '''
    def __init__(self, year=2018):
        
        self.year = year
        
        self._accumulator = processor.dict_accumulator( out_dict )
        self.btagSF = btag_scalefactor(year)

        self.leptonSF = LeptonSF(year=year)


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
        
        gen_lep = ev.GenL
        
        ## Muons
        muon     = Collections(ev, "Muon", "vetoTTH").get()
        tightmuon = Collections(ev, "Muon", "tightSSTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "vetoTTH").get()
        tightelectron = Collections(ev, "Electron", "tightSSTTH").get()
        dielectron   = choose(electron, 2)
        SSelectron   = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
        ## Merge electrons and muons - this should work better now in ak1
        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

        lepton   = ak.concatenate([muon, electron], axis=1)
        lepton   = get_four_vec(lepton)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = get_four_vec(lepton[leading_lepton_idx])
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = get_four_vec(lepton[trailing_lepton_idx])
        
        dilepton_mass = (leading_lepton+trailing_lepton).mass
        dilepton_pt = (leading_lepton+trailing_lepton).pt
        dilepton_dR = delta_r(leading_lepton, trailing_lepton)

        mt_lep_met = mt(lepton.pt, lepton.phi, ev.MET.pt, ev.MET.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)

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
        fwd_cleaned = fwd[~match(fwd, getFwdJet(jet[:,0:5]), deltaRCut=0.1)]  # the leading forward jets that are not in the 5 leading jets overall
        
        tau       = getTaus(ev)
        track     = getIsoTracks(ev)
        ## forward jets
        j_fwd = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator

        high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]
        
        bl          = cross(lepton, high_score_btag)
        bl_dR       = delta_r(bl['0'], bl['1'])
        min_bl_dR   = ak.min(bl_dR, axis=1)

        jf          = cross(j_fwd, jet)
        mjf         = (jf['0']+jf['1']).mass
        j_fwd2      = jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'] # this is the jet that forms the largest invariant mass with j_fwd
        delta_eta   = ak.fill_none(ak.pad_none(abs(j_fwd2.eta - j_fwd.eta), 1, clip=True), 0)

        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        ## other variables
        ht = ak.sum(jet.pt, axis=1)
        st = met_pt + ht + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)
        
        ## event selectors
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        
        dilep     = ((ak.num(tightelectron) + ak.num(tightmuon))==2)
        lep0pt    = ((ak.num(electron[(electron.pt>25)]) + ak.num(muon[(muon.pt>25)]))>0)
        lep1pt    = ((ak.num(electron[(electron.pt>20)]) + ak.num(muon[(muon.pt>20)]))>1)
        lepveto   = ((ak.num(electron) + ak.num(muon))==2)
        
        
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
        
        #ss_reqs = ['lepveto', 'dilep', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20', 'SS']
        ss_reqs = ['lepveto', 'dilep', 'filter', 'p_T(lep0)>25', 'p_T(lep1)>20', 'SS']
        #bl_reqs = ss_reqs + ['N_jet>3', 'N_central>2', 'N_btag>0', 'N_fwd>0']
        bl_reqs = ss_reqs + ['N_jet>3', 'N_central>2', 'N_btag>0']

        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_selection = selection.require(**ss_reqs_d)
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)

        weight = Weights( len(ev) )

        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight)

            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)

            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))

            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))


        #cutflow     = Cutflow(output, ev, weight=weight)
        #cutflow_reqs_d = {}
        #for req in bl_reqs:
        #    cutflow_reqs_d.update({req: True})
        #    cutflow.addRow( req, selection.require(**cutflow_reqs_d) )

        labels = {'topW_v3': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'ttbar': 4, 'ttbar1l_MG': 4, 'DY': 6, 'topW_EFT_cp8':100 }
        if dataset in labels:
            label_mult = labels[dataset]
        else:
            label_mult = 5

        label = np.ones(len(ev[BL])) * label_mult

        n_nonprompt = (getNonPromptFromFlavour(tightelectron) + getNonPromptFromFlavour(tightmuon))[BL]
        n_chargeflip = (getChargeFlips(tightelectron, ev.GenPart) + getChargeFlips(tightmuon, ev.GenPart))[BL]
        n_genlep = ak.num(ev.GenL, axis=1)[BL]

        label_cat = (n_nonprompt>0)*100 + (n_chargeflip>0)*1000 + (n_genlep>2)*10 + np.ones(len(ev[BL])) # >1000 for charge flip, >100 for non prompt, >10 for more than 2 gen lep, 1 for prompt
        if dataset=='topW_v3':
            label_cat = np.ones(len(ev[BL])) * 0
        else:
            label_cat = 4*(label_cat>=1000) + 3*((label_cat>=100) & (label_cat<1000)) + 2*((label_cat>=10) & (label_cat<100)) + 1*(label_cat<10)  # this makes charge flip 4, nonprompt 3...
            label_cat = np.array(label_cat)

        output["n_lep"] += processor.column_accumulator(ak.to_numpy( (ak.num(electron) + ak.num(muon))[BL] ))
        output["n_lep_tight"] += processor.column_accumulator(ak.to_numpy( (ak.num(tightelectron) + ak.num(tightmuon))[BL] ))

        o_leading_lepton = get_four_vec(leading_lepton[BL])
        output["lead_lep_pt"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.pt, axis=1)))
        output["lead_lep_eta"]    += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.eta, axis=1)))
        output["lead_lep_phi"]    += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.phi, axis=1)))
        output["lead_lep_charge"] += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.charge, axis=1)))
        output["lead_lep_energy"] += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.energy, axis=1)))
        output["lead_lep_px"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.px, axis=1)))
        output["lead_lep_py"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.py, axis=1)))
        output["lead_lep_pz"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_leading_lepton.pz, axis=1)))

        o_trailing_lepton = get_four_vec(trailing_lepton[BL])
        output["sublead_lep_pt"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.pt, axis=1)))
        output["sublead_lep_eta"]    += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.eta, axis=1)))
        output["sublead_lep_phi"]    += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.phi, axis=1)))
        output["sublead_lep_charge"] += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.charge, axis=1)))
        output["sublead_lep_energy"] += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.energy, axis=1)))
        output["sublead_lep_px"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.px, axis=1)))
        output["sublead_lep_py"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.py, axis=1)))
        output["sublead_lep_pz"]     += processor.column_accumulator(ak.to_numpy(ak.flatten(o_trailing_lepton.pz, axis=1)))

        output["lead_jet_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 0:1][BL].pt, axis=1)))
        output["lead_jet_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 0:1][BL].eta, axis=1)))
        output["lead_jet_phi"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 0:1][BL].phi, axis=1)))

        output["sublead_jet_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 1:2][BL].pt, axis=1)))
        output["sublead_jet_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 1:2][BL].eta, axis=1)))
        output["sublead_jet_phi"] += processor.column_accumulator(ak.to_numpy(ak.flatten(jet[:, 1:2][BL].phi, axis=1)))

        for i in range(5):
            output["j%s_pt"%i]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].pt)))
            output["j%s_eta"%i]     += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].eta)))
            output["j%s_phi"%i]     += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].phi)))
            output["j%s_energy"%i]  += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].energy)))
            output["j%s_px"%i]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].px)))
            output["j%s_py"%i]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].py)))
            output["j%s_pz"%i]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(jet[:,i:i+1][BL].pz)))

        output["j5_pt"]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].pt)))
        output["j5_eta"]     += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].eta)))
        output["j5_phi"]     += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].phi)))
        output["j5_energy"]  += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].energy)))
        output["j5_px"]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].px)))
        output["j5_py"]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].py)))
        output["j5_pz"]      += processor.column_accumulator(ak.to_numpy(pad_and_flatten(fwd_cleaned[:,0:1][BL].pz)))

        output["lead_btag_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].pt, axis=1)))
        output["lead_btag_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].eta, axis=1)))
        output["lead_btag_phi"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].phi, axis=1)))
        output["lead_btag_energy"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].energy, axis=1)))
        output["lead_btag_px"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].px, axis=1)))
        output["lead_btag_py"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].py, axis=1)))
        output["lead_btag_pz"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 0:1][BL].pz, axis=1)))

        output["sublead_btag_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].pt, axis=1)))
        output["sublead_btag_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].eta, axis=1)))
        output["sublead_btag_phi"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].phi, axis=1)))
        output["sublead_btag_energy"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].energy, axis=1)))
        output["sublead_btag_px"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].px, axis=1)))
        output["sublead_btag_py"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].py, axis=1)))
        output["sublead_btag_pz"] += processor.column_accumulator(ak.to_numpy(ak.flatten(high_score_btag[:, 1:2][BL].pz, axis=1)))

        output["fwd_jet_p"]   += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].p, 1, clip=True), 0), axis=1)))
        output["fwd_jet_pt"]  += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].pt, 1, clip=True), 0), axis=1)))
        output["fwd_jet_eta"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].eta,1, clip=True), 0), axis=1)))
        output["fwd_jet_phi"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].phi,1, clip=True), 0), axis=1)))
        output["fwd_jet_energy"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].energy,1, clip=True), 0), axis=1)))
        output["fwd_jet_px"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].px,1, clip=True), 0), axis=1)))
        output["fwd_jet_py"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].py,1, clip=True), 0), axis=1)))
        output["fwd_jet_pz"] += processor.column_accumulator(ak.to_numpy(ak.flatten(ak.fill_none(ak.pad_none(j_fwd[BL].pz,1, clip=True), 0), axis=1)))

        output["mjj_max"] += processor.column_accumulator(ak.to_numpy(ak.fill_none(ak.max(mjf[BL], axis=1),0)))
        output["delta_eta_jj"] += processor.column_accumulator(ak.to_numpy(ak.flatten(delta_eta[BL], axis=1)))

        output["met"] += processor.column_accumulator(ak.to_numpy(met_pt[BL]))
        output["ht"] += processor.column_accumulator(ak.to_numpy(ht[BL]))
        output["st"] += processor.column_accumulator(ak.to_numpy(st[BL]))
        output["n_jet"] += processor.column_accumulator(ak.to_numpy(ak.num(jet[BL])))
        output["n_btag"] += processor.column_accumulator(ak.to_numpy(ak.num(btag[BL])))
        output["n_fwd"] += processor.column_accumulator(ak.to_numpy(ak.num(fwd[BL])))
        output["n_central"] += processor.column_accumulator(ak.to_numpy(ak.num(central[BL])))
        output["n_tau"] += processor.column_accumulator(ak.to_numpy(ak.num(tau[BL])))
        output["n_track"] += processor.column_accumulator(ak.to_numpy(ak.num(track[BL])))
        
        output["dilepton_pt"] += processor.column_accumulator(ak.to_numpy(ak.flatten(dilepton_pt[BL], axis=1)))
        output["dilepton_mass"] += processor.column_accumulator(ak.to_numpy(ak.flatten(dilepton_mass[BL], axis=1)))
        output["min_bl_dR"] += processor.column_accumulator(ak.to_numpy(min_bl_dR[BL]))
        output["min_mt_lep_met"] += processor.column_accumulator(ak.to_numpy(min_mt_lep_met[BL]))

        output["label"] += processor.column_accumulator(label)
        output["label_cat"] += processor.column_accumulator(label_cat)
        output["weight"] += processor.column_accumulator(weight.weight()[BL])
        
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
        ##'topW_v2': fileset_2018['topW_v2'],
        'topW_v3': fileset_2018['topW_v3'], # 6x larger stats
        #'topW_EFT_cp8': fileset_2018['topW_EFT_cp8']
        ##'topW_v3': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/ProjectMetis_TTWplusJetsToLNuEWK_5f_NLO_v2_RunIIAutumn18_NANO_v4/*_1.root'), # 6x larger stats
        'TTW': fileset_2018['TTW'],
        'TTZ': fileset_2018['TTZ'],
        'TTH': fileset_2018['TTH'],
        'ttbar': fileset_2018['ttbar'],
        'rare': fileset_2018['TTTT'] + fileset_2018['diboson'], # also contains triboson
        'DY': fileset_2018['DY'],
        ##'ttbar1l_MG': fileset_2018['ttbar1l_MG'],
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

    df_out.to_hdf('data/multiclass_input_v4.h5', key='df', format='table', mode='w')#, append=True)
