import os
import re
import datetime

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist, util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

import pandas as pd

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, prompt_no_conv, nonprompt_no_conv, external_conversion, fast_match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM, get_samples
from Tools.config_helpers import loadConfig, make_small, data_pattern, get_latest_output, load_yaml, data_path
from Tools.triggers import getFilters, getTriggers
from Tools.trigger_scalefactors import triggerSF
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer

class forward_jet_analysis(processor.ProcessorABC):
    def __init__(self,
                 year=2016,
                 variations=[],
                 accumulator={},
                 evaluate=False,
                 training='v8',
                 dump=False,
                 era=None,
                 reweight=1,
                 ):

        self.variations = variations
        self.year = year
        self.era = era

        self.btagSF = btag_scalefactor(year)
        self.leptonSF = LeptonSF(year=year)
        self.triggerSF = triggerSF(year=year)

        self.reweight = reweight
        
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
        
        ## Muons
        muon     = Collections(ev, "Muon", "tightSSTTH").get()
        muon['p4'] = get_four_vec_fromPtEtaPhiM(muon, get_pt(muon), muon.eta, muon.phi, muon.mass, copy=False)
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        OSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)<0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightSSTTH").get()
        electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False)
        vetoelectron = Collections(ev, "Electron", "vetoTTH").get()
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
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.p4.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.p4.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]
        
        dilepton_mass = (leading_lepton+trailing_lepton).mass
        dilepton_pt = (leading_lepton+trailing_lepton).pt
        dilepton_dR = delta_r(leading_lepton, trailing_lepton)
        
        tau       = getTaus(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] # remove taus that overlap with muons
        tau       = tau[~match(tau, electron, deltaRCut=0.4)] # remove taus that overlap with electrons

        if not re.search(data_pattern, dataset):
            gen = ev.GenPart
            gen_photon = gen[gen.pdgId==22]
            external_conversions = external_conversion(lepton, gen_photon)
            conversion_veto = ((ak.num(external_conversions))==0)
            conversion_req = ((ak.num(external_conversions))>0)

            def add_conversion_req(dataset, req):
                if dataset.count("TTGamma") or dataset.count("WGTo") or dataset.count("ZGTo") or dataset.count("WZG_"):
                    return (req & conversion_req)
                elif dataset.count("TTTo") or dataset.count('DYJets'):
                    return (req & conversion_veto)
                return req
               
        # define the weight
        weight = Weights( len(ev) )

        n_ele = ak.num(electron, axis=1)  # This is useful to split into ee/emu/mumu

        # FIXME: reintegrate this into the loop
        #if re.search(data_pattern, dataset):
        #    #rle = ak.to_numpy(ak.zip([ev.run, ev.luminosityBlock, ev.event]))
        #    run_ = ak.to_numpy(ev.run)
        #    lumi_ = ak.to_numpy(ev.luminosityBlock)
        #    event_ = ak.to_numpy(ev.event)
        #    output['%s_run'%dataset] += processor.column_accumulator(run_[BL])
        #    output['%s_lumi'%dataset] += processor.column_accumulator(lumi_[BL])
        #    output['%s_event'%dataset] += processor.column_accumulator(event_[BL])


        if re.search(data_pattern, dataset):
            variations = self.variations[:1]
        else:
            variations = self.variations

        for var in variations:

            pt_var   = var['pt_var']
            var_name = var['name']
            shift    = var['weight']

            met = getMET(ev, pt_var=pt_var)

            ## Jets
            jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var=pt_var)
            jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
            jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

            central   = jet[(abs(jet.eta)<2.4)]
            btag      = getBTagsDeepFlavB(jet, era=era, year=self.year)
            light     = getBTagsDeepFlavB(jet, era=era, year=self.year, invert=True)
            light_central = light[(abs(light.eta)<2.5)]
            fwd       = getFwdJet(light)

            ## forward jets
            high_p_fwd   = fwd[ak.singletons(ak.argmax(fwd.p4.p, axis=1))] # highest momentum spectator
            high_pt_fwd  = fwd[ak.singletons(ak.argmax(fwd.p4.pt, axis=1))]  # highest transverse momentum spectator
            high_eta_fwd = fwd[ak.singletons(ak.argmax(abs(fwd.p4.eta), axis=1))] # most forward spectator
        
            ## Get the two leading b-jets in terms of btag score
            high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

            jf          = cross(high_p_fwd, jet)
            mjf         = (jf['0'].p4+jf['1'].p4).mass
            deltaEta    = abs(high_p_fwd.eta - jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'].p4.eta)
            deltaEtaMax = ak.max(deltaEta, axis=1)
            mjf_max     = ak.max(mjf, axis=1)

            jj          = choose(jet, 2)
            mjj_max     = ak.max((jj['0'].p4+jj['1'].p4).mass, axis=1)

            bl          = cross(lepton, high_score_btag)
            bl_dR       = delta_r(bl['0'], bl['1'])
            min_bl_dR   = ak.min(bl_dR, axis=1)

            #mt_lep_met = mt(lepton.p4.pt, lepton.phi, ev.MET.T1_pt, ev.MET.phi)
            mt_lep_met = mt(lepton.p4.pt, lepton.phi, met.pt, met.phi)
            min_mt_lep_met = ak.min(mt_lep_met, axis=1)

            # define the weight
            weight = Weights( len(ev) )

            if not re.search(data_pattern, dataset):
                # lumi weight
                weight.add("weight", ev.genWeight)

                if isinstance(self.reweight[dataset], int) or isinstance(self.reweight[dataset], float):
                    pass  # NOTE: this can be implemented later
                    #if self.reweight != 1:
                    #    weight.add("reweight", self.reweight[dataset])
                else:
                    weight.add("reweight", getattr(ev, self.reweight[dataset][0])[:,self.reweight[dataset][1]])

                # PU weight
                weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)

                # b-tag SFs # NOTE this is not super sophisticated rn, but we need more than two shifts
                if var['name'] == 'l_up':
                    weight.add("btag", self.btagSF.Method1a(btag, light_central, b_direction='central', c_direction='up'))
                elif var['name'] == 'l_down':
                    weight.add("btag", self.btagSF.Method1a(btag, light_central, b_direction='central', c_direction='down'))
                elif var['name'] == 'b_up':
                    weight.add("btag", self.btagSF.Method1a(btag, light_central, b_direction='up', c_direction='central'))
                elif var['name'] == 'b_down':
                    weight.add("btag", self.btagSF.Method1a(btag, light_central, b_direction='down', c_direction='central'))
                else:
                    weight.add("btag", self.btagSF.Method1a(btag, light_central))

                # lepton SFs
                weight.add("lepton", self.leptonSF.get(electron, muon))
            
            sel = Selection(
                dataset = dataset,
                events = ev,
                year = self.year,
                era = self.era,
                ele = electron,
                ele_veto = vetoelectron,
                mu = muon,
                mu_veto = vetomuon,
                jet_all = jet,
                jet_central = central,
                jet_btag = btag,
                jet_fwd = fwd,
                jet_light = light,
                met = met,
            )

            if var_name == 'central':
                cutflow = Cutflow(output, ev, weight=weight)
                BL = sel.dilep_baseline(cutflow=cutflow, SS=False)

            BL = sel.dilep_baseline(SS=False, omit=['N_fwd>0', 'N_central>2'])
            if not re.search(data_pattern, dataset):
                BL = add_conversion_req(dataset, BL)
            output['N_jet'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                multiplicity=ak.num(jet)[BL],
                weight=weight.weight()[BL],
            )

            output['N_fwd'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                multiplicity=ak.num(fwd)[BL],
                weight=weight.weight()[BL],
            )

            output['dilep_pt'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=ak.flatten(dilepton_pt[BL]),
                weight=weight.weight()[BL],
            )
            output['dilep_mass'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                mass=ak.flatten(dilepton_mass[BL]),
                weight=weight.weight()[BL],
            )

            output['PV_npvs'].fill(
                dataset=dataset,
                systematic = var_name,
                multiplicity=ev.PV[BL].npvs,
                weight=weight.weight()[BL],
            )

            output['PV_npvsGood'].fill(
                dataset=dataset,
                systematic = var_name,
                multiplicity=ev.PV[BL].npvsGood,
                weight=weight.weight()[BL],
            )

            output['N_tau'].fill(
                dataset=dataset,
                systematic = var_name,
                multiplicity=ak.num(tau)[BL],
                weight=weight.weight()[BL],
            )

            #output['N_track'].fill(dataset=dataset, multiplicity=ak.num(track)[BL], weight=weight.weight()[BL])

            BL_minusNb = sel.dilep_baseline(SS=False, omit=['N_btag>0','N_central>2', 'N_fwd>0'])
            if not re.search(data_pattern, dataset):
                BL_minusNb = add_conversion_req(dataset, BL_minusNb)
            output['N_b'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_minusNb],
                multiplicity=ak.num(btag)[BL_minusNb],
                weight=weight.weight()[BL_minusNb],
            )

            # This is the real baseline, although N_fwd is removed for training. FIXME: decide what to call "baseline"
            BL = sel.dilep_baseline(SS=False)
            if not re.search(data_pattern, dataset):
                BL = add_conversion_req(dataset, BL)

            output['mjf_max'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                mass=mjf_max[BL],
                weight=weight.weight()[BL],
            )

            output['mjj_max'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                mass=mjj_max[BL],
                weight=weight.weight()[BL],
            )

            output['deltaEta'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                eta=deltaEtaMax[BL],
                weight=weight.weight()[BL],
            )

            output['min_bl_dR'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                eta=min_bl_dR[BL],
                weight=weight.weight()[BL],
            )

            output['min_mt_lep_met'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=min_mt_lep_met[BL],
                weight=weight.weight()[BL],
            )

            output['lead_lep'].fill(
                dataset = dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].p4.pt)),
                eta = ak.to_numpy(ak.flatten(leading_lepton[BL].p4.eta)),
                #phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
                weight = weight.weight()[BL]
               )

            output['trail_lep'].fill(
                dataset = dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].p4.pt)),
                eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].p4.eta)),
                #phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
                weight = weight.weight()[BL]
               )

            output['lead_jet'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=ak.flatten(jet[:, 0:1][BL].p4.pt),
                eta=ak.flatten(jet[:, 0:1][BL].p4.eta),
                weight=weight.weight()[BL],
            )

            output['sublead_jet'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=ak.flatten(jet[:, 1:2][BL].p4.pt),
                eta=ak.flatten(jet[:, 1:2][BL].p4.eta),
                weight=weight.weight()[BL],
            )

            output['fwd_jet'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=ak.flatten(high_p_fwd[BL].p4.pt),
                eta=ak.flatten(high_p_fwd[BL].p4.eta),
                weight=weight.weight()[BL],
            )

            BL_minusMET = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['MET>30', 'N_central>2','N_fwd>0'])
            if not re.search(data_pattern, dataset):
                BL_minusMET = add_conversion_req(dataset, BL_minusMET)
            output['MET'].fill(
                dataset = dataset,
                systematic = var_name,
                #n_ele = n_ele[BL_minusMET],
                pt  = met[BL_minusMET].pt,
                phi  = met[BL_minusMET].phi,
                weight = weight.weight()[BL_minusMET]
            )

#            '''
#            output['b1_'+var].fill(
#                dataset = dataset,
#                pt  = ak.flatten(high_score_btag[:, 0:1].p4.pt[:, 0:1][BL]),
#                eta = ak.flatten(high_score_btag[:, 0:1].p4.eta[:, 0:1][BL]),
#                phi = ak.flatten(high_score_btag[:, 0:1].phi[:, 0:1][BL]),
#                weight = weight.weight()[BL]
#            )
#            '''

        return output

    def postprocess(self, accumulator):
        return accumulator



if __name__ == '__main__':
    
    #from processor.default_accumulators import *

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=None, help="Rerun or try using existing results??")
    argParser.add_argument('--keep', action='store_true', default=None, help="Keep/use existing results??")
    argParser.add_argument('--central', action='store_true', default=None, help="Only run the central value (no systematics)")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on a DASK cluster?")
    argParser.add_argument('--profile', action='store_true', default=None, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=None, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2016', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    argParser.add_argument('--sample', action='store', default='all', )
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = args.rerun
    small       = args.small
    verysmall   = args.verysmall
    
    if verysmall:
        small = True
    
    year        = int(args.year[0:4])
    ul          = "UL%s"%(args.year[2:])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config
    cfg = loadConfig()

    samples = get_samples("samples_%s.yaml"%ul)
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    if args.sample == 'MCall':
        #sample_list = ['DY', 'topW', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
        sample_list = ['DY', 'topW', 'top', 'TTW', 'TTZ', 'XG', 'rare', 'diboson']
    elif args.sample == 'data':
        if year == 2018:
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

        from Tools.nano_mapping import make_fileset
        from default_accumulators import add_processes_to_output, desired_output

        fileset = make_fileset([sample], samples, year=ul, skim=True, small=small, n_max=1)

        add_processes_to_output(fileset, desired_output)

        variations = [
            {'name': 'central',     'ext': '',                  'weight': None,   'pt_var': 'pt_nom'},
            {'name': 'jes_up',      'ext': '_pt_jesTotalUp',    'weight': None,   'pt_var': 'pt_jesTotalUp'},
            {'name': 'jes_down',    'ext': '_pt_jesTotalDown',  'weight': None,   'pt_var': 'pt_jesTotalDown'},
            {'name': 'PU_up',       'ext': '_PUUp',             'weight': 'PUUp', 'pt_var': 'pt_nom'},
            {'name': 'PU_down',     'ext': '_PUDown',           'weight': 'PUDown', 'pt_var': 'pt_nom'},
            {'name': 'b_up',        'ext': '_bUp',              'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'b_down',      'ext': '_bDown',            'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'l_up',        'ext': '_lUp',              'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'l_down',      'ext': '_lDown',            'weight': None,    'pt_var': 'pt_nom'},
           ]

        if args.central: variations = variations[:1]

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

        # add some histograms that we defined in the processor
        # everything else is taken the default_accumulators.py
        from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, one_axis, mass_axis
        from processor.default_accumulators import systematic_axis, eft_axis, charge_axis, n_ele_axis, eta_axis, delta_eta_axis, pt_axis
        desired_output.update({
            "ST": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "HT": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "LT": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "N_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_b": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_fwd": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_tau": hist.Hist("Counts", dataset_axis, systematic_axis, multiplicity_axis),
            "dilep_pt": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis),
            "dilep_mass": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, mass_axis),
            "mjf_max": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, mass_axis),
            "mjj_max": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, mass_axis),
            "deltaEta": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, delta_eta_axis),
            "min_bl_dR": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, delta_eta_axis),
            "min_mt_lep_met": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis),
            "lead_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "trail_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "lead_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "sublead_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),
            "fwd_jet": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis),

           })

        for rle in ['run', 'lumi', 'event']:
            desired_output.update({
                    'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                    'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                    'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
            })

        print ("I'm running now")

        runner = processor.Runner(
            exe,
            #retries=3,
            schema=NanoAODSchema,
            chunksize=50000,
            maxchunks=None,
           )

        # define the cache name
        cache_name = f'OS_analysis_{sample}_{year}{era}'
        # find an old existing output
        output = get_latest_output(cache_name, cfg)

        if overwrite or output is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_name += f'_{timestamp}.coffea'
            if small: cache_name += '_small'
            cache = os.path.join(os.path.expandvars(cfg['caches']['base']), cache_name)

            print (variations)

            output = runner(
                fileset,
                treename="Events",
                processor_instance=forward_jet_analysis(
                    year=year,
                    variations=variations,
                    accumulator=desired_output,
                    evaluate=args.evaluate,
                    training=args.training,
                    era=era,
                    reweight=reweight,
                ),
            )

            util.save(output, cache)

        if not local:
            # clean up the DASK workers. this partially frees up memory on the workers
            c.cancel(output)
            # NOTE: this really restarts the cluster, but is the only fully effective
            # way of deallocating all the accumulated memory...
            c.restart()
