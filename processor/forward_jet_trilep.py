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

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, prompt_no_conv, nonprompt_no_conv, external_conversion, fast_match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM, get_samples
from Tools.config_helpers import loadConfig, make_small, data_pattern, get_latest_output, load_yaml, data_path
from Tools.triggers import getFilters, getTriggers
from Tools.trigger_scalefactors import triggerSF
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.pileup import pileup
from Tools.selections import Selection, get_pt

import warnings
warnings.filterwarnings("ignore")



class forwardJetAnalyzer(processor.ProcessorABC):
    def __init__(self,
                 year=2016,
                 variations=[],
                 accumulator={},
                 evaluate=False,
                 training='v8',
                 era=None,
                 reweight=1,
                 ):

        self.variations = variations
        self.year = year
        self.era = era

        self.btagSF = btag_scalefactor(year, era=era)
        self.leptonSF = LeptonSF(year=year, era=era)
        self.triggerSF = triggerSF(year=year)
        self.pu = pileup(year=year, UL=True, era=era)

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
        vetomuon['p4'] = get_four_vec_fromPtEtaPhiM(vetomuon, get_pt(vetomuon), vetomuon.eta, vetomuon.phi, vetomuon.mass, copy=False)
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        OSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)<0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightSSTTH").get()
        electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False)
        vetoelectron = Collections(ev, "Electron", "vetoTTH").get()
        vetoelectron['p4'] = get_four_vec_fromPtEtaPhiM(vetoelectron, get_pt(vetoelectron), vetoelectron.eta, vetoelectron.phi, vetoelectron.mass, copy=False)
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
        second_lepton = lepton[~(trailing_lepton_idx & leading_lepton_idx)]  # FIXME I don't think this does what it was intended to

        tau       = getTaus(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] # remove taus that overlap with muons
        tau       = tau[~match(tau, electron, deltaRCut=0.4)] # remove taus that overlap with electrons
        #

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

        #

        vetolepton   = ak.concatenate([vetomuon, vetoelectron], axis=1)
        trilep = choose(lepton, 3)
        trilep_m = ak.max(trilep.mass, axis=1)  # there should only be one trilep mass anyway

        dimu_veto = choose(vetomuon,2)
        diele_veto = choose(vetoelectron,2)
        OS_dimu_veto = dimu_veto[(dimu_veto['0'].charge*dimu_veto['1'].charge < 0)]
        OS_diele_veto = diele_veto[(diele_veto['0'].charge*diele_veto['1'].charge < 0)]

        OS_dimuon_bestZmumu = OS_dimu_veto[ak.singletons(ak.argmin(abs(OS_dimu_veto.mass-91.2), axis=1))]
        OS_dielectron_bestZee = OS_diele_veto[ak.singletons(ak.argmin(abs(OS_diele_veto.mass-91.2), axis=1))]
        OS_dilepton_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.mass, OS_dielectron_bestZee.mass], axis=1), 1, clip=True), -1)

        OS_dilepton_pt = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.pt, OS_dielectron_bestZee.pt], axis=1), 1, clip=True), -1)
        OS_dilepton_eta = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.eta, OS_dielectron_bestZee.eta], axis=1), 1, clip=True), -1)

        OS_dilepton_all_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimu_veto.mass, OS_diele_veto.mass], axis=1), 1, clip=True), -1)

        SFOS = ak.concatenate([OS_diele_veto, OS_dimu_veto], axis=1)
        OS_dimu_veto2 = OS_dimu_veto[ak.num(SFOS)>1]
        OS_diele_veto2 = OS_diele_veto[ak.num(SFOS)>1]
        OS_dimuon_worstZmumu = OS_dimu_veto[ak.singletons(ak.argmax(abs(OS_dimu_veto.mass-91.2), axis=1))]
        OS_dielectron_worstZee = OS_diele_veto[ak.singletons(ak.argmax(abs(OS_diele_veto.mass-91.2), axis=1))]
        OS_dilepton_worst_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_worstZmumu.mass, OS_dielectron_worstZee.mass], axis=1), 1, clip=True), -1)

        OS_min_mass = ak.fill_none(ak.min(ak.concatenate([OS_dimu_veto.mass, OS_diele_veto.mass], axis=1), axis=1),0)


        # define the weight
        weight = Weights( len(ev) )

        n_ele = ak.num(electron, axis=1)  # This is useful to split into ee/emu/mumu

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

                ## PU weight
                weight.add("PU",
                           self.pu.reweight(ev.Pileup.nTrueInt.to_numpy()),
                           weightUp = self.pu.reweight(ev.Pileup.nTrueInt.to_numpy(), to='up'),
                           weightDown = self.pu.reweight(ev.Pileup.nTrueInt.to_numpy(), to='down'),
                           shift=False,
                           )

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
                if var['name'] == 'ele_up':
                    weight.add("lepton", self.leptonSF.get(electron, muon, variation='up', collection='ele'))
                elif var['name'] == 'ele_down':
                    weight.add("lepton", self.leptonSF.get(electron, muon, variation='down', collection='ele'))
                elif var['name'] == 'mu_up':
                    weight.add("lepton", self.leptonSF.get(electron, muon, variation='up', collection='mu'))
                elif var['name'] == 'mu_down':
                    weight.add("lepton", self.leptonSF.get(electron, muon, variation='down', collection='mu'))
                else:
                    weight.add("lepton", self.leptonSF.get(electron, muon))

                # trigger SFs
                weight.add("trigger", self.triggerSF.get(electron, muon))

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
                BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0'])

            # Selection for ttZ / tZq / Z+V/VV region
            BL = sel.trilep_baseline(omit=['N_fwd>0'])
            if not re.search(data_pattern, dataset):
                BL = add_conversion_req(dataset, BL)

            # Selection for conversion region
            BL_offZ = sel.trilep_baseline(
                add=['offZ', 'N_btag=0', 'N_jet>0', 'N_central>0'],
                omit=['N_fwd>0', 'onZ', 'MET>50', 'N_jet>2', 'N_central>1'],
            )

            if not re.search(data_pattern, dataset):
                BL_offZ = add_conversion_req(dataset, BL_offZ)


            output['lead_lep'].fill(
                dataset = dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
                eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
                phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
                weight = weight.weight()[BL]
            )

            output['trail_lep'].fill(
                dataset = dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
                eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
                phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
                weight = weight.weight()[BL]
            )

            output['second_lep'].fill(
                dataset = dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt  = ak.to_numpy(ak.flatten(second_lepton[BL].pt)),
                eta = ak.to_numpy(ak.flatten(second_lepton[BL].eta)),
                phi = ak.to_numpy(ak.flatten(second_lepton[BL].phi)),
                weight = weight.weight()[BL]
            )

            output['N_jet_offZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_offZ],
                multiplicity=ak.num(jet)[BL_offZ],
                weight=weight.weight()[BL_offZ],
            )

            output['N_jet_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                multiplicity=ak.num(jet)[BL],
                weight=weight.weight()[BL],
            )


            output['N_b_offZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_offZ],
                multiplicity=ak.num(btag)[BL_offZ],
                weight=weight.weight()[BL_offZ],
            )

            output['N_b_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                multiplicity=ak.num(btag)[BL],
                weight=weight.weight()[BL],
            )


            output['N_fwd_offZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_offZ],
                multiplicity=ak.num(fwd)[BL_offZ],
                weight=weight.weight()[BL_offZ],
            )

            output['N_fwd_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                multiplicity=ak.num(fwd)[BL],
                weight=weight.weight()[BL],
            )

            output['min_mass_SFOS'].fill(
                dataset=dataset,
                mass=(OS_min_mass[BL]),
                weight=weight.weight()[BL],
            )

            output['dilep_pt_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                pt=ak.flatten(OS_dilepton_pt[BL]),
                weight=weight.weight()[BL],
            )

            output['dilep_eta_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                eta=ak.flatten(OS_dilepton_eta[BL]),
                weight=weight.weight()[BL],
            )

            output['M3l_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                mass=(trilep_m[BL]),
                weight=weight.weight()[BL],
            )

            output['M3l_offZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_offZ],
                mass=(trilep_m[BL_offZ]),
                weight=weight.weight()[BL_offZ],
            )

            output['best_M_ll_onZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL],
                mass=ak.flatten(OS_dilepton_mass[BL]),
                weight=weight.weight()[BL],
            )

            output['best_M_ll_offZ'].fill(
                dataset=dataset,
                systematic = var_name,
                n_ele = n_ele[BL_offZ],
                mass=ak.flatten(OS_dilepton_mass[BL_offZ]),
                weight=weight.weight()[BL_offZ],
            )


        return output

    def postprocess(self, accumulator):
        return accumulator



if __name__ == '__main__':
    
    from processor.default_accumulators import *

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
    argParser.add_argument('--workers', action='store', default=10, help="How many threads for local running?")
    argParser.add_argument('--sample', action='store', default='all', )
    argParser.add_argument('--buaf', action='store', default="false", help="Run on BU AF")
    argParser.add_argument('--skim', action='store', default="topW_v0.7.1_SS", help="Define the skim to run on")
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
        sample_list = ['DY', 'topW_lep', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
        #sample_list = ['DY', 'topW', 'top', 'TTW', 'TTZ', 'XG', 'rare', 'diboson']
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

        fileset = make_fileset(
            [sample],
            samples,
            year=ul,
            skim=args.skim,
            small=small,
            n_max=1,
            buaf=args.buaf,
            merged=True,
        )

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
            {'name': 'ele_up',      'ext': '_eleUp',            'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'ele_down',    'ext': '_eleDown',          'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'mu_up',       'ext': '_muUp',             'weight': None,    'pt_var': 'pt_nom'},
            {'name': 'mu_down',     'ext': '_muDown',           'weight': None,    'pt_var': 'pt_nom'},
           ]

        if args.central: variations = variations[:1]

        if local:# and not profile:
            exe = processor.FuturesExecutor(workers=int(args.workers))

        elif iterative:
            exe = processor.IterativeExecutor()

        else:
            from Tools.helpers import get_scheduler_address
            from dask.distributed import Client, progress

            scheduler_address = get_scheduler_address()
            c = Client(scheduler_address)

            exe = processor.DaskExecutor(client=c, status=True, retries=3)


        desired_output.update({
            "lead_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis, phi_axis),
            "trail_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis, phi_axis),
            "second_lep": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis, eta_axis, phi_axis),
            "N_jet_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_jet_offZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_b_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_b_offZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_fwd_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "N_fwd_offZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, multiplicity_axis),
            "M3l_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, ext_mass_axis),
            "M3l_offZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, ext_mass_axis),
            "ST": hist.Hist("Counts", dataset_axis, ht_axis),
            "HT": hist.Hist("Counts", dataset_axis, ht_axis),
            "LT": hist.Hist("Counts", dataset_axis, ht_axis),
            "dilep_pt_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, pt_axis),
            "dilep_eta_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, eta_axis),
            "min_mass_SFOS": hist.Hist("Counts", dataset_axis, mass_axis),
            "best_M_ll_onZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, mass_axis),
            "best_M_ll_offZ": hist.Hist("Counts", dataset_axis, systematic_axis, n_ele_axis, mass_axis),
            "M_ll_worst": hist.Hist("Counts", dataset_axis, mass_axis),
            "M_ll_all": hist.Hist("Counts", dataset_axis, mass_axis),
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
        cache_name = f'trilep_analysis_{sample}_{year}{era}'
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
                processor_instance=forwardJetAnalyzer(
                    year=year,
                    variations=variations,
                    accumulator=desired_output,
                    evaluate=args.evaluate,
                    #training=args.training,
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
