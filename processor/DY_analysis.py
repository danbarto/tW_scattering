import os
import re
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, prompt_no_conv, nonprompt_no_conv, external_conversion, fast_match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM
from Tools.config_helpers import loadConfig, make_small, data_pattern
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer


class DY_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False, era=None, hyperpoly=None, points=[[]]):
        self.variations = variations

        self.year = year
        self.era = era  # this is here for 2016 APV
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        
        self.btagSF = btag_scalefactor(year)  # this might be the only place where era in 2016 matters
        
        self.leptonSF = LeptonSF(year=year)

        self.nonpromptWeight = NonpromptWeight(year=year)
        self.chargeflipWeight = charge_flip(year=year)

        self.hyperpoly = hyperpoly
        self.points = points
        
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
        
        if not re.search(data_pattern, dataset):
            ## Generated leptons
            gen_lep = ev.GenL
            leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
            trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ####################
        ### Reco objects ###
        ####################

        # Get the leptons. This has changed a couple of times now, but we are using fakeable objects as baseline leptons.
        # The added p4 instance has the corrected pt (conePt for fakeable) and should be used for any following selection or calculation
        # Any additional correction (if we choose to do so) should be added here, e.g. Rochester corrections, ...
        ## Muons
        mu_v     = Collections(ev, "Muon", "vetoTTH", year=self.year).get()  # these include all muons, tight and fakeable
        mu_t     = Collections(ev, "Muon", "tightSSTTH", year=self.year).get()
        mu_f     = Collections(ev, "Muon", "fakeableSSTTH", year=self.year).get()
        muon     = ak.concatenate([mu_t, mu_f], axis=1)
        muon['p4'] = get_four_vec_fromPtEtaPhiM(muon, get_pt(muon), muon.eta, muon.phi, muon.mass, copy=False) #FIXME new
        
        ## Electrons
        el_v        = Collections(ev, "Electron", "vetoTTH", year=self.year).get()
        el_t        = Collections(ev, "Electron", "tightSSTTH", year=self.year).get()
        el_f        = Collections(ev, "Electron", "fakeableSSTTH", year=self.year).get()
        electron    = ak.concatenate([el_t, el_f], axis=1)
        electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False) #FIXME new
        
        if not re.search(data_pattern, dataset):
            gen_photon = ev.GenPart[ev.GenPart.pdgId==22]
            #el_t_p  = prompt(el_t)
            el_t_p  = prompt_no_conv(el_t, gen_photon)
            el_t_np = nonprompt_no_conv(el_t, gen_photon)
            el_t_conv = external_conversion(el_t, gen_photon)
            #el_t_np = nonprompt(el_t)
            el_f_p  = prompt(el_f)
            el_f_np = nonprompt_no_conv(el_f, gen_photon)
            #el_f_np = nonprompt(el_f)
            #mu_t_p  = prompt(mu_t)
            mu_t_p  = prompt_no_conv(mu_t, gen_photon)
            mu_t_np = nonprompt_no_conv(mu_t, gen_photon)
            mu_t_conv = external_conversion(mu_t, gen_photon)
            #mu_t_np = nonprompt(mu_t)
            mu_f_p  = prompt(mu_f)
            mu_f_np = nonprompt_no_conv(mu_f, gen_photon)
            #mu_f_np = nonprompt(mu_f)

            is_flipped = ( (el_t_p.matched_gen.pdgId*(-1) == el_t_p.pdgId) & (abs(el_t_p.pdgId) == 11) )
            el_t_p_cc  = el_t_p[~is_flipped]  # this is tight, prompt, and charge consistent
            el_t_p_cf  = el_t_p[is_flipped]  # this is tight, prompt, and charge flipped


        ## Merge electrons and muons. These are fakeable leptons now
        lepton   = ak.concatenate([muon, electron], axis=1)
        #lead_leptons = lepton[ak.argsort(lepton.p4.pt)][:,:3]
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.p4.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.p4.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]

        dilepton_mass = (leading_lepton.p4 + trailing_lepton.p4).mass
        dilepton_pt = (leading_lepton.p4 + trailing_lepton.p4).pt
        #dilepton_dR = delta_r(leading_lepton, trailing_lepton)
        dilepton_dR = leading_lepton.p4.delta_r(trailing_lepton.p4)
        
        lepton_pdgId_pt_ordered = ak.fill_none(ak.pad_none(lepton[ak.argsort(lepton.p4.pt, ascending=False)].pdgId, 2, clip=True), 0)
        
        if not re.search(data_pattern, dataset):
            n_nonprompt = getNonPromptFromFlavour(electron) + getNonPromptFromFlavour(muon)
            n_chargeflip = getChargeFlips(electron, ev.GenPart) + getChargeFlips(muon, ev.GenPart)
            gp = ev.GenPart
            gp_e = gp[((abs(gp.pdgId)==11)&(gp.status==1)&((gp.statusFlags&(1<<0))==1)&(gp.statusFlags&(1<<8)==256))]
            gp_m = gp[((abs(gp.pdgId)==13)&(gp.status==1)&((gp.statusFlags&(1<<0))==1)&(gp.statusFlags&(1<<8)==256))]
            n_gen_lep = ak.num(gp_e) + ak.num(gp_m)
        else:
            n_gen_lep = np.zeros(len(ev))

        LL = (n_gen_lep > 2)  # this is the classifier for LL events (should mainly be ttZ/tZ/WZ...)

        ## Tau and other stuff
        tau       = getTaus(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] 
        tau       = tau[~match(tau, electron, deltaRCut=0.4)]

        track     = getIsoTracks(ev)

        pt_var = 'pt_nom'
        shift = None

        met = getMET(ev, pt_var=pt_var)

        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var=pt_var)
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        central   = jet[(abs(jet.eta)<2.4)]
        btag      = getBTagsDeepFlavB(jet, year=self.year) # should study working point for DeepJet
        light     = getBTagsDeepFlavB(jet, year=self.year, invert=True)
        fwd       = getFwdJet(light)
        #fwd_noPU  = getFwdJet(light, puId=False)
        
        high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

        bl          = cross(lepton, high_score_btag)
        bl_dR       = delta_r(bl['0'], bl['1'])
        min_bl_dR   = ak.min(bl_dR, axis=1)

        ## forward jets
        j_fwd = fwd[ak.singletons(ak.argmax(fwd.p4.p, axis=1))] # highest momentum spectator

        # try to get either the most forward light jet, or if there's more than one with eta>1.7, the highest pt one
        most_fwd = light[ak.argsort(abs(light.eta))][:,0:1]
        #most_fwd = light[ak.singletons(ak.argmax(abs(light.eta)))]
        best_fwd = ak.concatenate([j_fwd, most_fwd], axis=1)[:,0:1]
        
        jf          = cross(j_fwd, jet)
        mjf         = (jf['0'].p4+jf['1'].p4).mass
        j_fwd2      = jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'] # this is the jet that forms the largest invariant mass with j_fwd
        delta_eta   = abs(j_fwd2.eta - j_fwd.eta)

        ## other variables
        ht = ak.sum(jet.p4.pt, axis=1)
        st = met.pt + ht + ak.sum(lepton.p4.pt, axis=1)
        lt = met.pt + ak.sum(lepton.p4.pt, axis=1)

        mt_lep_met = mt(lepton.p4.pt, lepton.p4.phi, met.pt, met.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)
        
        # define the weight
        weight = Weights( len(ev) )

        if not re.search(data_pattern, dataset):
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            
            # PU weight
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        
        if dataset=='topW_full_EFT':
            for point in self.points:
                point['weight'] = Weights( len(ev) )
                point['weight'].add("EFT", self.hyperpoly.eval(ev.Pol, point['point']))


        cutflow     = Cutflow(output, ev, weight=weight)

        # slightly restructured
        # calculate everything from loose, require two tights on top
        # since n_tight == n_loose == 2, the tight and loose leptons are the same in the end

        # in this selection we'll get events with exactly two fakeable+tight and two loose leptons.
        sel = Selection(
            dataset = dataset,
            events = ev,
            year = self.year,
            era = self.era,
            ele = electron,
            ele_veto = el_v,
            mu = muon,
            mu_veto = mu_v,
            jet_all = jet,
            jet_central = central,
            jet_btag = btag,
            jet_fwd = fwd,
            jet_light = light,
            met = met,
        )
        
        baseline = sel.dilep_baseline(cutflow=cutflow, DY=True)
        baseline_OS = sel.dilep_baseline(cutflow=cutflow, DY=True)
        
        # this defines all the dedicated selections for charge flip, nonprompt, conversions
        if not re.search(data_pattern, dataset):

            BL = (baseline & ((ak.num(el_t_p_cc)+ak.num(mu_t_p))==2))  # this is the MC baseline for events with two tight prompt leptons
            BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2)) # this is the MC baseline for events with two tight leptons

            np_est_sel_mc = (baseline & \
                #((((ak.num(el_t_p_cc)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p_cc)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
                ((((ak.num(el_t_p)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # FIXME check if not requiring electron charge consistency actually makes a difference
            np_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
            np_est_sel_data = (baseline & ~baseline)  # this has to be false

            cf_est_sel_mc = (baseline_OS & ((ak.num(el_t_p)+ak.num(mu_t_p))==2))
            cf_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_p_cf))>=1) )  # two tight leptons, at least one electron charge flip
            cf_est_sel_data = (baseline & ~baseline)  # this has to be false

            if dataset == 'top':
                conv_sel = BL  # anything that has tight, prompt, charge-consistent, non-external-conv, same-sign dileptons has to be internal conversion.
            elif dataset == 'XG':
                conv_sel = BL_incl & (((ak.num(el_t_conv)+ak.num(mu_t_conv))>0))
            else:
                conv_sel = (baseline & ~baseline)  # this has to be false


            data_sel = (baseline & ~baseline)  # this has to be false

            weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')
            weight_cf_mc = self.chargeflipWeight.flip_weight(el_t_p)

        else:
            BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2))

            BL_incl = BL

            np_est_sel_mc = (baseline & ~baseline)
            np_obs_sel_mc = (baseline & ~baseline)
            np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)==1) & (ak.num(el_f)+ak.num(mu_f)==1) )

            cf_est_sel_mc = (baseline & ~baseline)
            cf_obs_sel_mc = (baseline & ~baseline)
            cf_est_sel_data = (baseline_OS & ((ak.num(el_t)+ak.num(mu_t))==2) )
            conv_sel = (baseline & ~baseline)  # this has to be false

            weight_np_mc = np.zeros(len(ev))
            weight_cf_mc = np.zeros(len(ev))

            #rle = ak.to_numpy(ak.zip([ev.run, ev.luminosityBlock, ev.event]))
            run_ = ak.to_numpy(ev.run)
            lumi_ = ak.to_numpy(ev.luminosityBlock)
            event_ = ak.to_numpy(ev.event)

            data_sel = (baseline | ~baseline)  # this is always true

            if False:
                output['%s_run'%dataset] += processor.column_accumulator(run_[BL])
                output['%s_lumi'%dataset] += processor.column_accumulator(lumi_[BL])
                output['%s_event'%dataset] += processor.column_accumulator(event_[BL])




        weight_BL = weight.weight(modifier=shift)[BL]  # this is just a shortened weight list for the two prompt selection
        weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data')
        weight_cf_data = self.chargeflipWeight.flip_weight(el_t)

        out_sel = (BL | np_est_sel_mc | cf_est_sel_mc)

        dummy = (np.ones(len(ev))==1)
        def fill_multiple_np(hist, arrays, add_sel=dummy):
            #reg_sel = [BL, np_est_sel_mc, np_obs_sel_mc, np_est_sel_data, cf_est_sel_mc, cf_obs_sel_mc, cf_est_sel_data],
            #print ('len', len(reg_sel[0]))
            #print ('sel', reg_sel[0])
            reg_sel = [
                BL&add_sel,
                BL_incl&add_sel,
                np_est_sel_mc&add_sel,
                np_obs_sel_mc&add_sel,
                np_est_sel_data&add_sel,
                cf_est_sel_mc&add_sel,
                cf_obs_sel_mc&add_sel,
                cf_est_sel_data&add_sel,
                conv_sel&add_sel,
            ],
            fill_multiple(
                hist,
                datasets=[
                    dataset, # only prompt contribution from process
                    dataset+"_incl", # everything from process (inclusive MC truth)
                    "np_est_mc", # MC based NP estimate
                    "np_obs_mc", # MC based NP observation
                    "np_est_data",
                    "cf_est_mc",
                    "cf_obs_mc",
                    "cf_est_data",
                    "conv_mc",
                ],
                arrays=arrays,
                selections=reg_sel[0],  # no idea where the additional dimension is coming from...
                weights=[
                    weight.weight(modifier=shift)[reg_sel[0][0]],
                    weight.weight(modifier=shift)[reg_sel[0][1]],
                    weight.weight(modifier=shift)[reg_sel[0][2]]*weight_np_mc[reg_sel[0][2]],
                    weight.weight(modifier=shift)[reg_sel[0][3]],
                    weight.weight(modifier=shift)[reg_sel[0][4]]*weight_np_data[reg_sel[0][4]],
                    weight.weight(modifier=shift)[reg_sel[0][5]]*weight_cf_mc[reg_sel[0][5]],
                    weight.weight(modifier=shift)[reg_sel[0][6]],
                    weight.weight(modifier=shift)[reg_sel[0][7]]*weight_cf_data[reg_sel[0][7]],
                    weight.weight(modifier=shift)[reg_sel[0][8]],
                ],
            )


        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight_BL)
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
        fill_multiple_np(output['N_jet'],     {'multiplicity': ak.num(jet)})
        fill_multiple_np(output['N_b'],       {'multiplicity': ak.num(btag)})
        fill_multiple_np(output['N_central'], {'multiplicity': ak.num(central)})
        fill_multiple_np(output['N_ele'],     {'multiplicity':ak.num(electron)})
        fill_multiple_np(output['N_mu'],      {'multiplicity':ak.num(muon)})
        fill_multiple_np(output['N_fwd'],     {'multiplicity':ak.num(fwd)})
        fill_multiple_np(output['ST'],        {'ht': st})
        fill_multiple_np(output['HT'],        {'ht': ht})

        if not re.search(data_pattern, dataset):
            output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight_BL)
            output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight_BL)
            output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight_BL)
            output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight_BL)
            output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight_BL)
            output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight_BL)
            output['chargeFlip_vs_nonprompt'].fill(dataset=dataset, n1=n_chargeflip[BL], n2=n_nonprompt[BL], n_ele=ak.num(electron)[BL], weight=weight_BL)

            output['lead_gen_lep'].fill(
                dataset = dataset,
                pt  = ak.to_numpy(ak.flatten(leading_gen_lep[BL].pt)),
                eta = ak.to_numpy(ak.flatten(leading_gen_lep[BL].eta)),
                phi = ak.to_numpy(ak.flatten(leading_gen_lep[BL].phi)),
                weight = weight_BL
            )

            output['trail_gen_lep'].fill(
                dataset = dataset,
                pt  = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].pt)),
                eta = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].eta)),
                phi = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].phi)),
                weight = weight_BL
            )


        fill_multiple_np(output['MET'], {'pt':met.pt, 'phi':met.phi})
        fill_multiple_np(output['LT'], {'ht':lt})
        
        fill_multiple_np(
            output['lead_lep'],
            {
                'pt':  pad_and_flatten(leading_lepton.p4.pt),
                'eta': pad_and_flatten(leading_lepton.eta),
                'phi': pad_and_flatten(leading_lepton.phi),
            },
        )

        fill_multiple_np(
            output['trail_lep'],
            {
                'pt':  pad_and_flatten(trailing_lepton.p4.pt),
                'eta': pad_and_flatten(trailing_lepton.eta),
                'phi': pad_and_flatten(trailing_lepton.phi),
            },
        )

        fill_multiple_np(
            output['fwd_jet'],
            {
                'pt':  pad_and_flatten(best_fwd.pt),
                'eta': pad_and_flatten(best_fwd.eta),
                'phi': pad_and_flatten(best_fwd.phi),
            },
        )
        
        #output['fwd_jet'].fill(
        #    dataset = dataset,
        #    pt  = ak.flatten(j_fwd[BL].pt),
        #    eta = ak.flatten(j_fwd[BL].eta),
        #    phi = ak.flatten(j_fwd[BL].phi),
        #    weight = weight_BL
        #)
            
        #output['high_p_fwd_p'].fill(dataset=dataset, p = ak.flatten(best_fwd[BL].p), weight = weight_BL)
        output['dilepton_mass'].fill(dataset=dataset, mass = ak.flatten(dilepton_mass[BL]), weight = weight_BL)
            
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
            eta = ak.flatten(jet.eta[:, 0:1][BL]),
            phi = ak.flatten(jet.phi[:, 0:1][BL]),
            weight = weight_BL
        )
        
        output['j2'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 1:2][BL].pt_nom),
            eta = ak.flatten(jet[:, 1:2][BL].eta),
            phi = ak.flatten(jet[:, 1:2][BL].phi),
            weight = weight_BL
        )
        
        #output['j3'].fill(
        #    dataset = dataset,
        #    pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
        #    eta = ak.flatten(jet[:, 2:3][BL].eta),
        #    phi = ak.flatten(jet[:, 2:3][BL].phi),
        #    weight = weight_BL
        #)
            
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import get_babies
    from processor.default_accumulators import *

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--keep', action='store_true', default=None, help="Keep/use existing results??")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on a DASK cluster?")
    argParser.add_argument('--profile', action='store_true', default=None, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=None, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--verysmall', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = not args.keep
    small       = args.small
    verysmall   = args.verysmall
    if verysmall:
        small = True
    
    year        = int(args.year[0:4])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'DY_analysis_%s%s'%(year,era)
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    in_path = '/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.4.0_dilep/'

    fileset_all = get_babies(in_path, year='UL%s%s'%(year,era))
    #fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/', year=2018)
    
    fileset = {
        'topW_v3': fileset_all['topW_NLO'],
        ##'topW_v3': fileset_all['topW_v3'],
        ###'topW_EFT_mix': fileset_all['topW_EFT'],
        ##'topW_EFT_cp8': fileset_all['topW_EFT_cp8'],
        ##'topW_EFT_mix': fileset_all['topW_EFT_mix'],
        'TTW': fileset_all['TTW'],
        'TTZ': fileset_all['TTZ'],
        'TTH': fileset_all['TTH'],
        'diboson': fileset_all['diboson'],
        'rare': fileset_all['TTTT']+fileset_all['triboson'],
        #'ttbar': fileset_all['ttbar1l'],
        'DY': fileset_all['DY'],
        'ttbar': fileset_all['top'],
        'XG': fileset_all['XG'],
        'MuonEG': fileset_all['MuonEG'],
        'DoubleMuon': fileset_all['DoubleMuon'],
        'EGamma': fileset_all['EGamma'],
        'DoubleEG': fileset_all['DoubleEG'],
        'SingleElectron': fileset_all['SingleElectron'],
        'SingleMuon': fileset_all['SingleMuon'],
        ####'topW_full_EFT': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_UL17_v7/*.root'),
        ####'topW_NLO': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_SMEFTatNLO_weight_RunIIAutumn18_NANO_UL17_v7/*.root'),
    }
    
    fileset = make_small(fileset, small, n_max=10)

    if verysmall:
        fileset = {'topW_v3': fileset['topW_v3'], 'MuonEG': fileset['MuonEG'], 'ttbar': fileset['ttbar']}

    add_processes_to_output(fileset, desired_output)

    variations = [
        {'name': 'central',     'ext': '',                  'weight': None,   'pt_var': 'pt_nom'},
        {'name': 'jes_up',      'ext': '_pt_jesTotalUp',    'weight': None,   'pt_var': 'pt_jesTotalUp'},
        {'name': 'jes_down',    'ext': '_pt_jesTotalDown',  'weight': None,   'pt_var': 'pt_jesTotalDown'},
        {'name': 'PU_up',       'ext': '_PUUp',             'weight': 'PUUp', 'pt_var': 'pt_nom'},
        {'name': 'PU_down',     'ext': '_PUDown',           'weight': 'PUDown', 'pt_var': 'pt_nom'},
    ] # FIXME this was changed to avoid the double loops. Fix also in the processor FIXME

    if args.dump:
        variables = [
            'n_jet',
            'n_b',
            'n_fwd',
            'n_tau',
            #'n_track',
            'st',
            'met',
            'mjj_max',
            'delta_eta_jj',
            'lead_lep_pt',
            'lead_lep_eta',
            'sublead_lep_pt',
            'sublead_lep_eta',
            'dilepton_mass',
            'dilepton_pt',
            'fwd_jet_pt',
            'fwd_jet_p',
            'fwd_jet_eta',
            'lead_jet_pt',
            'sublead_jet_pt',
            'lead_jet_eta',
            'sublead_jet_eta',
            'lead_btag_pt',
            'sublead_btag_pt',
            'lead_btag_eta',
            'sublead_btag_eta',
            'min_bl_dR',
            'min_mt_lep_met',
            'weight',
            'weight_np',
            'weight_cf',
            'SS',
            'OS',
            'AR',
            'LL',
            'conv',
            'label',
            'total_charge',
        ]

        for var in variables:
            desired_output.update({var: processor.column_accumulator(np.zeros(shape=(0,)))})

    if local:# and not profile:
        exe_args = {
            'workers': 24,
            'function_args': {'flatten': False},
            "schema": NanoAODSchema,
        }
        exe = processor.futures_executor

    elif iterative:
        exe_args = {
            'function_args': {'flatten': False},
            "schema": NanoAODSchema,
        }
        exe = processor.iterative_executor

    else:
        from Tools.helpers import get_scheduler_address
        from dask.distributed import Client, progress

        scheduler_address = get_scheduler_address()
        c = Client(scheduler_address)

        exe_args = {
            'client': c,
            'function_args': {'flatten': False},
            "schema": NanoAODSchema,
            "tailtimeout": 300,
            "retries": 3,
            "skipbadfiles": True
        }
        exe = processor.dask_executor

    # add some histograms that we defined in the processor
    # everything else is taken the default_accumulators.py
    from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, mass_axis
    desired_output.update({
        "ST": hist.Hist("Counts", dataset_axis, ht_axis),
        "HT": hist.Hist("Counts", dataset_axis, ht_axis),
        "LT": hist.Hist("Counts", dataset_axis, ht_axis),
        "dilepton_mass": hist.Hist("Counts", dataset_axis, mass_axis),
        "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, pt_axis),
        "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, pt_axis),
        "LT_SR_pp": hist.Hist("Counts", dataset_axis, ht_axis),
        "LT_SR_mm": hist.Hist("Counts", dataset_axis, ht_axis),
        "node": hist.Hist("Counts", dataset_axis, multiplicity_axis),
        "node0_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node4_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node4_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score_pp": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score_mm": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score_transform_pp": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score_transform_mm": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score_pp": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score_mm": hist.Hist("Counts", dataset_axis, score_axis),
    })



    for rle in ['run', 'lumi', 'event']:
        desired_output.update({
                'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
        })

    histograms = sorted(list(desired_output.keys()))
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            DY_analysis(year=year, variations=variations, accumulator=desired_output, evaluate=args.evaluate, training=args.training, dump=args.dump, era=era),
            exe,
            exe_args,
            chunksize=250000,  # I guess that's already running into the max events/file
        )
        
        if save:
            cache['fileset']        = fileset
            cache['cfg']            = cfg
            cache['histograms']     = histograms
            cache['simple_output']  = output
            cache.dump()

    ## output for DNN training
    if args.dump:
        if overwrite:
            df_dict = {}
            for var in variables:
                df_dict.update({var: output[var].value})

            df_out = pd.DataFrame( df_dict )
            if not args.small:
                df_out.to_hdf('multiclass_input_%s_v2.h5'%args.year, key='df', format='table', mode='w')
        else:
            print ("Loading DF")
            df_out = pd.read_hdf('multiclass_input_%s_v2.h5'%args.year)

    print ("\nNN debugging:")
    print (output['node'].sum('multiplicity').values())


    ## Data double counting checks
    if args.check_double_counting:
        em = zip_run_lumi_event(output, 'MuonEG')
        e  = zip_run_lumi_event(output, 'EGamma')
        mm = zip_run_lumi_event(output, 'DoubleMuon')

        print ("Total events from MuonEG:", len(em))
        print ("Total events from EGamma:", len(e))
        print ("Total events from DoubleMuon:", len(mm))

        em_mm = np.intersect1d(em, mm)
        print ("Overlap MuonEG/DoubleMuon:", len(em_mm))

        e_mm = np.intersect1d(e, mm)
        print ("Overlap EGamma/DoubleMuon:", len(e_mm))

        em_e = np.intersect1d(em, e)
        print ("Overlap MuonEG/EGamma:", len(em_e))

    
    ## some plots
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    from plots.helpers import makePlot
    
    # defining some new axes for rebinning.
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
    
    my_labels = {
        'topW_v3': 'top-W scat.',
        'topW_EFT_cp8': 'EFT, cp8',
        'topW_EFT_mix': 'EFT mix',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'rare': 'rare',
        'ttbar': r'$t\bar{t}$',
        'np_obs_mc': 'nonprompt (MC true)',
        'np_est_mc': 'nonprompt (MC est)',
        'cf_obs_mc': 'charge flip (MC true)',
        'cf_est_mc': 'charge flip (MC est)',
        'np_est_data': 'nonprompt (est)',
        'cf_est_data': 'charge flip (est)',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'rare': '#EE82EE',
        'ttbar': '#1982C4',
        'np_obs_mc': '#1982C4',
        'np_est_mc': '#1982C4',
        'np_est_data': '#1982C4',
        'cf_obs_mc': '#0F7173',
        'cf_est_mc': '#0F7173',
        'cf_est_data': '#0F7173',
    }

    makePlot(output, 'node', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['rare', 'diboson', 'TTW', 'TTH', 'TTZ', 'np_est_data', 'cf_est_data', 'topW_v3'],
         #order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         #signals=['topW_v3'],
         omit=['ttbar', 'cf_est_mc', 'cf_obs_mc', 'np_est_mc', 'np_obs_mc',],
         save=os.path.expandvars('$TWHOME/dump/ML_node'),
        )

    makePlot(output, 'node', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['rare', 'diboson', 'TTW', 'TTH', 'TTZ', 'np_obs_mc', 'cf_obs_mc', 'topW_v3'],
         #order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         #signals=['topW_v3'],
         omit=['ttbar', 'cf_est_mc', 'cf_est_data', 'np_est_mc', 'np_est_data',],
         save=os.path.expandvars('$TWHOME/dump/ML_node_MC'),
        )

    # This is a partial unblinding
    makePlot(output, 'node0_score_incl', 'score',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'np_est_data', 'cf_est_data'],
         signals=['topW_v3'],
         omit=['ttbar', 'rare', 'cf_est_mc', 'cf_obs_mc', 'np_est_mc', 'np_obs_mc',],
         save=os.path.expandvars('$TWHOME/dump/ML_node0_score_incl'),
        )

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score', shape=True, ymax=0.35,
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'ttbar', 'TTH', 'TTZ', 'cf_est_data', 'cf_est_mc', 'cf_obs_mc', 'np_est_data', 'np_est_mc', 'np_obs_mc', 'rare'],
         save=os.path.expandvars('$TWHOME/dump/ML_node0_score_shape'),
        )

