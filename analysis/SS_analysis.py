'''
Central analysis script for the SS channel.
All histograms and numbers are produced here,
starting from skimmed NanoAOD samples.
We deal with 4 eras:
- 2016APV = preVFP = HIPM, B-F
- 2016 = postVFP, F-H
- 2017
- 2018
'''


import os
import re
import datetime
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist, util
from coffea.processor import accumulate
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd
import pickle

from analysis.Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, prompt_no_conv, nonprompt_no_conv, external_conversion, fast_match
from analysis.Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from analysis.Tools.cutflow import Cutflow
from analysis.Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM, get_samples
from analysis.Tools.config_helpers import loadConfig, make_small, data_pattern, get_latest_output, load_yaml, data_path
from analysis.Tools.triggers import getFilters, getTriggers
from analysis.Tools.btag_scalefactors import btag_scalefactor
from analysis.Tools.tau_scalefactors import tau_scalefactor
from analysis.Tools.trigger_scalefactors import triggerSF
from analysis.Tools.ttH_lepton_scalefactors import LeptonSF
from analysis.Tools.selections import Selection, get_pt
from analysis.Tools.nonprompt_weight import NonpromptWeight
from analysis.Tools.chargeFlip import charge_flip
from analysis.Tools.pileup import pileup

import warnings
warnings.filterwarnings("ignore")

from analysis.Tools.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer
from analysis.BIT.BoostedInformationTreeP3 import BoostedInformationTree
from analysis.boosted_information_tree import get_bit_score

from sklearn.preprocessing import QuantileTransformer


class SS_analysis(processor.ProcessorABC):
    def __init__(self,
                 year=2016,
                 era=None,
                 variations=[],
                 accumulator={},
                 evaluate=False,
                 bit=True,
                 training='v8',
                 dump=False,
                 hyperpoly=None,  # maybe can go?
                 points=[[]],  # maybe can go?
                 weights=[],
                 reweight=1,
                 minimal=False,
                 fixed_template=False,
                 toys=0,
                 ):
        self.variations = variations

        self.year = year
        self.era = era  # this is here for 2016 APV
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        self.fixed_template = fixed_template

        self.bit = bit
        self.minimal=minimal

        if self.minimal:
            print ("Will be running minimal")
        
        self.btagSF = btag_scalefactor(year, era=era)
        self.leptonSF = LeptonSF(year=year, era=era)
        self.tauSF = tau_scalefactor(year=year, era=era)
        self.triggerSF = triggerSF(year=year)
        self.pu = pileup(year=year, UL=True, era=era)

        self.nonpromptWeight = NonpromptWeight(year=year)  # NOTE no era split. Does not need it!
        self.chargeflipWeight = charge_flip(year=year, era=era)

        self.hyperpoly = hyperpoly
        self.points = points
        self.weights = weights

        self.reweight = reweight
        
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        presel = ak.num(events.Jet)>2

        ev = events[presel]
        dataset = ev.metadata['dataset']
        
        # don't distinguish between data and MC anymore
        variations = self.variations

        return processor.accumulate(self.process_shift(ev, var) for var in variations)

    def process_shift(self, ev, var=None):
        # use a very loose preselection to filter the events
        output = self.accumulator.identity()

        #output['totalEvents']['all'] += len(events)
        output['skimmedEvents']['all'] += len(ev)
        dataset = ev.metadata['dataset']
        
        if not re.search(data_pattern, dataset):
            ## Generated leptons
            gen_lep = ev.GenL
            leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
            trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ####################
        ### Reco objects ###
        ####################


        # NOTE: lepton objects are unaffected by any of the variations, therefore we keep them outside the variation loop.

        # Get the leptons. This has changed a couple of times now, but we are using fakeable objects as baseline leptons.
        # The added p4 instance has the corrected pt (conePt for fakeable) and should be used for any following selection or calculation
        # Any additional correction (if we choose to do so) should be added here, e.g. Rochester corrections, ...
        ## Muons
        mu_v     = Collections(ev, "Muon", "vetoTTH", year=self.year, era=self.era).get()  # these include all muons, tight and fakeable
        mu_t     = Collections(ev, "Muon", "tightSSTTH", year=self.year, era=self.era).get()
        mu_f     = Collections(ev, "Muon", "fakeableSSTTH", year=self.year, era=self.era).get()
        muon     = ak.concatenate([mu_t, mu_f], axis=1)
        muon['p4'] = get_four_vec_fromPtEtaPhiM(muon, get_pt(muon), muon.eta, muon.phi, muon.mass, copy=False)
        
        ## Electrons
        el_v        = Collections(ev, "Electron", "vetoTTH", year=self.year, era=self.era).get()
        el_t        = Collections(ev, "Electron", "tightSSTTH", year=self.year, era=self.era).get()
        el_f        = Collections(ev, "Electron", "fakeableSSTTH", year=self.year, era=self.era).get()
        electron    = ak.concatenate([el_t, el_f], axis=1)
        electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False)


        if not re.search(data_pattern, dataset):
            gen_photon = ev.GenPart[ev.GenPart.pdgId==22]

            el_t_p  = prompt_no_conv(el_t, gen_photon)
            el_t_np = nonprompt_no_conv(el_t, gen_photon)
            el_t_conv = external_conversion(el_t, gen_photon)
            el_f_p  = prompt(el_f)
            el_f_np = nonprompt_no_conv(el_f, gen_photon)

            mu_t_p  = prompt_no_conv(mu_t, gen_photon)
            mu_t_np = nonprompt_no_conv(mu_t, gen_photon)
            mu_t_conv = external_conversion(mu_t, gen_photon)
            mu_f_p  = prompt(mu_f)
            mu_f_np = nonprompt_no_conv(mu_f, gen_photon)

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

        # this is where the real JEC dependent stuff happens

        pt_var   = var['pt_var']
        var_name = var['name']
        shift    = var['weight']

        met = getMET(ev, pt_var=pt_var)

        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var=pt_var)
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

        central   = jet[(abs(jet.eta)<2.4)]
        btag      = getBTagsDeepFlavB(jet, era=self.era, year=self.year)
        light     = getBTagsDeepFlavB(jet, era=self.era, year=self.year, invert=True)
        light_central = light[(abs(light.eta)<2.5)]
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
            weight.add("weight", ev.genWeight)

            if isinstance(self.reweight[dataset], int) or isinstance(self.reweight[dataset], float):
                pass  # NOTE: this can be implemented later
                #if self.reweight != 1:
                #    weight.add("reweight", self.reweight[dataset])
            else:
                print(f"A reweight is applied to dataset {dataset}")
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

            if var['name'] == 'tau_up':
                weight.add("tau", self.tauSF.get(tau, var='up', WP='Loose'))
            elif var['name'] == 'tau_down':
                weight.add("tau", self.tauSF.get(tau, var='down', WP='Loose'))
            else:
                weight.add("tau", self.tauSF.get(tau, var='nom', WP='Loose'))

            # trigger SFs
            weight.add("trigger", self.triggerSF.get(electron, muon))

        if dataset.count('EFT'):
            #print (self.points)
            # FIXME legacy code ready to be retired
            for point in self.points:
                point['weight'] = Weights( len(ev) )
                point['weight'].add("EFT", self.hyperpoly.eval(ev.Pol, point['point']))

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

        if var['name'] == 'central':
            # Only fill the cutflow onece :)
            cutflow     = Cutflow(output, ev, weight=weight)
            baseline = sel.dilep_baseline(cutflow=cutflow, SS=True, omit=['N_fwd>0'])
        else:
            baseline = sel.dilep_baseline(cutflow=None, SS=True, omit=['N_fwd>0'])
        baseline_OS = sel.dilep_baseline(cutflow=None, SS=False, omit=['N_fwd>0'])  # this is for charge flip estimation

        # this defines all the dedicated selections for charge flip, nonprompt, conversions
        if not re.search(data_pattern, dataset):

            BL = (baseline & ((ak.num(el_t_p_cc)+ak.num(mu_t_p))==2))  # this is the MC baseline for events with two tight prompt leptons
            BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2)) # this is the MC baseline for events with two tight leptons

            np_est_sel_mc = (baseline & \
                #((((ak.num(el_t_p_cc)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p_cc)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
                ((((ak.num(el_t_p)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # FIXME check if not requiring electron charge consistency actually makes a difference
            np_sub_sel_mc = (baseline & \
                ((((ak.num(el_t_p)+ak.num(mu_t_p))==1) & ((ak.num(el_f_p)+ak.num(mu_f_p))==1)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_p)+ak.num(mu_f_p))==2)) ))
            np_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
            np_est_sel_data = (baseline & ~baseline)  # this has to be false

            cf_est_sel_mc = (baseline_OS & ((ak.num(el_t_p)+ak.num(mu_t_p))==2))
            cf_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_p_cf))>=1) )  # two tight leptons, at least one electron charge flip
            cf_est_sel_data = (baseline & ~baseline)  # this has to be false

            if dataset.count("TTTo") or dataset.count("DY"):
                conv_sel = BL  # anything that has tight, prompt, charge-consistent, non-external-conv, same-sign dileptons has to be internal conversion.
            elif dataset.count("Gamma") or dataset.count("WGTo") or dataset.count("ZGTo") or dataset.count("WZG_"):
                conv_sel = BL_incl & (((ak.num(el_t_conv)+ak.num(mu_t_conv))>0))
            else:
                conv_sel = (baseline & ~baseline)  # this has to be false


            data_sel = (baseline & ~baseline)  # this has to be false

            #if var["name"].count("fake")
            weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT', variation=(var["ext"] if var["name"].count("fake") else ''))
            #weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT', variation=(var["ext"] if var["name"].count("fake") else ''))  # define one for subtraction??
            weight_np_mc_qcd = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='QCD', variation=(var["ext"] if var["name"].count("fake") else ''))
            weight_cf_mc = self.chargeflipWeight.get(el_t_p)

        else:
            BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2))

            BL_incl = BL

            np_est_sel_mc = (baseline & ~baseline)
            np_sub_sel_mc = (baseline & ~baseline)
            np_obs_sel_mc = (baseline & ~baseline)
            np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)==1) & (ak.num(el_f)+ak.num(mu_f)==1) )

            cf_est_sel_mc = (baseline & ~baseline)
            cf_obs_sel_mc = (baseline & ~baseline)
            cf_est_sel_data = (baseline_OS & ((ak.num(el_t)+ak.num(mu_t))==2) )
            conv_sel = (baseline & ~baseline)  # this has to be false

            weight_np_mc = np.zeros(len(ev))
            weight_np_mc_qcd = np.zeros(len(ev))
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

        out_sel = (BL | np_est_sel_mc | cf_est_sel_mc)

        if self.bit:
            ## Evaluate the BIT ##
            BIT_inputs_d = {
                'n_jet':            ak.to_numpy(ak.num(jet)),
                'n_fwd':            ak.to_numpy(ak.num(fwd)),
                'n_b':              ak.to_numpy(ak.num(btag)),
                'n_tau':            ak.to_numpy(ak.num(tau)),
                'st':               ak.to_numpy(st),
                'lt':               ak.to_numpy(lt),
                'met':              ak.to_numpy(met.pt),
                'mjj_max':          ak.to_numpy(ak.fill_none(ak.max(mjf, axis=1),0)),
                'delta_eta_jj':     ak.to_numpy(pad_and_flatten(delta_eta)),
                'lead_lep_pt':      ak.to_numpy(pad_and_flatten(leading_lepton.p4.pt)),
                'lead_lep_eta':     ak.to_numpy(pad_and_flatten(leading_lepton.p4.eta)),
                'sublead_lep_pt':   ak.to_numpy(pad_and_flatten(trailing_lepton.p4.pt)),
                'sublead_lep_eta':  ak.to_numpy(pad_and_flatten(trailing_lepton.p4.eta)),
                'dilepton_mass':    ak.to_numpy(pad_and_flatten(dilepton_mass)),
                'dilepton_pt':      ak.to_numpy(pad_and_flatten(dilepton_pt)),
                'fwd_jet_pt':       ak.to_numpy(pad_and_flatten(best_fwd.p4.pt)),
                'fwd_jet_p':        ak.to_numpy(pad_and_flatten(best_fwd.p4.p)),
                'fwd_jet_eta':      ak.to_numpy(pad_and_flatten(best_fwd.p4.eta)),
                'lead_jet_pt':      ak.to_numpy(pad_and_flatten(jet[:, 0:1].p4.pt)),
                'sublead_jet_pt':   ak.to_numpy(pad_and_flatten(jet[:, 1:2].p4.pt)),
                'lead_jet_eta':     ak.to_numpy(pad_and_flatten(jet[:, 0:1].p4.eta)),
                'sublead_jet_eta':  ak.to_numpy(pad_and_flatten(jet[:, 1:2].p4.eta)),
                'lead_btag_pt':     ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1].p4.pt)),
                'sublead_btag_pt':  ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2].p4.pt)),
                'lead_btag_eta':    ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1].p4.eta)),
                'sublead_btag_eta': ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2].p4.eta)),
                'min_bl_dR':        ak.to_numpy(ak.fill_none(min_bl_dR, 0)),
                'min_mt_lep_met':   ak.to_numpy(ak.fill_none(min_mt_lep_met, 0)),
            }

            #print (sorted(BIT_inputs_d.keys()))  # NOTE: the order is already correct
            #BIT_inputs = np.stack( [BIT_inputs_d[k] for k in sorted(BIT_inputs_d.keys())] )
            #BIT_inputs = np.nan_to_num(BIT_inputs, 0, posinf=1e5, neginf=-1e5)  # events with posinf/neginf/nan will not pass the BL selection anyway
            #BIT_inputs = np.moveaxis(BIT_inputs, 0, 1)
            #
            variables = sorted(BIT_inputs_d.keys())
            BIT_inputs_df = pd.DataFrame(BIT_inputs_d)
            BIT_inputs = BIT_inputs_df[variables].values

            bit_file = 'analysis/Tools/data/networks/bits_v40.pkl'  # was v31
            with open(bit_file, 'rb') as f:
                bits = pickle.load(f)

            bit_pred = {}
            for i in range(6):
                bit_pred["pred_%s"%i] = bits[i].vectorized_predict(BIT_inputs)

            bit_pred = pd.DataFrame(bit_pred)

        weight_BL = weight.weight(modifier=shift)[BL]  # this is just a shortened weight list for the two prompt selection
        weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data', variation=(var["ext"] if var["name"].count("fake") else ''))
        weight_cf_data = self.chargeflipWeight.get(el_t)

        dummy = (np.ones(len(ev))==1)
        dummy_weight = Weights(len(ev))
        def fill_multiple_np(hist, arrays, add_sel=dummy, other=None, weight_multiplier=dummy_weight.weight()):

            if not re.search(data_pattern, dataset) and self.minimal:
                # NOTE:
                # For minimal we don't need all the different predictions for MC
                reg_sel = [
                    BL&add_sel,
                    conv_sel&add_sel,
                    np_sub_sel_mc&add_sel,
                ]
                fill_multiple(
                    hist,
                    dataset = dataset,
                    predictions=[
                        "central", # only prompt contribution from process
                        "conv_mc",
                        "np_sub_mc",  # NOTE prompt subtraction - WIP
                    ],
                    arrays=arrays,
                    selections=reg_sel,
                    weights=[
                        weight_multiplier[reg_sel[0]]*weight.weight(modifier=shift)[reg_sel[0]],
                        weight_multiplier[reg_sel[1]]*weight.weight(modifier=shift)[reg_sel[1]],
                        weight_multiplier[reg_sel[2]]*weight.weight(modifier=shift)[reg_sel[2]]*weight_np_data[reg_sel[2]],
                    ],
                    systematic = var_name,
                    other = other,
                    )

            elif re.search(data_pattern, dataset) and self.minimal:
                reg_sel = [
                    BL&add_sel,
                    np_est_sel_data&add_sel,
                    cf_est_sel_data&add_sel,
                ]
                fill_multiple(
                    hist,
                    dataset = dataset,
                    predictions=[
                        "central", # only prompt contribution from process
                        "np_est_data",
                        "cf_est_data",
                    ],
                    arrays=arrays,
                    selections=reg_sel,
                    weights=[
                        weight_multiplier[reg_sel[0]]*weight.weight(modifier=shift)[reg_sel[0]],
                        weight_multiplier[reg_sel[1]]*weight.weight(modifier=shift)[reg_sel[1]]*weight_np_data[reg_sel[1]],
                        weight_multiplier[reg_sel[2]]*weight.weight(modifier=shift)[reg_sel[2]]*weight_cf_data[reg_sel[2]],
                    ],
                    systematic = var_name,
                    other = other,
                )

            else:
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
                    np_est_sel_mc&add_sel,  # MC based NP estimate with QCD FR
                    np_sub_sel_mc&add_sel,  # prompt subtraction
                ]
                fill_multiple(
                    hist,
                    dataset = dataset,
                    predictions=[
                        "central", # only prompt contribution from process
                        "inclusive", # everything from process (inclusive MC truth)
                        "np_est_mc", # MC based NP estimate
                        "np_obs_mc", # MC based NP observation
                        "np_est_data",
                        "cf_est_mc",
                        "cf_obs_mc",
                        "cf_est_data",
                        "conv_mc",
                        "np_est_mc_qcd",  # MC based NP estimate with QCD FR
                        "np_sub_mc",  # NOTE prompt subtraction - WIP
                    ],
                    arrays=arrays,
                    selections=reg_sel,
                    weights=[
                        weight_multiplier[reg_sel[0]]*weight.weight(modifier=shift)[reg_sel[0]],
                        weight_multiplier[reg_sel[1]]*weight.weight(modifier=shift)[reg_sel[1]],
                        weight_multiplier[reg_sel[2]]*weight.weight(modifier=shift)[reg_sel[2]]*weight_np_mc[reg_sel[2]],
                        weight_multiplier[reg_sel[3]]*weight.weight(modifier=shift)[reg_sel[3]],
                        weight_multiplier[reg_sel[4]]*weight.weight(modifier=shift)[reg_sel[4]]*weight_np_data[reg_sel[4]],
                        weight_multiplier[reg_sel[5]]*weight.weight(modifier=shift)[reg_sel[5]]*weight_cf_mc[reg_sel[5]],
                        weight_multiplier[reg_sel[6]]*weight.weight(modifier=shift)[reg_sel[6]],
                        weight_multiplier[reg_sel[7]]*weight.weight(modifier=shift)[reg_sel[7]]*weight_cf_data[reg_sel[7]],
                        weight_multiplier[reg_sel[8]]*weight.weight(modifier=shift)[reg_sel[8]],
                        weight_multiplier[reg_sel[9]]*weight.weight(modifier=shift)[reg_sel[9]]*weight_np_mc_qcd[reg_sel[9]],
                        weight_multiplier[reg_sel[10]]*weight.weight(modifier=shift)[reg_sel[10]]*weight_np_data[reg_sel[10]],
                    ],
                    systematic = var_name,
                    other = other,
                )

        if self.bit:

            for p in self.points:
                x,y = p['point']
                point = p['point']
                if self.fixed_template:
                    qt = load_transformer(f'v40_cpt_5_cpqm_5')  # was v31
                    score_trans = get_bit_score(bit_pred, cpt=5, cpqm=5, trans=qt)
                else:
                    qt = load_transformer(f'v40_cpt_{x}_cpqm_{y}')  # was v31
                    score_trans = get_bit_score(bit_pred, cpt=x, cpqm=y, trans=qt)

                # Get the weights
                if dataset.count('EFT'):
                    eft_weight = self.hyperpoly.eval(ev.Pol, point)
                else:
                    eft_weight = dummy_weight.weight()

                SR_sel_pp = ((ak.num(fwd)>0) & (ak.sum(lepton.charge, axis=1)>0))
                SR_sel_mm = ((ak.num(fwd)>0) & (ak.sum(lepton.charge, axis=1)<0))

                if dataset.count('EFT'):
                    fill_multiple_np(
                        output['bit_score_incl'],
                        {'bit': score_trans},
                        add_sel = (ak.num(fwd)>0),  # NOTE: this is to sync with the BIT development script
                        other={'EFT': f"bsm_cpt_{x}_cpqm_{y}"},
                        weight_multiplier = eft_weight,
                       )

                    fill_multiple_np(
                        output['bit_score_pp'],
                        {'bit': score_trans},
                        add_sel = SR_sel_pp,  # NOTE: this is to sync with the BIT development script
                        other={'EFT': f"bsm_cpt_{x}_cpqm_{y}"},
                        weight_multiplier = eft_weight,
                       )

                    fill_multiple_np(
                        output['bit_score_mm'],
                        {'bit': score_trans},
                        add_sel = SR_sel_mm,  # NOTE: this is to sync with the BIT development script
                        other={'EFT': f"bsm_cpt_{x}_cpqm_{y}"},
                        weight_multiplier = eft_weight,
                       )
                # ENDIF

                fill_multiple_np(
                    output['bit_score_incl'],
                    {'bit': score_trans},
                    add_sel = (ak.num(fwd)>0),  # NOTE: this is to sync with the BIT development script
                    other={'EFT': f"cpt_{x}_cpqm_{y}"},
                   )

                fill_multiple_np(
                    output['bit_score_pp'],
                    {'bit': score_trans},
                    add_sel = SR_sel_pp,  # NOTE: this is to sync with the BIT development script
                    other={'EFT': f"cpt_{x}_cpqm_{y}"},
                   )

                fill_multiple_np(
                    output['bit_score_mm'],
                    {'bit': score_trans},
                    add_sel = SR_sel_mm,  # NOTE: this is to sync with the BIT development script
                    other={'EFT': f"cpt_{x}_cpqm_{y}"},
                   )


                if not re.search(data_pattern, dataset) and var['name'] == 'central':

                    ## poor mans (?) toys only for central value
                    #weights_pp = (eft_weight*weight.weight())[(BL & SR_sel_pp)]
                    #weights_mm = (eft_weight*weight.weight())[(BL & SR_sel_mm)]
                    #for toy in range(toys):
                    #    pass
                    #    # FIXME on monday
                    #    # https://github.com/HephyAnalysisSW/TMB/blob/main/plots/plotsRobert/limits-toy/limit-VH-toy-2D.py#L134-L139
                    #print ("Running PDFs")
                    # if we just run central (len(variations)=1) we don't need the PDF variations either
                    for i in range(1,101):
                        pdf_ext = "pdf_%s"%i

                        output['bit_score_pp'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = f"cpt_{x}_cpqm_{y}",
                            bit         = score_trans[(BL & SR_sel_pp)],
                            weight      = weight.weight()[(BL & SR_sel_pp)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['bit_score_mm'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = f"cpt_{x}_cpqm_{y}",
                            bit         = score_trans[(BL & SR_sel_mm)],
                            weight      = weight.weight()[(BL & SR_sel_mm)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                    for i in ([0,1,3,5,7,8] if not (dataset.count('EFT') or dataset.count('ZZTo2Q2L_mllmin4p0')) else [0,1,3,4,6,7]):
                        # NOTE I have no idea why there are less weights in some samples. Confirmed correct indices.
                        # SAMPLES WITH JUST 8 SCALE WEIGHTS: EFT SIGNALS, ZZTo2Q2L_mllmin4p0
                        # LHE scale variation weights (w_var / w_nominal); [0] is MUF="0.5" MUR="0.5"; [1] is MUF="1.0" MUR="0.5"; [2] is MUF="2.0" MUR="0.5"; [3] is MUF="0.5" MUR="1.0"; [4] is MUF="1.0" MUR="1.0"; [5] is MUF="2.0" MUR="1.0"; [6] is MUF="0.5" MUR="2.0"; [7] is MUF="1.0" MUR="2.0"; [8] is MUF="2.0" MUR="2.0"
                        # LHE scale variation weights (w_var / w_nominal); [0] is MUF="0.5" MUR="0.5"; [1] is MUF="1.0" MUR="0.5"; [2] is MUF="2.0" MUR="0.5"; [3] is MUF="0.5" MUR="1.0"; [4] is MUF="2.0" MUR="1.0"; [5] is MUF="0.5" MUR="2.0"; [6] is MUF="1.0" MUR="2.0"; [7] is MUF="2.0" MUR="2.0"
                        pdf_ext = "scale_%s"%i

                        output['bit_score_pp'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = f"cpt_{x}_cpqm_{y}",
                            bit         = score_trans[(BL & SR_sel_pp)],
                            weight      = weight.weight()[(BL & SR_sel_pp)] * ev.LHEScaleWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['bit_score_mm'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = f"cpt_{x}_cpqm_{y}",
                            bit         = score_trans[(BL & SR_sel_mm)],
                            weight      = weight.weight()[(BL & SR_sel_mm)] * ev.LHEScaleWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                    if len(ev.PSWeight[0]) > 1:
                        for i in range(4):
                            pdf_ext = "PS_%s"%i

                            output['bit_score_pp'].fill(
                                dataset     = dataset,
                                systematic  = pdf_ext,
                                prediction  = 'central',
                                EFT         = f"cpt_{x}_cpqm_{y}",
                                bit         = score_trans[(BL & SR_sel_pp)],
                                weight      = weight.weight()[(BL & SR_sel_pp)] * ev.PSWeight[:,i][(BL & SR_sel_pp)],
                            )

                            output['bit_score_mm'].fill(
                                dataset     = dataset,
                                systematic  = pdf_ext,
                                prediction  = 'central',
                                EFT         = f"cpt_{x}_cpqm_{y}",
                                bit         = score_trans[(BL & SR_sel_mm)],
                                weight      = weight.weight()[(BL & SR_sel_mm)] * ev.PSWeight[:,i][(BL & SR_sel_mm)],
                            )

        if self.dump and var['name']=='central' and not self.minimal:
            #output['label']     += processor.column_accumulator(np.ones(len(ev[out_sel])) * label_mult)
            output['dump_'+dataset]['SS']        += processor.column_accumulator(ak.to_numpy(BL[out_sel]))
            output['dump_'+dataset]['OS']        += processor.column_accumulator(ak.to_numpy(cf_est_sel_mc[out_sel]))
            output['dump_'+dataset]['AR']        += processor.column_accumulator(ak.to_numpy(np_est_sel_mc[out_sel]))
            output['dump_'+dataset]['LL']        += processor.column_accumulator(ak.to_numpy(LL[out_sel]))
            output['dump_'+dataset]['conv']      += processor.column_accumulator(ak.to_numpy(conv_sel[out_sel]))
            output['dump_'+dataset]['weight']    += processor.column_accumulator(ak.to_numpy(weight.weight()[out_sel]))
            output['dump_'+dataset]['weight_np'] += processor.column_accumulator(ak.to_numpy(weight_np_mc[out_sel]))
            output['dump_'+dataset]['weight_cf'] += processor.column_accumulator(ak.to_numpy(weight_cf_mc[out_sel]))
            output['dump_'+dataset]['total_charge'] += processor.column_accumulator(ak.to_numpy(ak.sum(lepton.charge, axis=1)[out_sel]))

        # first, make a few super inclusive plots

        #print (var['name'], self.minimal)
        #if var['name'] == 'central' and not self.minimal:
        if not self.minimal:
            '''
            Don't fill these histograms for the variations
            '''

            output['PV_npvs'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvs, weight=weight_BL)
            output['PV_npvsGood'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
            if var['name'] == 'central':
                # NOTE we don't have systematic uncertainties on the NP estimate for the histograms below.
                fill_multiple_np(output['N_jet'],     {'multiplicity': ak.num(jet)})
                fill_multiple_np(output['N_b'],       {'multiplicity': ak.num(btag)})
                fill_multiple_np(output['N_central'], {'multiplicity': ak.num(central)})
                fill_multiple_np(output['N_ele'],     {'multiplicity':ak.num(electron)})
                fill_multiple_np(output['N_fwd'],     {'multiplicity':ak.num(fwd)})
                fill_multiple_np(output['N_tau'],     {'multiplicity':ak.num(tau)})
                fill_multiple_np(output['ST'],        {'ht': st})
                fill_multiple_np(output['HT'],        {'ht': ht})
                fill_multiple_np(output['MET'],       {'pt':met.pt, 'phi':met.phi})
                fill_multiple_np(output['LT'],        {'ht':lt})
                fill_multiple_np(output['mjj_max'],   {'mass':ak.fill_none(ak.max(mjf, axis=1),0)})
                fill_multiple_np(output['delta_eta_jj'],   {'delta': pad_and_flatten(delta_eta)})
                fill_multiple_np(output['dilepton_mass'],  {'mass': pad_and_flatten(dilepton_mass)})
                fill_multiple_np(output['dilepton_pt'],    {'pt': pad_and_flatten(dilepton_pt)})
                fill_multiple_np(output['min_bl_dR'],      {'delta': ak.fill_none(min_bl_dR,0)})
                fill_multiple_np(output['min_mt_lep_met'], {'mass': ak.fill_none(min_mt_lep_met,0)})

                fill_multiple_np(
                    output['lead_lep'],
                    {
                        'pt':  pad_and_flatten(leading_lepton.p4.pt),
                        'eta': pad_and_flatten(leading_lepton.eta),
                    },
                )
                fill_multiple_np(
                    output['dilepton_mass_onZ'],
                    {'mass': pad_and_flatten(dilepton_mass)},
                    add_sel = (sel.dilep_baseline(only=['SSee_onZ']) & ak.num(electron, axis=1)),
                )

                fill_multiple_np(
                    output['trail_lep'],
                    {
                        'pt':  pad_and_flatten(trailing_lepton.p4.pt),
                        'eta': pad_and_flatten(trailing_lepton.eta),
                    },
                )

                fill_multiple_np(
                    output['fwd_jet'],
                    {
                        'p':   pad_and_flatten(best_fwd.p4.p),
                        'pt':  pad_and_flatten(best_fwd.p4.pt),
                        'eta': pad_and_flatten(best_fwd.p4.eta),
                    },
                )

                fill_multiple_np(
                    output['lead_jet'],
                    {
                        'pt':  pad_and_flatten(jet[:, 0:1].p4.pt),
                        'eta': pad_and_flatten(jet[:, 0:1].p4.eta),
                    },
                )

                fill_multiple_np(
                    output['sublead_jet'],
                    {
                        'pt':  pad_and_flatten(jet[:, 1:2].p4.pt),
                        'eta': pad_and_flatten(jet[:, 1:2].p4.eta),
                    },
                )

                fill_multiple_np(
                    output['lead_bjet'],
                    {
                        'pt':  pad_and_flatten(high_score_btag[:, 0:1].p4.pt),
                        'eta': pad_and_flatten(high_score_btag[:, 0:1].p4.eta),
                    },
                )

                fill_multiple_np(
                    output['sublead_bjet'],
                    {
                        'pt':  pad_and_flatten(high_score_btag[:, 1:2].p4.pt),
                        'eta': pad_and_flatten(high_score_btag[:, 1:2].p4.eta),
                    },
                )

            else:
                # NOTE we run into memory issues when filling too many histograms. don't fill the alternative predictions for variations
                # this is probably not needed except maybe (??) for conversions
                output['N_jet'].fill(dataset=dataset, systematic=var['name'], prediction='central', multiplicity=ak.num(jet)[BL], weight=weight_BL)
                output['N_b'].fill(dataset=dataset, systematic=var['name'], prediction='central',  multiplicity=ak.num(btag)[BL], weight=weight_BL)
                output['N_central'].fill(dataset=dataset, systematic=var['name'], prediction='central', multiplicity=ak.num(central)[BL], weight=weight_BL)
                output['N_ele'].fill(dataset=dataset, systematic=var['name'], prediction='central', multiplicity=ak.num(electron)[BL], weight=weight_BL)
                output['N_fwd'].fill(dataset=dataset, systematic=var['name'], prediction='central', multiplicity=ak.num(fwd)[BL], weight=weight_BL)
                output['N_tau'].fill(dataset=dataset, systematic=var['name'], prediction='central', multiplicity=ak.num(tau)[BL], weight=weight_BL)
                output['ST'].fill(dataset=dataset, systematic=var['name'], prediction='central', ht=st[BL], weight=weight_BL)
                output['HT'].fill(dataset=dataset, systematic=var['name'], prediction='central', ht=ht[BL], weight=weight_BL)
                output['MET'].fill(dataset=dataset, systematic=var['name'], prediction='central', EFT='central', pt=met.pt[BL], phi=met.phi[BL], weight=weight_BL)
                output['LT'].fill(dataset=dataset, systematic=var['name'], prediction='central', EFT='central', ht=lt[BL], weight=weight_BL)
                output['mjj_max'].fill(dataset=dataset, systematic=var['name'], prediction='central', mass=ak.fill_none(ak.max(mjf, axis=1),0)[BL], weight=weight_BL)
                output['delta_eta_jj'].fill(dataset=dataset, systematic=var['name'], prediction='central', delta=pad_and_flatten(delta_eta)[BL], weight=weight_BL)
                output['dilepton_mass'].fill(dataset=dataset, systematic=var['name'], prediction='central', mass=pad_and_flatten(dilepton_mass)[BL], weight=weight_BL)
                output['dilepton_pt'].fill(dataset=dataset, systematic=var['name'], prediction='central', pt=pad_and_flatten(dilepton_pt)[BL], weight=weight_BL)
                output['min_bl_dR'].fill(dataset=dataset, systematic=var['name'], prediction='central', delta=ak.fill_none(min_bl_dR,0)[BL], weight=weight_BL)
                output['min_mt_lep_met'].fill(dataset=dataset, systematic=var['name'], prediction='central', mass=ak.fill_none(min_mt_lep_met,0)[BL], weight=weight_BL)

                SSee_onZ = sel.dilep_baseline(only=['SSee_onZ'])
                output['dilepton_mass_onZ'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    mass=pad_and_flatten(dilepton_mass)[(BL & SSee_onZ & ak.num(electron, axis=1))],
                    weight=weight.weight(modifier=shift)[(BL & SSee_onZ & ak.num(electron, axis=1))],
                )
                # just moving the histograms below cut the memory consumption by 30-50%
                output['lead_lep'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    EFT='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(leading_lepton.p4.pt)[BL],
                    eta=pad_and_flatten(leading_lepton.eta)[BL],
                )

                output['trail_lep'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    EFT='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(trailing_lepton.p4.pt)[BL],
                    eta=pad_and_flatten(trailing_lepton.eta)[BL],
                )

                output['fwd_jet'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    weight=weight_BL,
                    p=pad_and_flatten(best_fwd.p4.p)[BL],
                    pt=pad_and_flatten(best_fwd.p4.pt)[BL],
                    eta=pad_and_flatten(best_fwd.p4.eta)[BL],
                )

                output['lead_jet'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(jet[:, 0:1].p4.pt)[BL],
                    eta=pad_and_flatten(jet[:, 0:1].p4.eta)[BL],
                )

                output['sublead_jet'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(jet[:, 1:2].p4.pt)[BL],
                    eta=pad_and_flatten(jet[:, 1:2].p4.eta)[BL],
                )

                output['lead_bjet'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(high_score_btag[:, 0:1].p4.pt)[BL],
                    eta=pad_and_flatten(high_score_btag[:, 0:1].p4.eta)[BL],
                )

                output['sublead_bjet'].fill(
                    dataset=dataset,
                    systematic=var['name'],
                    prediction='central',
                    weight=weight_BL,
                    pt=pad_and_flatten(high_score_btag[:, 1:2].p4.pt)[BL],
                    eta=pad_and_flatten(high_score_btag[:, 1:2].p4.eta)[BL],
                )

            if not re.search(data_pattern, dataset):
                # NOTE: gen quantities - don't care about systematics
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

            if dataset.count('EFT'):
                # FIXME these should potentially be different histograms???
                for point in self.points:
                    output['MET'].fill(
                        dataset = dataset,
                        systematic = var_name,
                        EFT = point['name'],
                        prediction = 'central',
                        pt  = met[BL].pt,
                        phi  = met[BL].phi,
                        weight = weight_BL*(point['weight'].weight()[BL])
                    )

                    output['lead_lep'].fill(
                        dataset = dataset,
                        systematic = var_name,
                        EFT = point['name'],
                        prediction = 'central',
                        pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
                        eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
                        weight = weight_BL*(point['weight'].weight()[BL])
                    )

                    output['trail_lep'].fill(
                        dataset = dataset,
                        systematic = var_name,
                        EFT = point['name'],
                        prediction = 'central',
                        pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
                        eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
                        weight = weight_BL*(point['weight'].weight()[BL])
                    )

                    output['LT'].fill(
                        dataset = dataset,
                        systematic = var_name,
                        EFT = point['name'],
                        prediction = 'central',
                        ht = ak.to_numpy(lt)[BL],
                        weight = weight_BL*(point['weight'].weight()[BL]),
                    )

                # NOTE this was for debugging, I guess
                #for point in self.weights:
                #    #(getattr(ev.LHEWeight, point)[(BL&SR_sel_pp)]
                #    output['LT'].fill(
                #        dataset = dataset,
                #        systematic = var_name,
                #        EFT = point,
                #        prediction = 'central',
                #        ht = ak.to_numpy(lt)[BL],
                #        weight = weight_BL*(getattr(ev.LHEWeight, point)[BL]),
                #    )

        
        return output

    def postprocess(self, accumulator):
        return accumulator


from analysis.default_accumulators import *

histograms = {
    "bit_score_incl": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
    "bit_score_pp": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
    "bit_score_mm": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
    "ST": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
    "HT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
    "LT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis, eft_axis),
    "lead_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
    "trail_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
    "lead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
    "sublead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
    "lead_bjet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
    "sublead_bjet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
    "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis),
    "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis),
    "LT_SR_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
    "LT_SR_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
    "node": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "node0_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node1_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node2_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node3_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node4_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node0_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node1_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node2_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node3_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node4_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node0_score_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node0_score_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node0_score_transform_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node0_score_transform_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node1_score_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "node1_score_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
    "PS": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
    "scale": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
    "pdf": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
    "norm": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
    "MET": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, phi_axis, eft_axis),
    "fwd_jet":      hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, p_axis, pt_axis, eta_axis),
    "N_b" :         hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "N_ele" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "N_central" :   hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "N_jet" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "N_fwd" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "N_tau" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
    "mjj_max":      hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ext_mass_axis),
    "delta_eta_jj": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, delta_axis),
    "dilepton_mass": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
    "dilepton_mass_onZ": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
    "dilepton_pt": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis),
    "min_bl_dR": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, delta_axis),
    "min_mt_lep_met": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
}

# JES uncertainties from here https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources
variations_jet_all_list = [
     # all kept in v0.8.0: https://github.com/danbarto/nanoAOD-tools/blob/topW_v0.8.0/scripts/run_processor.py#L96
     'jesAbsoluteStat',
     'jesAbsoluteScale',
     'jesAbsoluteMPFBias',
     'jesFragmentation',
     'jesSinglePionECAL',
     'jesSinglePionHCAL',
     'jesFlavorQCD',  # default, others currently not kept in our samples
     'jesTimePtEta',
     'jesRelativeJEREC1',
     'jesRelativeJEREC2',
     'jesRelativeJERHF',
     'jesRelativePtBB',
     'jesRelativePtEC1',
     'jesRelativePtEC2',
     'jesRelativePtHF',
     'jesRelativeBal',
     'jesRelativeSample',
     'jesRelativeFSR',
     'jesRelativeStatFSR',
     'jesRelativeStatEC',
     'jesRelativeStatHF',
     'jesPileUpDataMC',
     'jesPileUpPtRef',
     'jesPileUpPtBB',
     'jesPileUpPtEC1',
     'jesPileUpPtEC2',
     'jesPileUpPtHF',
]


def make_vars_from_list(in_list):
    vars = []
    for var in in_list:
        vars.append({'name': f'{var}_up', 'ext': f'_pt_{var}Up', 'weight': None, 'pt_var': f'pt_{var}Up'})
        vars.append({'name': f'{var}_down', 'ext': f'_pt_{var}Down', 'weight': None, 'pt_var': f'pt_{var}Down'})
    return vars

variations_jet_all = make_vars_from_list(variations_jet_all_list)

central_variation = [
    {'name': 'central',                 'ext': '',                  'weight': None,   'pt_var': 'pt_nom'},
]

base_variations = [
    {'name': 'jer_up',                  'ext': '_pt_jerUp',         'weight': None,   'pt_var': 'pt_jerUp'},# https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution#Smearing_procedures
    {'name': 'jer_down',                'ext': '_pt_jerDown',       'weight': None,   'pt_var': 'pt_jerDown'},
    {'name': 'jesTotal_up',             'ext': '_pt_jesTotalUp',    'weight': None,   'pt_var': 'pt_jesTotalUp'},
    {'name': 'jesTotal_down',           'ext': '_pt_jesTotalDown',  'weight': None,   'pt_var': 'pt_jesTotalDown'},
    {'name': 'PU_up',                   'ext': '_PUUp',             'weight': 'PUUp', 'pt_var': 'pt_nom'},
    {'name': 'PU_down',                 'ext': '_PUDown',           'weight': 'PUDown', 'pt_var': 'pt_nom'},
    {'name': 'b_up',                    'ext': '_bUp',              'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'b_down',                  'ext': '_bDown',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'l_up',                    'ext': '_lUp',              'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'l_down',                  'ext': '_lDown',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'ele_up',                  'ext': '_eleUp',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'ele_down',                'ext': '_eleDown',          'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'mu_up',                   'ext': '_muUp',             'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'mu_down',                 'ext': '_muDown',           'weight': None,    'pt_var': 'pt_nom'},
    ]

nonprompt_variations = [
    {'name': 'fake_el_up',              'ext': 'el_up',             'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_down',            'ext': 'el_down',           'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_up',              'ext': 'mu_up',             'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_down',            'ext': 'mu_down',           'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_pt1',             'ext': 'mu_pt1',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_pt2',             'ext': 'mu_pt2',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_pt1',             'ext': 'el_pt1',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_pt2',             'ext': 'el_pt2',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_be1',             'ext': 'mu_be1',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_be2',             'ext': 'mu_be2',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_be1',             'ext': 'el_be1',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_be2',             'ext': 'el_be2',            'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_closure_up',      'ext': 'mu_closure_up',     'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_mu_closure_down',    'ext': 'mu_closure_down',   'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_closure_up',      'ext': 'el_closure_up',     'weight': None,    'pt_var': 'pt_nom'},
    {'name': 'fake_el_closure_down',    'ext': 'el_closure_down',   'weight': None,    'pt_var': 'pt_nom'},
    ]

variations = central_variation + base_variations + variations_jet_all

def check_jes_components(output, in_list, direction='up'):
    central = output['bit_score_pp'].integrate('EFT', 'cpt_0_cpqm_0').integrate('prediction', 'central').sum('bit').sum('dataset').integrate('systematic', 'central').values()[()]
    unc = 0
    for var in in_list:
        if var['name'].count(direction):
            var_tmp = output['bit_score_pp'].integrate('EFT', 'cpt_0_cpqm_0').integrate('prediction', 'central').sum('bit').sum('dataset').integrate('systematic', var['name']).values()[()]
            unc += (var_tmp - central)**2
    return np.sqrt(unc)

if __name__ == '__main__':

    from processor.default_accumulators import *
    from Tools.reweighting import get_coordinates_and_ref, get_coordinates

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=False, help="Rerun or try using existing results??")
    argParser.add_argument('--minimal', action='store_true', default=False, help="Only run minimal set of histograms")
    argParser.add_argument('--dask', action='store_true', default=False, help="Run on a DASK cluster?")
    argParser.add_argument('--central', action='store_true', default=False, help="Only run the central value (no systematics)")
    argParser.add_argument('--profile', action='store_true', default=False, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=False, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=False, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--workers', action='store', default=10, help="How many threads for local running?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    argParser.add_argument('--sample', action='store', default='all', )
    argParser.add_argument('--cpt', action='store', default=0, help="Select the cpt point")
    argParser.add_argument('--cpqm', action='store', default=0, help="Select the cpqm point")
    argParser.add_argument('--buaf', action='store', default="false", help="Run on BU AF")
    argParser.add_argument('--skim', action='store', default="topW_v0.7.1_SS", help="Define the skim to run on")
    argParser.add_argument('--scan', action='store_true', default=None, help="Run the entire cpt/cpqm scan")
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = args.rerun
    small       = args.small

    year        = int(args.year[0:4])
    ul          = "UL%s"%(args.year[2:])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config
    cfg = loadConfig()


    # FIXME do sth like this??
    data_variations = [

    ]

    if args.central: variations = variations[:1]

    # inclusive EFT weights
    eft_weights = [\
        'cpt_0p_cpqm_0p_nlo',
        'cpt_0p_cpqm_3p_nlo',
        'cpt_0p_cpqm_6p_nlo',
        'cpt_3p_cpqm_0p_nlo',
        'cpt_6p_cpqm_0p_nlo',
        'cpt_3p_cpqm_3p_nlo',
    ]

    # NOTE new way of defining points.
    if args.scan:
        x = np.arange(-7,8,1)
        y = np.arange(-7,8,1)
    else:
        x = np.array([int(args.cpt)])
        y = np.array([int(args.cpqm)])

    CPT, CPQM = np.meshgrid(x, y)

    points = []
    for cpt, cpqm in zip(CPT.flatten(), CPQM.flatten()):
        points.append({
            'name': f'eft_cpt_{cpt}_cpqm_{cpqm}',
            'point': [cpt, cpqm],
        })

    if args.buaf == 'remote':
        f_in = 'root://redirector.t2.ucsd.edu:1095//store/user/dspitzba/nanoAOD/ttw_samples//topW_v0.7.0_dilep/ProjectMetis_TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO_RunIISummer20UL18_NanoAODv9_NANO_v14/merged/nanoSkim_1.root'
    elif args.buaf == 'local':
        f_in = '/media/data_hdd/daniel/ttw_samples/topW_v0.7.0_dilep/ProjectMetis_TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO_RunIISummer20UL16_postVFP_NanoAODv9_NANO_v14/merged/nanoSkim_1.root'
    else:
        f_in = '/ceph/cms/store/user/dspitzba/nanoAOD/ttw_samples//topW_v0.7.0_dilep/ProjectMetis_TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO_RunIISummer20UL18_NanoAODv9_NANO_v14/merged/nanoSkim_1.root'

    coordinates, ref_coordinates = get_coordinates_and_ref(f_in)
    coordinates = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (6.0, 0.0), (3.0, 3.0), (0.0, 6.0)]
    ref_coordinates = [0,0]

    from Tools.awkwardHyperPoly import *
    hp = HyperPoly(2)
    hp.initialize( coordinates, ref_coordinates )


    samples = get_samples("samples_%s.yaml"%ul)
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    if args.sample == 'MCall':
        sample_list = ['DY', 'topW_lep', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
    elif args.sample == 'data':
        if year == 2018:
            sample_list = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        else:
            sample_list = ['DoubleMuon', 'MuonEG', 'DoubleEG', 'SingleMuon', 'SingleElectron']
    else:
        sample_list = [args.sample]

    cutflow_output = {}

    for sample in sample_list:
        # NOTE we could also rescale processes here?
        #
        print (f"Working on samples: {sample}")

        # NOTE we could also rescale processes here?
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
        fileset = make_fileset(
            [sample],
            samples,
            year=ul,
            #skim='topW_v0.7.0_dilep',
            skim=args.skim,
            small=small,
            n_max=1,
            buaf=args.buaf,
            merged=True,
        )

        # define the cache name
        cache_name = f'SS_analysis_{sample}_{year}{era}'
        if not args.scan:
            cache_name += f'cpt_{args.cpt}_cpqm_{args.cpqm}'
        # find an old existing output
        output = get_latest_output(cache_name, cfg)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_name += f'_{timestamp}.coffea'
        if small: cache_name += '_small'
        cache = os.path.join(os.path.expandvars(cfg['caches']['base']), cache_name)

        if overwrite or output is None:
            ## Try running all files separately
            outputs = []
            for f in fileset.keys():

                fileset_tmp = {f:fileset[f]}
                add_processes_to_output(fileset_tmp, desired_output)

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
                        'nLepFromTop',
                        'label',
                        'total_charge',
                        'event',
                    ]

                    if sample.count('topW'):
                        print ("topW sample")
                        variables += eft_weights

                    for dataset in mapping[ul][sample]:

                        desired_output.update({
                            'dump_%s'%dataset: processor.dict_accumulator({})
                        })#processor.column_accumulator(np.zeros(shape=(0,))),
                        for var in variables:
                            desired_output['dump_%s'%dataset].update({var: processor.column_accumulator(np.zeros(shape=(0,)))})
                            #'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                            #'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                            #})

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

                # add some histograms that we defined in the processor
                # everything else is taken the default_accumulators.py
                from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, one_axis, systematic_axis, eft_axis, charge_axis, pred_axis, bit_axis
                from processor.default_accumulators import ext_mass_axis, delta_axis, p_axis

                desired_output.update({
                    "bit_score_incl": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
                    "bit_score_pp": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
                    "bit_score_mm": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, bit_axis),
                })
                if not args.minimal:
                    desired_output.update({
                        "ST": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
                        "HT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
                        "LT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis, eft_axis),
                        "lead_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
                        "trail_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
                        "lead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                        "sublead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                        "lead_bjet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                        "sublead_bjet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                        "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis),
                        "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis),
                        "LT_SR_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
                        "LT_SR_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
                        "node": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "node0_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node1_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node2_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node3_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node4_score_incl": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node0_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node1_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node2_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node3_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node4_score": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node0_score_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node0_score_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node0_score_transform_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node0_score_transform_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node1_score_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "node1_score_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, score_axis),
                        "PS": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
                        "scale": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
                        "pdf": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
                        "norm": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
                        "MET": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, phi_axis, eft_axis),
                        "fwd_jet":      hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, p_axis, pt_axis, eta_axis),
                        "N_b" :         hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "N_ele" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "N_central" :   hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "N_jet" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "N_fwd" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "N_tau" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                        "mjj_max":      hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ext_mass_axis),
                        "delta_eta_jj": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, delta_axis),
                        "dilepton_mass": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
                        "dilepton_mass_onZ": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
                        "dilepton_pt": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis),
                        "min_bl_dR": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, delta_axis),
                        "min_mt_lep_met": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, mass_axis),
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


                output = runner(
                    fileset_tmp,
                    treename="Events",
                    processor_instance=SS_analysis(
                        year=year,
                        variations=variations,
                        #variations=variations[:1],
                        accumulator=desired_output,
                        evaluate=args.evaluate,
                        training=args.training,
                        dump=args.dump,
                        era=era,
                        weights=eft_weights,
                        reweight=reweight,
                        points=points,
                        hyperpoly=hp,
                        minimal=args.minimal,
                    ),
                )

                outputs.append(output)

            output = accumulate(outputs)
            util.save(output, cache)

        ## output for DNN training
        labels = {'topW': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'top': 4, 'rare':5, 'diboson':6, 'XG': 7, 'topW_lep': 0}
        if sample in labels:
            label_mult = labels[sample]
        else:
            label_mult = 8  # data or anything else

        if args.dump:
            df_dict = {var: np.zeros(shape=(0,)) for var in variables}
            df_dict['label'] = np.zeros(shape=(0,))
            for dataset in mapping[ul][sample]:
                for var in variables:
                    if var == 'weight':
                        # NOTE: not modifying the CF/NP weights
                        lumi_fac = renorm[dataset] * float(samples[dataset]['xsec']) * cfg['lumi'][year] * 1000 / float(samples[dataset]['sumWeight'])
                        df_dict[var] = np.concatenate([df_dict[var], (output['dump_'+dataset][var].value * lumi_fac)])
                    else:
                        df_dict[var] = np.concatenate([df_dict[var], output['dump_'+dataset][var].value])

                df_dict['label'] = np.concatenate([df_dict['label'], label_mult*np.ones_like(output['dump_'+dataset][var].value)])

            df_out = pd.DataFrame( df_dict )

            if not args.small:
                df_out.to_hdf(f"multiclass_input_{sample}_{args.year}.h5", key='df', format='table', mode='w')

        if args.evaluate:
            print ("\nNN debugging:")
            print (output['node'].sum('multiplicity').values())

        # Scale the cutflow output. This should be packed into a function?
        cutflow_output[sample] = {}
        dataset_0 = mapping[ul][sample][0]

        print ("Scaling to {}/fb".format(cfg['lumi'][year]))
        for dataset in mapping[ul][sample]:
            print ("Sample {}".format(dataset))
            print ("sigma*BR: {}".format(float(samples[dataset]['xsec']) * cfg['lumi'][year] * 1000))

        for key in output[dataset_0]:
            cutflow_output[sample][key] = 0.
            for dataset in mapping[ul][sample]:
                if float(samples[dataset]['sumWeight'])>0:
                    cutflow_output[sample][key] += (renorm[dataset]*output[dataset][key] * float(samples[dataset]['xsec']) * cfg['lumi'][year] * 1000 / float(samples[dataset]['sumWeight']))
                else:
                    cutflow_output[sample][key] += output[dataset][key]

        if not local:
            # clean up the DASK workers. this partially frees up memory on the workers
            c.cancel(output)
            # NOTE: this really restarts the cluster, but is the only fully effective
            # way of deallocating all the accumulated memory...
            c.restart()

    from Tools.helpers import getCutFlowTable
    #if args.sample == 'MCall':
    #    processes = ['topW_lep', 'TTW', 'TTZ', 'TTH', 'rare', 'diboson', 'XG', 'top']
    #elif args.sample == 'data':
    #    processes = []
    #else:
    #    processes = [args.sample]
    lines= [
            'filter',
            'dilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS',
            'N_jet>3',
            'N_central>2',
            'N_btag>0',
            'N_light>0',
            'MET>30',
            'N_fwd>0',
            'min_mll'
        ]

    print (getCutFlowTable(cutflow_output,
                           processes=sample_list,
                           lines=lines,
                           significantFigures=3,
                           absolute=True,
                           #signal='topW_v3',
                           total=False,
                           ))

    #from Tools.config_helpers import get_merged_output
    #output_scaled = get_merged_output('SS_analysis', str(year), select_datasets=processes)

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

    make_plots = False
    if make_plots:
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
