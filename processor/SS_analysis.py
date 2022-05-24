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
from Tools.btag_scalefactors import btag_scalefactor
from Tools.trigger_scalefactors import triggerSF
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip
from Tools.pileup import pileup

import warnings
warnings.filterwarnings("ignore")

#from ML.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer


class SS_analysis(processor.ProcessorABC):
    def __init__(self,
                 year=2016,
                 era=None,
                 variations=[],
                 accumulator={},
                 evaluate=False,
                 training='v8',
                 dump=False,
                 hyperpoly=None,  # maybe can go?
                 points=[[]],  # maybe can go?
                 weights=[],
                 reweight=1,
                 ):
        self.variations = variations

        print (variations)

        self.year = year
        self.era = era  # this is here for 2016 APV
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        
        self.btagSF = btag_scalefactor(year, era=era)
        self.leptonSF = LeptonSF(year=year, era=era)
        self.triggerSF = triggerSF(year=year)
        self.pu = pileup(year=year, UL=True, era=era)

        self.nonpromptWeight = NonpromptWeight(year=year)  # NOTE no era split
        self.chargeflipWeight = charge_flip(year=year)  # NOTE no era split

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
        
        output = self.accumulator.identity()
        
        # use a very loose preselection to filter the events
        presel = ak.num(events.Jet)>2
        
        ev = events[presel]
        dataset = ev.metadata['dataset']
        
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

        track     = getIsoTracks(ev)

        # this is where the real JEC dependent stuff happens

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
                weight.add("lepton", self.leptonSF.get(electron, muon))

                # trigger SFs
                weight.add("trigger", self.triggerSF.get(electron, muon))
            
            if dataset=='topW_full_EFT':
                # FIXME legacy code ready to be retired
                for point in self.points:
                    point['weight'] = Weights( len(ev) )
                    point['weight'].add("EFT", self.hyperpoly.eval(ev.Pol, point['point']))

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
                weight_np_mc_qcd = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='QCD')
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

            if self.evaluate or self.dump:
                # define the inputs to the NN
                # this is super stupid. there must be a better way.
                # used a np.stack which is ok performance wise. pandas data frame seems to be slow and memory inefficient

                NN_inputs_d = {
                    'n_jet':            ak.to_numpy(ak.num(jet)),
                    'n_fwd':            ak.to_numpy(ak.num(fwd)),
                    'n_b':              ak.to_numpy(ak.num(btag)),
                    'n_tau':            ak.to_numpy(ak.num(tau)),
                    #'n_track':          ak.to_numpy(ak.num(track)),
                    'st':               ak.to_numpy(st),
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
                    'event':            ak.to_numpy(ev.event),
                    'nLepFromTop':      ak.to_numpy(ev.nLepFromTop),
                }

                if dataset.count('TTW_5f_EFT') or dataset.count('EFT'):
                    for w in self.weights:
                        NN_inputs_d.update({w: ak.to_numpy(getattr(ev.LHEWeight, w))})

                if self.dump and var['name'] == 'central':
                    for k in NN_inputs_d.keys():
                        output['dump_'+dataset][k] += processor.column_accumulator(NN_inputs_d[k][out_sel])

                if self.evaluate:
                
                    NN_inputs = np.stack( [NN_inputs_d[k] for k in NN_inputs_d.keys()] )

                    NN_inputs = np.nan_to_num(NN_inputs, 0, posinf=1e5, neginf=-1e5)  # events with posinf/neginf/nan will not pass the BL selection anyway

                    NN_inputs = np.moveaxis(NN_inputs, 0, 1)  # this is needed for a np.stack (old version)

                    model, scaler = load_onnx_model('%s%s_%s'%(self.year, self.era, self.training))

                    try:
                        #NN_inputs_scaled = scaler.transform(NN_inputs)
                        #NN_pred    = predict_onnx(model, NN_inputs_scaled)

                        best_score = np.argmax(NN_pred, axis=1)


                    except ValueError:
                        print ("Problem with prediction. Showing the shapes here:")
                        print (np.shape(NN_inputs))
                        NN_pred = np.array([])
                        best_score = np.array([])
                        NN_inputs_scaled = NN_inputs
                        raise

            weight_BL = weight.weight(modifier=shift)[BL]  # this is just a shortened weight list for the two prompt selection
            weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data')
            weight_cf_data = self.chargeflipWeight.flip_weight(el_t)

            #out_sel = (BL | np_est_sel_mc | cf_est_sel_mc)

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
                    np_est_sel_mc&add_sel,  # MC based NP estimate with QCD FR
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
                        "np_est_mc_qcd",  # MC based NP estimate with QCD FR
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
                        weight.weight(modifier=shift)[reg_sel[0][9]]*weight_np_mc_qcd[reg_sel[0][9]],
                    ],
                    systematic = var_name,  # NOTE check this.
                )

            #if self.evaluate or self.dump:
            if self.evaluate:
                blind_sel = ((data_sel & (best_score>1)) | ~data_sel)
                if var['name'] == 'central':

                    #fill_multiple_np(output['node'], {'multiplicity':best_score}, add_sel=blind_sel)  # this should blind me
                    fill_multiple_np(output['node0_score_incl'], {'score':NN_pred[:,0]})
                    fill_multiple_np(output['node1_score_incl'], {'score':NN_pred[:,1]})
                    fill_multiple_np(output['node2_score_incl'], {'score':NN_pred[:,2]})
                    fill_multiple_np(output['node3_score_incl'], {'score':NN_pred[:,3]})
                    fill_multiple_np(output['node4_score_incl'], {'score':NN_pred[:,4]})
                    
                    fill_multiple_np(output['node0_score'], {'score':NN_pred[:,0]}, add_sel=(best_score==0))
                    #fill_multiple_np(output['node1_score'], {'score':NN_pred[:,1]}, add_sel=(best_score==1))
                    fill_multiple_np(output['node2_score'], {'score':NN_pred[:,2]}, add_sel=(best_score==2))
                    fill_multiple_np(output['node3_score'], {'score':NN_pred[:,3]}, add_sel=(best_score==3))
                    fill_multiple_np(output['node4_score'], {'score':NN_pred[:,4]}, add_sel=(best_score==4))

                SR_sel_pp = ((best_score==0) & (ak.sum(lepton.charge, axis=1)>0))
                SR_sel_mm = ((best_score==0) & (ak.sum(lepton.charge, axis=1)<0))

                CR_sel_pp = ((best_score==1) & (ak.sum(lepton.charge, axis=1)>0))
                CR_sel_mm = ((best_score==1) & (ak.sum(lepton.charge, axis=1)<0))


                if dataset.count('TTW_5f_EFT'):

                    for point in self.points:
                        # FIXME this is legacy EFT stuff and can go?
                        output['lead_lep_SR_pp'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            pt  = ak.to_numpy(pad_and_flatten(leading_lepton.p4.pt[(BL&SR_sel_pp)])),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_pp)]*(point['weight'].weight()[(BL&SR_sel_pp)]))
                        )

                        output['lead_lep_SR_mm'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            pt  = ak.to_numpy(pad_and_flatten(leading_lepton.p4.pt[(BL&SR_sel_mm)])),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_mm)]*(point['weight'].weight()[(BL&SR_sel_mm)]))
                        )

                        output['LT_SR_pp'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            ht  = ak.to_numpy(lt[(BL&SR_sel_pp)]),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_pp)]*(point['weight'].weight()[(BL&SR_sel_pp)]))
                        )

                        output['LT_SR_mm'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            ht  = ak.to_numpy(lt[(BL&SR_sel_mm)]),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_mm)]*(point['weight'].weight()[(BL&SR_sel_mm)]))
                        )

                    for point in self.weights:
                        output['lead_lep_SR_pp'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point,
                            pt  = ak.to_numpy(pad_and_flatten(leading_lepton.p4.pt[(BL&SR_sel_pp)])),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_pp)]*(getattr(ev.LHEWeight, point)[(BL&SR_sel_pp)]))
                        )

                        output['lead_lep_SR_mm'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point,
                            pt  = ak.to_numpy(pad_and_flatten(leading_lepton.p4.pt[(BL&SR_sel_mm)])),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_mm)]*(getattr(ev.LHEWeight, point)[(BL&SR_sel_mm)]))
                        )

                        output['LT_SR_pp'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point,
                            ht  = ak.to_numpy(lt[(BL&SR_sel_pp)]),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_pp)]*(getattr(ev.LHEWeight, point)[(BL&SR_sel_pp)]))
                        )

                        output['LT_SR_mm'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point,
                            ht  = ak.to_numpy(lt[(BL&SR_sel_mm)]),
                            weight = (weight.weight(modifier=shift)[(BL&SR_sel_mm)]*(getattr(ev.LHEWeight, point)[(BL&SR_sel_mm)]))
                        )

                else:

                    fill_multiple_np(output['lead_lep_SR_pp'], {'pt':  pad_and_flatten(leading_lepton.p4.pt)}, add_sel=SR_sel_pp)
                    fill_multiple_np(output['lead_lep_SR_mm'], {'pt':  pad_and_flatten(leading_lepton.p4.pt)}, add_sel=SR_sel_mm)

                    fill_multiple_np(output['LT_SR_pp'], {'ht':  lt}, add_sel=SR_sel_pp)
                    fill_multiple_np(output['LT_SR_mm'], {'ht':  lt}, add_sel=SR_sel_mm)

                fill_multiple_np(output['node'], {'multiplicity':best_score}, add_sel=blind_sel)  # this should blind me
                fill_multiple_np(output['node1_score'], {'score':NN_pred[:,1]}, add_sel=(best_score==1))

                fill_multiple_np(output['node0_score_pp'], {'score': NN_pred[:,0]}, add_sel=SR_sel_pp)
                fill_multiple_np(output['node0_score_mm'], {'score': NN_pred[:,0]}, add_sel=SR_sel_mm)

                fill_multiple_np(output['node1_score_pp'], {'score': NN_pred[:,1]}, add_sel=CR_sel_pp)
                fill_multiple_np(output['node1_score_mm'], {'score': NN_pred[:,1]}, add_sel=CR_sel_mm)

                transformer = load_transformer('%s%s_%s'%(self.year, self.era, self.training))

                NN_pred_0_trans = transformer.transform(NN_pred[:,0].reshape(-1, 1)).flatten()

                fill_multiple_np(output['node0_score_transform_pp'], {'score': NN_pred_0_trans}, add_sel=SR_sel_pp)
                fill_multiple_np(output['node0_score_transform_mm'], {'score': NN_pred_0_trans}, add_sel=SR_sel_mm)

                if var['name'] == 'central':
                    output["norm"].fill(
                        dataset = dataset,
                        systematic = var['name'],
                        one   = ak.ones_like(met.pt),
                        weight  = weight.weight(),
                    )

                # Manually hack in the PDF weights - we don't really want to have them for all the distributions
                if not re.search(data_pattern, dataset) and var['name'] == 'central':
                    for i in range(1,101):
                        pdf_ext = "pdf_%s"%i

                        output['pdf'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            one   = ak.ones_like(ev.LHEPdfWeight[:,i]),
                            weight  = weight.weight() * ev.LHEPdfWeight[:,i] if len(ev.LHEPdfWeight[0])>0 else weight.weight(),
                        )

                        output['node0_score_transform_pp'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score   = NN_pred_0_trans[(BL & SR_sel_pp)],
                            weight  = weight.weight()[(BL & SR_sel_pp)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['node0_score_transform_mm'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score   = NN_pred_0_trans[(BL & SR_sel_mm)],
                            weight  = weight.weight()[(BL & SR_sel_mm)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                        output['node1_score'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score = NN_pred[:,1][(BL & (best_score==1))],
                            weight = weight.weight()[(BL & (best_score==1))] * ev.LHEPdfWeight[:,i][(BL & (best_score==1))] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & (best_score==1))],
                        )

                        output['node'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            multiplicity = best_score[(BL & blind_sel)],
                            weight = weight.weight()[(BL & blind_sel)] * ev.LHEPdfWeight[:,i][(BL & blind_sel)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & blind_sel)],
                        )

                        output['LT_SR_pp'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            ht = lt[(BL & SR_sel_pp)],
                            weight  = weight.weight()[(BL & SR_sel_pp)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['LT_SR_mm'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            ht = lt[(BL & SR_sel_mm)],
                            weight  = weight.weight()[(BL & SR_sel_mm)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                    for i in [0,1,3,5,7,8]:
                        pdf_ext = "scale_%s"%i

                        output['scale'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            one   = ak.ones_like(ev.LHEScaleWeight[:,i]),
                            weight  = weight.weight() * ev.LHEScaleWeight[:,i] if len(ev.LHEScaleWeight[0])>0 else weight.weight(),
                        )

                        output['node0_score_transform_pp'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score   = NN_pred_0_trans[(BL & SR_sel_pp)],
                            weight  = weight.weight()[(BL & SR_sel_pp)] * ev.LHEScaleWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['node0_score_transform_mm'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score   = NN_pred_0_trans[(BL & SR_sel_mm)],
                            weight  = weight.weight()[(BL & SR_sel_mm)] * ev.LHEScaleWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                        output['node1_score'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            score = NN_pred[:,1][(BL & (best_score==1))],
                            weight = weight.weight()[(BL & (best_score==1))] * ev.LHEScaleWeight[:,i][(BL & (best_score==1))] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & (best_score==1))],
                        )

                        output['node'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            multiplicity = best_score[(BL & blind_sel)],
                            weight = weight.weight()[(BL & blind_sel)] * ev.LHEScaleWeight[:,i][(BL & blind_sel)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & blind_sel)]
                        )

                        output['LT_SR_pp'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            ht = lt[(BL & SR_sel_pp)],
                            weight  = weight.weight()[(BL & SR_sel_pp)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                        )

                        output['LT_SR_mm'].fill(
                            dataset = dataset,
                            systematic = pdf_ext,
                            ht = lt[(BL & SR_sel_mm)],
                            weight  = weight.weight()[(BL & SR_sel_mm)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                        )

                    if len(ev.PSWeight[0]) > 1:
                        for i in range(4):
                            pdf_ext = "PS_%s"%i

                            output['PS'].fill( # NOTE: these should be obsolete now
                                dataset = dataset,
                                systematic = pdf_ext,
                                one   = ak.ones_like(ev.LHEPdfWeight[:,i]),
                                weight  = weight.weight() * ev.PSWeight[:,i],
                            )

                            output['node0_score_transform_pp'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                score   = NN_pred_0_trans[(BL & SR_sel_pp)],
                                weight  = weight.weight()[(BL & SR_sel_pp)] * ev.PSWeight[:,i][(BL & SR_sel_pp)],
                            )

                            output['node0_score_transform_mm'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                score   = NN_pred_0_trans[(BL & SR_sel_mm)],
                                weight  = weight.weight()[(BL & SR_sel_mm)] * ev.PSWeight[:,i][(BL & SR_sel_mm)],
                            )

                            output['node1_score'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                score = NN_pred[:,1][(BL & (best_score==1))],
                                weight = weight.weight()[(BL & (best_score==1))] * ev.PSWeight[:,i][(BL & (best_score==1))],
                            )

                            output['node'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                multiplicity = best_score[(BL & blind_sel)],
                                weight = weight.weight()[(BL & blind_sel)] * ev.PSWeight[:,i][(BL & blind_sel)]
                            )

                            output['LT_SR_pp'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                ht = lt[(BL & SR_sel_pp)],
                                weight  = weight.weight()[(BL & SR_sel_pp)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_pp)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_pp)],
                            )
    
                            output['LT_SR_mm'].fill(
                                dataset = dataset,
                                systematic = pdf_ext,
                                ht = lt[(BL & SR_sel_mm)],
                                weight  = weight.weight()[(BL & SR_sel_mm)] * ev.LHEPdfWeight[:,i][(BL & SR_sel_mm)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel_mm)],
                            )
                #del model
                #del scaler
                #del NN_inputs
                #del NN_inputs_scaled, NN_pred


            if self.dump and var['name']=='central':
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

            if var['name'] == 'central':
                '''
                Don't fill these histograms for the variations
                '''

                output['PV_npvs'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvs, weight=weight_BL)
                output['PV_npvsGood'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
                fill_multiple_np(output['N_jet'],     {'multiplicity': ak.num(jet)})
                fill_multiple_np(output['N_b'],       {'multiplicity': ak.num(btag)})
                fill_multiple_np(output['N_central'], {'multiplicity': ak.num(central)})
                fill_multiple_np(output['N_ele'],     {'multiplicity':ak.num(electron)})
                fill_multiple_np(output['N_mu'],      {'multiplicity':ak.num(muon)})
                fill_multiple_np(output['N_fwd'],     {'multiplicity':ak.num(fwd)})
                fill_multiple_np(output['ST'],        {'ht': st})
                fill_multiple_np(output['HT'],        {'ht': ht})

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

                if dataset=='topW_full_EFT':
                    for point in self.points:
                        output['MET'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            pt  = met[BL].pt,
                            phi  = met[BL].phi,
                            weight = weight_BL*(point['weight'].weight()[BL])
                        )

                        output['lead_lep'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
                            eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
                            phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
                            weight = weight_BL*(point['weight'].weight()[BL])
                        )
                        
                        output['trail_lep'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
                            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
                            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
                            weight = weight_BL*(point['weight'].weight()[BL])
                        )

                        output['LT'].fill(
                            dataset = dataset,
                            systematic = var_name,
                            eft = point['name'],
                            ht = ak.to_numpy(lt)[BL],
                            weight = weight_BL*(point['weight'].weight()[BL]),
                        )

                else:

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
                    
                output['high_p_fwd_p'].fill(
                    dataset=dataset,
                    systematic = var_name,
                    p = ak.flatten(best_fwd[BL].p),
                    weight = weight_BL,
                )
                
            output['j1'].fill(
                dataset = dataset,
                systematic = var_name,
                pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
                eta = ak.flatten(jet.eta[:, 0:1][BL]),
                phi = ak.flatten(jet.phi[:, 0:1][BL]),
                weight = weight_BL
            )
            
            output['j2'].fill(
                dataset = dataset,
                systematic = var_name,
                pt  = ak.flatten(jet[:, 1:2][BL].pt_nom),
                eta = ak.flatten(jet[:, 1:2][BL].eta),
                phi = ak.flatten(jet[:, 1:2][BL].phi),
                weight = weight_BL
            )
            
            output['j3'].fill(
                dataset = dataset,
                systematic = var_name,
                pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
                eta = ak.flatten(jet[:, 2:3][BL].eta),
                phi = ak.flatten(jet[:, 2:3][BL].phi),
                weight = weight_BL
            )
                    
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from processor.default_accumulators import *

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=None, help="Rerun or try using existing results??")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on a DASK cluster?")
    argParser.add_argument('--central', action='store_true', default=None, help="Only run the central value (no systematics)")
    argParser.add_argument('--profile', action='store_true', default=None, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=None, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    argParser.add_argument('--sample', action='store', default='all', )
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

    samples = get_samples("samples_%s.yaml"%ul)
    mapping = load_yaml(data_path+"nano_mapping.yaml")


    if args.sample == 'MCall':
        sample_list = ['DY', 'topW', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
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

        # inclusive EFT weights
        eft_weights = [\
            'cpt_0p_cpqm_0p_nlo',
            'cpt_0p_cpqm_3p_nlo',
            'cpt_0p_cpqm_6p_nlo',
            'cpt_3p_cpqm_0p_nlo',
            'cpt_6p_cpqm_0p_nlo',
            'cpt_3p_cpqm_3p_nlo',
        ]

        if args.central: variations = variations[:1]

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
        from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, one_axis, systematic_axis, eft_axis, charge_axis
        desired_output.update({
            "ST": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "HT": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "LT": hist.Hist("Counts", dataset_axis, systematic_axis, ht_axis),
            "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, systematic_axis, eft_axis, pt_axis),
            "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, systematic_axis, eft_axis, pt_axis),
            "LT_SR_pp": hist.Hist("Counts", dataset_axis, systematic_axis, eft_axis, ht_axis),
            "LT_SR_mm": hist.Hist("Counts", dataset_axis, systematic_axis, eft_axis, ht_axis),
            "node": hist.Hist("Counts", dataset_axis, systematic_axis, multiplicity_axis),
            "node0_score_incl": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node1_score_incl": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node2_score_incl": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node3_score_incl": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node4_score_incl": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node0_score": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node1_score": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node2_score": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node3_score": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node4_score": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node0_score_pp": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node0_score_mm": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node0_score_transform_pp": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node0_score_transform_mm": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node1_score_pp": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "node1_score_mm": hist.Hist("Counts", dataset_axis, systematic_axis, score_axis),
            "PS": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
            "scale": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
            "pdf": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
            "norm": hist.Hist("Counts", dataset_axis, systematic_axis, one_axis),
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
        print ("Runner properties")
        print (runner.retries)
        #runner.automatic_retries(3)
        #print (runner.retries)

        # define the cache name
        cache_name = f'SS_analysis_{sample}_{year}{era}'
        # find an old existing output
        output = get_latest_output(cache_name, cfg)

        if overwrite or output is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_name += f'_{timestamp}.coffea'
            if small: cache_name += '_small'
            cache = os.path.join(os.path.expandvars(cfg['caches']['base']), cache_name)

            output = runner(
                fileset,
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
                ),
            )

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
                cutflow_output[sample][key] += (renorm[dataset]*output[dataset][key] * float(samples[dataset]['xsec']) * cfg['lumi'][year] * 1000 / float(samples[dataset]['sumWeight']))

        if not local:
            # clean up the DASK workers. this partially frees up memory on the workers
            c.cancel(output)
            # NOTE: this really restarts the cluster, but is the only fully effective
            # way of deallocating all the accumulated memory...
            c.restart()

    from Tools.helpers import getCutFlowTable
    processes = ['topW', 'TTW', 'TTZ', 'TTH', 'rare', 'diboson', 'XG', 'top'] if args.sample == 'MCall' else [args.sample]
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
                           processes=processes,
                           lines=lines,
                           significantFigures=3,
                           absolute=True,
                           #signal='topW_v3',
                           total=False,
                           ))

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
