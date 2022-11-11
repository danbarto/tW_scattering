import os
import re
import datetime
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak
import glob

from coffea import processor, hist, util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.processor import accumulate

import numpy as np
import pandas as pd

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, nonprompt_no_conv, external_conversion
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM, get_samples
from Tools.config_helpers import loadConfig, make_small, data_pattern, get_latest_output, load_yaml, data_path
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer


class trilep_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False, era=None, weights=[], hyperpoly=None, points=[[]]):
        self.variations = variations
        self.year = year
        self.era = era  # this is here for 2016 APV
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        
        self.btagSF = btag_scalefactor(year, era=era)
        
        self.leptonSF = LeptonSF(year=year)
        self.nonpromptWeight = NonpromptWeight(year=year)
        
        self._accumulator = processor.dict_accumulator( accumulator )

        #self.weights = weights
        self.hyperpoly = hyperpoly
        self.points = points

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
        if not re.search(data_pattern, dataset):
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
        mu_v     = Collections(ev, "Muon", "vetoTTH", year=year).get()  # these include all muons, tight and fakeable
        mu_t     = Collections(ev, "Muon", "tightTTH", year=year).get()
        mu_f     = Collections(ev, "Muon", "fakeableSSTTH", year=year).get()
        
        mu_v_mask = Collections(ev, "Muon", "vetoTTH", year=year)
        mu_f_mask = Collections(ev, "Muon", "fakeableTTH", year=year)
        mu_t_mask = Collections(ev, "Muon", "tightTTH", year=year)
        mu_v['id'] = (0*mu_v_mask.selection + 1*mu_f_mask.selection + 2*mu_t_mask.selection)[mu_v_mask.selection]

        muon     = ak.concatenate([mu_t, mu_f], axis=1)  # FIXME do I still need this in trilep?
        muon['p4'] = get_four_vec_fromPtEtaPhiM(muon, get_pt(muon), muon.eta, muon.phi, muon.mass, copy=False)
        mu_v['p4'] = get_four_vec_fromPtEtaPhiM(mu_v, get_pt(mu_v), mu_v.eta, mu_v.phi, mu_v.mass, copy=False)
        
        ## Electrons
        el_v        = Collections(ev, "Electron", "vetoTTH", year=year).get()
        el_t        = Collections(ev, "Electron", "tightTTH", year=year).get()
        el_f        = Collections(ev, "Electron", "fakeableTTH", year=year).get()

        el_v_mask = Collections(ev, "Electron", "vetoTTH", year=year)
        el_f_mask = Collections(ev, "Electron", "fakeableTTH", year=year)
        el_t_mask = Collections(ev, "Electron", "tightTTH", year=year)
        el_v['id'] = (0*el_v_mask.selection + 1*el_f_mask.selection + 2*el_t_mask.selection)[el_v_mask.selection]

        electron    = ak.concatenate([el_t, el_f], axis=1)
        electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False)
        el_v['p4'] = get_four_vec_fromPtEtaPhiM(el_v, get_pt(el_v), el_v.eta, el_v.phi, el_v.mass, copy=False)
        
        if not re.search(data_pattern, dataset):
            gen_photon = ev.GenPart[ev.GenPart.pdgId==22]
            # tight electrons
            el_t_p  = prompt(el_t)
            el_t_np = nonprompt_no_conv(el_t, gen_photon)
            el_t_conv = external_conversion(el_t, gen_photon)
            #el_t_np = nonprompt(el_t)
            # fakeable electrons
            el_f_p  = prompt(el_f)
            el_f_np = nonprompt_no_conv(el_f, gen_photon)
            #el_f_np = nonprompt(el_f)
            # loose/veto electrons
            el_v_p  = prompt(el_v)

            mu_t_p  = prompt(mu_t)
            mu_t_np = nonprompt_no_conv(mu_t, gen_photon)
            mu_t_conv = external_conversion(mu_t, gen_photon)
            #mu_t_np = nonprompt(mu_t)

            mu_f_p  = prompt(mu_f)
            mu_f_np = nonprompt_no_conv(mu_f, gen_photon)
            #mu_f_np = nonprompt(mu_f)

            mu_v_p  = prompt(mu_v)

        ## Merge electrons and muons. These are fakeable leptons now
        lepton   = ak.concatenate([muon, electron], axis=1)
        lead_leptons = lepton[ak.argsort(lepton.p4.pt)][:,:3]
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.p4.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.p4.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]

        dimuon = choose(muon, 2)
        #OS_dimuon_sel = (dimuon['0'].charge*dimuon['1'].charge)<0
        OS_dimuon = dimuon[(dimuon['0'].charge*dimuon['1'].charge)<0]
        dielectron = choose(electron, 2)
        OS_dielectron = dielectron[(dielectron['0'].charge*dielectron['1'].charge)<0]
        #OS_dielectron_sel = (dielectron['0'].charge*dielectron['1'].charge)<0
        #dilepton = ak.concatenate([dimuon, dielectron], axis=1)
        #OS_dilepton_sel = ak.concatenate([OS_dimuon_sel, OS_dielectron_sel], axis=1)
        #OS_dilepton = dilepton[OS_dilepton_sel]
        #OS_dilepton_Z_cand_idx = ak.singletons(ak.argmin(abs(OS_dilepton.mass-91.2), axis=1))

        N_SFOS = ak.num(OS_dimuon, axis=1) + ak.num(OS_dielectron, axis=1)

        # NOTE add the 4vecs to get the corrected momenta, not the uncorrected (default) ones
        OS_dimuon_mass = (OS_dimuon['0'].p4 + OS_dimuon['1'].p4).mass
        OS_dielectron_mass = (OS_dielectron['0'].p4 + OS_dielectron['1'].p4).mass

        SFOS_mass = ak.concatenate([OS_dimuon_mass, OS_dielectron_mass], axis=1)
        SFOS_mass_best = SFOS_mass[ak.singletons(ak.argmin(abs(SFOS_mass-91.2), axis=1))]

        trilep = choose(lepton, 3)
        M3l = (trilep['0'].p4 + trilep['1'].p4 + trilep['2'].p4).mass
        trilep_q = ak.sum(electron.charge, axis=1) + ak.sum(muon.charge, axis=1)

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
            lt = ak.sum(electron.p4.pt, axis=1) + ak.sum(muon.p4.pt, axis=1) + met.pt

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

            #high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

            #bl          = cross(lepton, high_score_btag)
            #bl_dR       = delta_r(bl['0'], bl['1'])
            #min_bl_dR   = ak.min(bl_dR, axis=1)

            ## forward jets
            j_fwd = fwd[ak.singletons(ak.argmax(fwd.p4.p, axis=1))] # highest momentum spectator

            # try to get either the most forward light jet, or if there's more than one with eta>1.7, the highest pt one
            most_fwd = light[ak.argsort(abs(light.eta))][:,0:1]
            #most_fwd = light[ak.singletons(ak.argmax(abs(light.eta)))]
            best_fwd = ak.concatenate([j_fwd, most_fwd], axis=1)[:,0:1]


            #################
            ### Variables ###
            #################
            jf          = cross(j_fwd, jet)
            mjf         = (jf['0'].p4+jf['1'].p4).mass
            j_fwd2      = jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'] # this is the jet that forms the largest invariant mass with j_fwd
            delta_eta   = abs(j_fwd2.eta - j_fwd.eta)

            ## other variables
            ht = ak.sum(jet.pt, axis=1)
            st = met.pt + ht + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)
            lt = met.pt + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)

            mt_lep_met = mt(lepton.p4.pt, lepton.p4.phi, met.pt, met.phi)
            min_mt_lep_met = ak.min(mt_lep_met, axis=1)
            
        
            # define the weight
            weight = Weights( len(ev) )
            
            if not re.search(data_pattern, dataset):
                # lumi weight
                weight.add("weight", ev.genWeight)
                #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
                
                # PU weight - not in the babies...
                weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
                
                # b-tag SFs
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
            
            cutflow     = Cutflow(output, ev, weight=weight)

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
                met = met,
            )

            baseline = sel.trilep_baseline(cutflow=cutflow)
            
            if not re.search(data_pattern, dataset):

                BL = (baseline & ((ak.num(el_t_p)+ak.num(mu_t_p))==3) & ((ak.num(el_v)+ak.num(mu_v))==3) )  #
                BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))==3) & ((ak.num(el_v)+ak.num(mu_v))==3) )

                np_est_sel_mc = (baseline & \
                    ((((ak.num(el_t_p)+ak.num(mu_t_p))>=1) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=2)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=3)) | (((ak.num(el_t_p)+ak.num(mu_t_p))>=2) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=1)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
                np_obs_sel_mc = (baseline & ( (ak.num(el_t)+ak.num(mu_t))>=3) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
                np_est_sel_data = (baseline & ~baseline)  # this has to be false

                if dataset.count("TTTo") or dataset.count("DY"):
                    conv_sel = BL  # anything that has tight, prompt, charge-consistent, non-external-conv, same-sign dileptons has to be internal conversion.
                elif dataset.count("Gamma") or dataset.count("WGTo") or dataset.count("ZGTo") or dataset.count("WZG_"):
                    conv_sel = BL_incl & (((ak.num(el_t_conv)+ak.num(mu_t_conv))>0))
                else:
                    conv_sel = (baseline & ~baseline)  # this has to be false

                weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')

            else:
                BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))>=3))

                BL_incl = BL

                np_est_sel_mc = (baseline & ~baseline)
                np_obs_sel_mc = (baseline & ~baseline)
                np_est_sel_data = (baseline & \
                    ((((ak.num(el_t)+ak.num(mu_t))>=1) & ((ak.num(el_f)+ak.num(mu_f))>=2)) \
                   | (((ak.num(el_t)+ak.num(mu_t))==0) & ((ak.num(el_f)+ak.num(mu_f))>=3)) \
                   | (((ak.num(el_t)+ak.num(mu_t))>=2) & ((ak.num(el_f)+ak.num(mu_f))>=1)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
                #np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)>=1) & (ak.num(el_f)+ak.num(mu_f)>=1) )

                conv_sel = (baseline & ~baseline)  # this has to be false

                weight_np_mc = np.zeros(len(ev))

            weight_BL = weight.weight()[BL]  # this is just a shortened weight list for the two prompt selection
            weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data')

            out_sel = (BL | np_est_sel_mc)
            
            dummy = (np.ones(len(ev))==1)
            dummy_weight = Weights(len(ev))

            def fill_multiple_np(hist, arrays, add_sel=dummy, other=None, weight_multiplier=dummy_weight.weight()):
                reg_sel = [
                    BL&add_sel,
                    BL_incl&add_sel,
                    np_est_sel_mc&add_sel,
                    np_obs_sel_mc&add_sel,
                    np_est_sel_data&add_sel,
                    conv_sel&add_sel,
                ]
                fill_multiple(
                    hist,
                    dataset=dataset,
                    predictions = [
                        "central",
                        "inclusive", # everything from process (inclusive MC truth)
                        "np_est_mc", # MC based NP estimate
                        "np_obs_mc", # MC based NP observation
                        "np_est_data",
                        "conv_mc",
                    ],
                    arrays=arrays,
                    selections=reg_sel,
                    weights=[
                        weight_multiplier[reg_sel[0]]*weight.weight(modifier=shift)[reg_sel[0]],
                        weight_multiplier[reg_sel[1]]*weight.weight(modifier=shift)[reg_sel[1]],
                        weight_multiplier[reg_sel[2]]*weight.weight(modifier=shift)[reg_sel[2]]*weight_np_mc[reg_sel[2]],
                        weight_multiplier[reg_sel[3]]*weight.weight(modifier=shift)[reg_sel[3]],
                        weight_multiplier[reg_sel[4]]*weight.weight(modifier=shift)[reg_sel[4]]*weight_np_data[reg_sel[4]],
                        weight_multiplier[reg_sel[5]]*weight.weight(modifier=shift)[reg_sel[5]],
                    ],
                    systematic=var_name,
                    other = other,
                )

            ttZ_sel = sel.trilep_baseline(only=['N_btag>0', 'onZ', 'MET>30'])
            WZ_sel  = sel.trilep_baseline(only=['N_btag=0', 'onZ', 'MET>30'])
            XG_sel  = sel.trilep_baseline(only=['N_btag=0', 'offZ'])
            sig_sel = sel.trilep_baseline(only=['N_btag>0', 'N_jet>2', 'offZ', 'N_fwd>0'])

            if var['name'] == 'central':
                '''
                Don't fill these histograms for the variations
                '''
                # first, make a few super inclusive plots
                output['PV_npvs'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvs, weight=weight_BL)
                output['PV_npvsGood'].fill(dataset=dataset, systematic=var['name'], multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
                fill_multiple_np(output['N_jet'],     {'multiplicity': ak.num(jet)})
                fill_multiple_np(output['N_b'],       {'multiplicity': ak.num(btag)})
                fill_multiple_np(output['N_central'], {'multiplicity': ak.num(central)})
                fill_multiple_np(output['N_ele'],     {'multiplicity':ak.num(electron)})
                fill_multiple_np(output['N_fwd'],     {'multiplicity':ak.num(fwd)})
                fill_multiple_np(output['ST'],        {'ht': st})
                fill_multiple_np(output['HT'],        {'ht': ht})
                fill_multiple_np(output['MET'],       {'pt':ev.MET.pt, 'phi':ev.MET.phi})

                fill_multiple_np(
                    output['dilepton_mass'],
                    {'mass': ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)},
                    #add_sel = (N_SFOS>0)
                )

                fill_multiple_np(
                    output['dilepton_mass_WZ'],
                    {'mass': ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)},
                    add_sel = WZ_sel
                )

                fill_multiple_np(
                    output['dilepton_mass_ttZ'],
                    {'mass': ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)},
                    add_sel = ttZ_sel
                )

                fill_multiple_np(
                    output['dilepton_mass_XG'],
                    {'mass': ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)},
                    add_sel = XG_sel
                )

                fill_multiple_np(
                    output['dilepton_mass_topW'],
                    {'mass': ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)},
                    add_sel = sig_sel
                )

                if dataset.count('EFT'):
                    eft_points = self.points
                else:
                    eft_points = [{
                        'name': f'eft_cpt_0_cpqm_0',
                        'point': [0,0],
                    }]
                for p in eft_points:
                    x,y = p['point']
                    point = p['point']
                    if dataset.count('EFT'):
                        eft_weight = self.hyperpoly.eval(ev.Pol, point)
                    else:
                        eft_weight = dummy_weight.weight()
                    fill_multiple_np(
                        output['signal_region_topW'],
                        {
                            'lt': lt,
                            'N': N_SFOS,
                            'charge': trilep_q,
                            },
                        add_sel = sig_sel,
                        other = {'EFT': f'eft_cpt_{x}_cpqm_{y}'},
                        weight_multiplier = eft_weight,
                    )

                fill_multiple_np(
                    output['lead_lep'],
                    {
                        'pt':  pad_and_flatten(leading_lepton.p4.pt),
                        'eta': pad_and_flatten(leading_lepton.eta),
                    },
                )

                fill_multiple_np(
                    output['trail_lep'],
                    {
                        'pt':  pad_and_flatten(trailing_lepton.p4.pt),
                        'eta': pad_and_flatten(trailing_lepton.eta),
                    },
                )


                if not re.search(data_pattern, dataset) and var['name'] == 'central' and len(variations) > 1:
                    add_sel = sig_sel
                    for i in range(1,101):
                        pdf_ext = "pdf_%s"%i
                        output['signal_region_topW'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = 'central',
                            lt          = lt[(BL & add_sel)],
                            N           = N_SFOS[(BL & add_sel)],
                            charge      = trilep_q[(BL & add_sel)],
                            weight      = weight.weight()[(BL & add_sel)] * ev.LHEPdfWeight[:,i][(BL & add_sel)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & add_sel)],
                        )

                    for i in ([0,1,3,5,7,8] if not (dataset.count('EFT') or dataset.count('ZZTo2Q2L_mllmin4p0')) else [0,1,3,4,6,7]):
                        pdf_ext = "scale_%s"%i
                        output['signal_region_topW'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = "central",
                            lt          = lt[(BL & add_sel)],
                            N           = N_SFOS[(BL & add_sel)],
                            charge      = trilep_q[(BL & add_sel)],
                            weight      = weight.weight()[(BL & add_sel)] * ev.LHEScaleWeight[:,i][(BL & add_sel)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & add_sel)],
                        )

                    if len(ev.PSWeight[0]) > 1:
                        for i in range(4):
                            pdf_ext = "PS_%s"%i
                            output['signal_region_topW'].fill(
                                dataset     = dataset,
                                systematic  = pdf_ext,
                                prediction  = 'central',
                                EFT         = "central",
                                lt          = lt[(BL & add_sel)],
                                N           = N_SFOS[(BL & add_sel)],
                                charge      = trilep_q[(BL & add_sel)],
                                weight      = weight.weight()[(BL & add_sel)] * ev.PSWeight[:,i][(BL & add_sel)],
                            )

                    add_sel = ttZ_sel
                    for i in range(1,101):
                        pdf_ext = "pdf_%s"%i
                        output['dilepton_mass_ttZ'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = 'central',
                            mass        = ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)[(BL & add_sel)],
                            weight      = weight.weight()[(BL & add_sel)] * ev.LHEPdfWeight[:,i][(BL & add_sel)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & add_sel)],
                        )

                    for i in ([0,1,3,5,7,8] if not (dataset.count('EFT') or dataset.count('ZZTo2Q2L_mllmin4p0')) else [0,1,3,4,6,7]):
                        pdf_ext = "scale_%s"%i
                        output['dilepton_mass_ttZ'].fill(
                            dataset     = dataset,
                            systematic  = pdf_ext,
                            prediction  = 'central',
                            EFT         = "central",
                            mass        = ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)[(BL & add_sel)],
                            weight      = weight.weight()[(BL & add_sel)] * ev.LHEScaleWeight[:,i][(BL & add_sel)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & add_sel)],
                        )

                    if len(ev.PSWeight[0]) > 1:
                        for i in range(4):
                            pdf_ext = "PS_%s"%i
                            output['dilepton_mass_ttZ'].fill(
                                dataset     = dataset,
                                systematic  = pdf_ext,
                                prediction  = 'central',
                                EFT         = "central",
                                mass        = ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)[(BL & add_sel)],
                                weight      = weight.weight()[(BL & add_sel)] * ev.PSWeight[:,i][(BL & add_sel)],
                            )
            
                #output['j1'].fill(
                #    dataset = dataset,
                #    pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
                #    eta = ak.flatten(jet.eta[:, 0:1][BL]),
                #    weight = weight_BL
                #)
                #
                #output['j2'].fill(
                #    dataset = dataset,
                #    pt  = ak.flatten(jet[:, 1:2][BL].pt_nom),
                #    eta = ak.flatten(jet[:, 1:2][BL].eta),
                #    weight = weight_BL
                #)
                #
                #output['j3'].fill(
                #    dataset = dataset,
                #    pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
                #    eta = ak.flatten(jet[:, 2:3][BL].eta),
                #    weight = weight_BL
                #)
                
                fill_multiple_np(
                    output['fwd_jet'],
                    {
                        'pt':  pad_and_flatten(best_fwd.pt),
                        'p':  pad_and_flatten(best_fwd.p4.p),
                        'eta': pad_and_flatten(best_fwd.eta),
                    },
                )
            else:
                if not re.search(data_pattern, dataset):
                    # similar to SS_analysis
                    # Don't fill for data
                    output['signal_region_topW'].fill(
                        dataset     = dataset,
                        systematic  = var['name'],
                        prediction  = 'central',
                        EFT         = 'central',
                        lt          = lt[(BL & sig_sel)],
                        N           = N_SFOS[(BL & sig_sel)],
                        charge      = trilep_q[(BL & sig_sel)],
                        weight      = weight.weight(modifier=shift)[(BL & sig_sel)],
                    )

                    output['dilepton_mass_ttZ'].fill(
                        dataset     = dataset,
                        systematic  = var['name'],
                        prediction  = 'central',
                        EFT         = 'central',
                        mass        = ak.fill_none(pad_and_flatten(SFOS_mass_best), 0)[(BL & ttZ_sel)],
                        weight      = weight.weight(modifier=shift)[(BL & ttZ_sel)],
                    )
                

        return output

    def postprocess(self, accumulator):
        return accumulator




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

    
    # define points
    if args.scan:
        x = np.arange(-7,8,1)
        y = np.arange(-7,8,1)
    else:
        x = x = np.array([int(args.cpt)])
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
    hp.initialize(coordinates,ref_coordinates)


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
        cache_name = f'trilep_analysis_{sample}_{year}{era}'
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


                if local and not iterative:# and not profile:
                    exe = processor.FuturesExecutor(workers=int(args.workers))

                elif iterative:
                    exe = processor.IterativeExecutor()

                else:
                    from Tools.helpers import get_scheduler_address
                    from dask.distributed import Client, progress

                    scheduler_address = get_scheduler_address()
                    c = Client(scheduler_address)

                    exe = processor.DaskExecutor(client=c, status=True, retries=3)

                from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis, pred_axis, systematic_axis, mass_axis, pred_axis, eft_axis
                sr_axis = hist.Bin("lt",  r"LT", 10, 0, 1000)
                charge_axis = hist.Bin("charge",  r"q", 3, -1.5, 1.5)
                nossf_axis = hist.Bin("N",  r"N", 3, -0.5, 2.5)

                desired_output.update({
                    "ST": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
                    "HT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis),
                    "LT": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, ht_axis, eft_axis),
                    "lead_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
                    "trail_lep": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, pt_axis, eta_axis),
                    "lead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                    "sublead_jet": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, eta_axis),
                    "LT_SR_pp": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
                    "LT_SR_mm": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, eft_axis, ht_axis),
                    "MET": hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, pt_axis, phi_axis, eft_axis),
                    "fwd_jet":      hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, p_axis, pt_axis, eta_axis),
                    "N_b" :         hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "N_ele" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "N_central" :   hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "N_jet" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "N_fwd" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "N_tau" :       hist.Hist("Counts", dataset_axis, pred_axis, systematic_axis, multiplicity_axis),
                    "dilepton_mass": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, mass_axis),
                    "dilepton_mass_WZ": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, mass_axis),
                    "dilepton_mass_XG": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, mass_axis),
                    "dilepton_mass_ttZ": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, mass_axis),
                    "dilepton_mass_topW": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, mass_axis),
                    "signal_region_topW": hist.Hist("Counts", dataset_axis, eft_axis, pred_axis, systematic_axis, sr_axis, charge_axis, nossf_axis),
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
                    processor_instance=trilep_analysis(
                        year=year,
                        variations=variations,
                        #variations=variations[:1],
                        accumulator=desired_output,
                        evaluate=args.evaluate,
                        #training=args.training,
                        #dump=args.dump,
                        era=era,
                        #weights=eft_weights,
                        #reweight=reweight,
                        points=points,
                        hyperpoly=hp,
                        #minimal=args.minimal,
                    ),
                )

                outputs.append(output)

            output = accumulate(outputs)
            util.save(output, cache)

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
                try:
                    cutflow_output[sample][key] += (renorm[dataset]*output[dataset][key] * float(samples[dataset]['xsec']) * cfg['lumi'][year] * 1000 / float(samples[dataset]['sumWeight']))
                except ZeroDivisionError:
                    cutflow_output[sample][key] += output[dataset][key]

        if not local:
            # clean up the DASK workers. this partially frees up memory on the workers
            c.cancel(output)
            # NOTE: this really restarts the cluster, but is the only fully effective
            # way of deallocating all the accumulated memory...
            c.restart()

    from Tools.helpers import getCutFlowTable
    processes = ['topW_lep', 'TTW', 'TTZ', 'TTH', 'rare', 'diboson', 'XG', 'top'] if args.sample == 'MCall' else [args.sample]
    lines= [
            'filter',
            'trigger',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'p_T(lep2)>10',
            'N_jet>1',
            'N_central>1',
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
