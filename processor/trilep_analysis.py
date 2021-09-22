import os
import re
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak
import glob

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, nonprompt_no_conv, external_conversion
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


class trilep_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False, era=None):
        self.variations = variations
        self.year = year
        self.era = era  # this is here for 2016 APV
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)
        self.nonpromptWeight = NonpromptWeight(year=year)
        
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
        OS_dimuon = dimuon[(dimuon['0'].charge*dimuon['1'].charge)<0]
        dielectron = choose(electron, 2)
        OS_dielectron = dielectron[(dielectron['0'].charge*dielectron['1'].charge)<0]

        n_sfos = ak.num(OS_dimuon, axis=1) + ak.num(OS_dielectron, axis=1)

        OS_dimuon_mass = (OS_dimuon['0'].p4 + OS_dimuon['1'].p4).mass
        OS_dielectron_mass = (OS_dielectron['0'].p4 + OS_dielectron['1'].p4).mass

        SFOS_mass = ak.concatenate([OS_dimuon_mass,OS_dielectron_mass], axis=1)
        SFOS_mass_sorted = SFOS_mass[ak.argsort(abs(SFOS_mass-91.2))]

        trilep = choose(lepton, 3)
        m3l = (trilep['0'].p4 + trilep['1'].p4 + trilep['2'].p4).mass

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

            pt_var  = var['pt_var']
            ext     = var['ext']
            shift   = var['weight']

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
                weight.add("weight", ev.weight*cfg['lumi'][self.year])
                #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
                
                # PU weight - not in the babies...
                weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
                
                # b-tag SFs
                if var['name'] == 'l_up':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='central', c_direction='up'))
                elif var['name'] == 'l_down':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='central', c_direction='down'))
                elif var['name'] == 'b_up':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='up', c_direction='central'))
                elif var['name'] == 'b_down':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='down', c_direction='central'))
                else:
                    weight.add("btag", self.btagSF.Method1a(btag, light))
                
                # lepton SFs
                weight.add("lepton", self.leptonSF.get(electron, muon))  # FIXME this needs to be evaluated for loose, too
            
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

            baseline = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0'])
            
            if not re.search(data_pattern, dataset):
                # The baseline selection is at least three loose leptons, at least two tight with SS.
                # For the way we estimate the background, I need to ask for the tight leptons to be prompt
                # Can I allow for loose fakes?

                BL = (baseline & ((ak.num(el_t_p)+ak.num(mu_t_p))>=3) & ((ak.num(el_v)+ak.num(mu_v))>=3) )  # 
                BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))>=3) & ((ak.num(el_v)+ak.num(mu_v))>=3) )

                np_est_sel_mc = (baseline & \
                    ((((ak.num(el_t_p)+ak.num(mu_t_p))>=1) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=2)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=3)) | (((ak.num(el_t_p)+ak.num(mu_t_p))>=2) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=1)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
                np_obs_sel_mc = (baseline & ( (ak.num(el_t)+ak.num(mu_t))>=3) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
                np_est_sel_data = (baseline & ~baseline)  # this has to be false

                if dataset == 'top':
                    conv_sel = BL  # anything that has tight, prompt, charge-consistent, non-external-conv, same-sign dileptons has to be internal conversion.
                elif dataset == 'XG':
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
            def fill_multiple_np(hist, arrays, add_sel=dummy):
                reg_sel = [
                    BL&add_sel,
                    BL_incl&add_sel,
                    np_est_sel_mc&add_sel,
                    np_obs_sel_mc&add_sel,
                    np_est_sel_data&add_sel,
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
                        "conv_mc",
                    ],
                    arrays=arrays,
                    selections=reg_sel[0],  # no idea where the additional dimension is coming from...
                    weights=[
                        weight.weight()[reg_sel[0][0]],
                        weight.weight()[reg_sel[0][1]],
                        weight.weight()[reg_sel[0][2]]*weight_np_mc[reg_sel[0][2]],
                        weight.weight()[reg_sel[0][3]],
                        weight.weight()[reg_sel[0][4]]*weight_np_data[reg_sel[0][4]],
                        weight.weight()[reg_sel[0][5]],
                    ],
                )

            if self.evaluate or self.dump:
                # define the inputs to the NN
                # this is super stupid. there must be a better way.
                # used a np.stack which is ok performance wise. pandas data frame seems to be slow and memory inefficient
                #FIXME no n_b, n_fwd back in v13/v14 of the DNN

                NN_inputs_d = {
                    'n_jet':            ak.to_numpy(ak.num(jet)),
                    'n_fwd':            ak.to_numpy(ak.num(fwd)),
                    'n_b':              ak.to_numpy(ak.num(btag)),
                    'n_tau':            ak.to_numpy(ak.num(tau)),
                    'n_ele':            ak.to_numpy(ak.num(electron)),
                    'n_sfos':           ak.to_numpy(n_sfos),
                    'charge':           ak.to_numpy(ak.sum(lepton.charge, axis=1)),
                    #'n_track':          ak.to_numpy(ak.num(track)),
                    'st':               ak.to_numpy(st),
                    'lt':               ak.to_numpy(lt),
                    'met':              ak.to_numpy(met.pt),
                    'mjj_max':          ak.to_numpy(ak.fill_none(ak.max(mjf, axis=1),0)),
                    'delta_eta_jj':     ak.to_numpy(pad_and_flatten(delta_eta)),
                    'lead_lep_pt':      ak.to_numpy(pad_and_flatten(lead_leptons[:,0:1].p4.pt)),
                    'lead_lep_eta':     ak.to_numpy(pad_and_flatten(lead_leptons[:,0:1].p4.eta)),
                    'sublead_lep_pt':   ak.to_numpy(pad_and_flatten(lead_leptons[:,1:2].p4.pt)),
                    'sublead_lep_eta':  ak.to_numpy(pad_and_flatten(lead_leptons[:,1:2].p4.eta)),
                    'trail_lep_pt':     ak.to_numpy(pad_and_flatten(lead_leptons[:,2:3].p4.pt)),
                    'trail_lep_eta':    ak.to_numpy(pad_and_flatten(lead_leptons[:,2:3].p4.eta)),
                    'm3l':              ak.to_numpy(pad_and_flatten(m3l)),
                    'close_mass':       ak.to_numpy(pad_and_flatten(SFOS_mass[:,0:1])),
                    'far_mass':         ak.to_numpy(pad_and_flatten(SFOS_mass[:,1:2])),
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

                if self.dump:
                    for k in NN_inputs_d.keys():
                        output[k] += processor.column_accumulator(NN_inputs_d[k][out_sel])

                if self.evaluate:
                
                    NN_inputs = np.stack( [NN_inputs_d[k] for k in NN_inputs_d.keys()] )

                    NN_inputs = np.nan_to_num(NN_inputs, 0, posinf=1e5, neginf=-1e5)  # events with posinf/neginf/nan will not pass the BL selection anyway

                    NN_inputs = np.moveaxis(NN_inputs, 0, 1)  # this is needed for a np.stack (old version)

                    model, scaler = load_onnx_model('%s%s_%s'%(self.year, self.era if self.era else '', self.training))

                    try:
                        NN_inputs_scaled = scaler.transform(NN_inputs)

                        NN_pred    = predict_onnx(model, NN_inputs_scaled)

                        best_score = np.argmax(NN_pred, axis=1)


                    except ValueError:
                        print ("Problem with prediction. Showing the shapes here:")
                        print (np.shape(NN_inputs))
                        print (np.shape(weight_BL))
                        NN_pred = np.array([])
                        best_score = np.array([])
                        NN_inputs_scaled = NN_inputs
                        raise

            if self.evaluate or self.dump:
                if var['name'] == 'central':
                    fill_multiple_np(output['node0_score_incl'], {'score':NN_pred[:,0]})
                    fill_multiple_np(output['node1_score_incl'], {'score':NN_pred[:,1]})
                    fill_multiple_np(output['node2_score_incl'], {'score':NN_pred[:,2]})
                    fill_multiple_np(output['node3_score_incl'], {'score':NN_pred[:,3]})
                    
                    fill_multiple_np(output['node2_score'], {'score':NN_pred[:,2]}, add_sel=(best_score==2))
                    fill_multiple_np(output['node3_score'], {'score':NN_pred[:,3]}, add_sel=(best_score==3))

                fill_multiple_np(output['node'+ext], {'multiplicity':best_score})
                fill_multiple_np(output['node0_score'+ext], {'score':NN_pred[:,0]}, add_sel=(best_score==0))
                fill_multiple_np(output['node1_score'+ext], {'score':NN_pred[:,1]}, add_sel=(best_score==1))


                transformer = load_transformer('%s%s_%s'%(self.year, self.era if self.era else '', self.training))
                NN_pred_0_trans = transformer.transform(NN_pred[:,0].reshape(-1, 1)).flatten()

                fill_multiple_np(output['node0_score_transform'+ext], {'score': NN_pred_0_trans}, add_sel=(best_score==0))

                SR_sel = (best_score==0)

                if var['name'] == 'central':
                    output["norm"].fill(
                        dataset = dataset,
                        one   = ak.ones_like(met.pt),
                        weight  = weight.weight(),
                    )

                # Manually hack in the PDF weights - we don't really want to have them for all the distributions
                if not re.search(data_pattern, dataset) and var['name'] == 'central' and dataset.count('rare')==0 and dataset.count('diboson')==0:  # FIXME: rare excluded because of missing samples
                    for i in range(1,101):
                        pdf_ext = "_pdf_%s"%i

                        output[pdf_ext].fill(
                            dataset = dataset,
                            one   = ak.ones_like(ev.LHEPdfWeight[:,i]),
                            weight  = weight.weight() * ev.LHEPdfWeight[:,i] if len(ev.LHEPdfWeight[0])>0 else weight.weight(),
                        )

                        output['node0_score_transform'+pdf_ext].fill(
                            dataset = dataset,
                            score   = NN_pred_0_trans[(BL & SR_sel)],
                            weight  = weight.weight()[(BL & SR_sel)] * ev.LHEPdfWeight[:,i][(BL & SR_sel)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & SR_sel)],
                        )

                        output['node1_score'+pdf_ext].fill(
                            dataset = dataset,
                            score = NN_pred[:,1][(BL & (best_score==1))],
                            weight = weight.weight()[(BL & (best_score==1))] * ev.LHEPdfWeight[:,i][(BL & (best_score==1))] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL & (best_score==1))],
                        )

                        output['node'+pdf_ext].fill(
                            dataset = dataset,
                            multiplicity = best_score[(BL)],
                            weight = weight.weight()[(BL)] * ev.LHEPdfWeight[:,i][(BL)] if len(ev.LHEPdfWeight[0])>0 else weight.weight()[(BL)],
                        )

                    for i in [0,1,3,5,7,8]:
                        pdf_ext = "_scale_%s"%i

                        output[pdf_ext].fill(
                            dataset = dataset,
                            one   = ak.ones_like(ev.LHEScaleWeight[:,i]),
                            weight  = weight.weight() * ev.LHEScaleWeight[:,i] if len(ev.LHEScaleWeight[0])>0 else weight.weight(),
                        )

                        output['node0_score_transform'+pdf_ext].fill(
                            dataset = dataset,
                            score   = NN_pred_0_trans[(BL & SR_sel)],
                            weight  = weight.weight()[(BL & SR_sel)] * ev.LHEScaleWeight[:,i][(BL & SR_sel)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & SR_sel)],
                        )

                        output['node1_score'+pdf_ext].fill(
                            dataset = dataset,
                            score = NN_pred[:,1][(BL & (best_score==1))],
                            weight = weight.weight()[(BL & (best_score==1))] * ev.LHEScaleWeight[:,i][(BL & (best_score==1))] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL & (best_score==1))],
                        )

                        output['node'+pdf_ext].fill(
                            dataset = dataset,
                            multiplicity = best_score[(BL)],
                            weight = weight.weight()[(BL)] * ev.LHEScaleWeight[:,i][(BL)] if len(ev.LHEScaleWeight[0])>0 else weight.weight()[(BL)]
                        )

                    if len(ev.PSWeight[0]) > 1:
                        for i in range(4):
                            pdf_ext = "_PS_%s"%i

                            output['node0_score_transform'+pdf_ext].fill(
                                dataset = dataset,
                                score   = NN_pred_0_trans[(BL & SR_sel)],
                                weight  = weight.weight()[(BL & SR_sel)] * ev.PSWeight[:,i][(BL & SR_sel)],
                            )

                            output['node1_score'+pdf_ext].fill(
                                dataset = dataset,
                                score = NN_pred[:,1][(BL & (best_score==1))],
                                weight = weight.weight()[(BL & (best_score==1))] * ev.PSWeight[:,i][(BL & (best_score==1))],
                            )

                            output['node'+pdf_ext].fill(
                                dataset = dataset,
                                multiplicity = best_score[(BL)],
                                weight = weight.weight()[(BL)] * ev.PSWeight[:,i][(BL)]
                            )

            labels = {'topW_v3': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'ttbar': 4, 'rare':5, 'diboson':6, 'XG':7, 'topW_old':100}  # these should be all?
            if dataset in labels:
                label_mult = labels[dataset]
            else:
                label_mult = 8  # data or anything else

            if self.dump:
                output['label']     += processor.column_accumulator(np.ones(len(ev[out_sel])) * label_mult)
                output['trilep']    += processor.column_accumulator(ak.to_numpy(BL[out_sel]))
                output['AR']        += processor.column_accumulator(ak.to_numpy(np_est_sel_mc[out_sel]))
                output['weight']    += processor.column_accumulator(ak.to_numpy(weight.weight()[out_sel]))
                output['conv']      += processor.column_accumulator(ak.to_numpy(conv_sel[out_sel]))
                output['weight_np'] += processor.column_accumulator(ak.to_numpy(weight_np_mc[out_sel]))

            if var['name'] == 'central':
                '''
                Don't fill these histograms for the variations
                '''

                # first, make a few super inclusive plots
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

                fill_multiple_np(output['MET'], {'pt':ev.MET.pt, 'phi':ev.MET.phi})
                
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
                
                output['j3'].fill(
                    dataset = dataset,
                    pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
                    eta = ak.flatten(jet[:, 2:3][BL].eta),
                    phi = ak.flatten(jet[:, 2:3][BL].phi),
                    weight = weight_BL
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
    
    cacheName = 'trilep_analysis_%s%s'%(year,era)
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    

    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.5.2_trilep/', year='UL%s%s'%(year,era))
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
        'rare': fileset_all['rare']+fileset_all['triboson'],
        #'ttbar': fileset_all['ttbar1l'],
        #'ttbar': fileset_all['ttbar2l'],
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

    if args.dump:
        fileset.update({
            'topW_old': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.4_trilep/ProjectMetis_TTW*JetsToLNuEWK_5f_NLO_v2_RunIIAutumn18_NANO_v4/*.root'),
        })

    fileset = make_small(fileset, small, n_max=10)

    if verysmall:
        fileset = {'topW_v3': fileset['topW_v3'], 'MuonEG': fileset['MuonEG'], 'ttbar': fileset['ttbar']}

    #fileset = make_small(fileset, small)
    
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


    if args.dump:
        variables = [
            'n_jet',
            'n_b',
            'n_fwd',
            'n_tau',
            'n_ele',
            'n_sfos',
            'charge',
            #'n_track',
            'st',
            'lt',
            'met',
            'mjj_max',
            'delta_eta_jj',
            'lead_lep_pt',
            'lead_lep_eta',
            'sublead_lep_pt',
            'sublead_lep_eta',
            'trail_lep_pt',
            'trail_lep_eta',
            'm3l',
            'close_mass',
            'far_mass',
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
            'trilep',
            'AR',
            'label',
            'conv',
        ]

        for var in variables:
            desired_output.update({var: processor.column_accumulator(np.zeros(shape=(0,)))})

    if local:# and not profile:
        exe_args = {
            'workers': 16,
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
    from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis, ht_axis
    desired_output.update({
        "ST": hist.Hist("Counts", dataset_axis, ht_axis),
        "HT": hist.Hist("Counts", dataset_axis, ht_axis),
        "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, pt_axis),
        "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, pt_axis),
        "node": hist.Hist("Counts", dataset_axis, multiplicity_axis),
        "node0_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score_transform": hist.Hist("Counts", dataset_axis, score_axis),
    })

    for variation in variations:
        ext = variation['ext']
        desired_output.update({
            "node0_score"+ext: hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score_transform"+ext: hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score"+ext: hist.Hist("Counts", dataset_axis, score_axis),
            "node"+ext: hist.Hist("Counts", dataset_axis, multiplicity_axis),
        })

    desired_output.update({
        "norm": hist.Hist("Counts", dataset_axis, one_axis),
    })

    for i in range(1,101):
        desired_output.update({
            "node0_score_transform_pdf_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_pdf_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node_pdf_%s"%i: hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "_pdf_%s"%i: hist.Hist("Counts", dataset_axis, one_axis),
        })

    for i in [0,1,3,5,7,8]:
        desired_output.update({
            "node0_score_transform_scale_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_scale_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node_scale_%s"%i: hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "_scale_%s"%i: hist.Hist("Counts", dataset_axis, one_axis),
        })

    for i in range(4):
        desired_output.update({
            "node0_score_transform_PS_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_PS_%s"%i: hist.Hist("Counts", dataset_axis, score_axis),
            "node_PS_%s"%i: hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "_PS_%s"%i: hist.Hist("Counts", dataset_axis, one_axis),
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
            trilep_analysis(year=year, variations=variations, accumulator=desired_output, evaluate=args.evaluate, training=args.training, dump=args.dump),
            exe,
            exe_args,
            chunksize=250000,  # I guess that's already running into the max events/file
            #chunksize=250000,
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
                df_out.to_hdf('multiclass_input_%s_trilep_v2.h5'%args.year, key='df', format='table', mode='w')
        else:
            print ("Loading DF")
            df_out = pd.read_hdf('multiclass_input_%s_trilep_v2.h5'%args.year)

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


    from Tools.helpers import getCutFlowTable
    lines = [
            'filter',
            'trilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS_dilep',
            'offZ',
            'MET>50',
            'N_jet>2',
            'N_central>1',
            'N_btag>0',
            'N_fwd>0',
            ]
    df = getCutFlowTable(output, processes=list(fileset.keys()), lines=lines, significantFigures=4, signal='topW_v3')



    #from klepto.archives import dir_archive
    #from Tools.samples import * # fileset_2018 #, fileset_2018_small
    #from processor.default_accumulators import *

    #overwrite = True
    #year = 2018
    #small = False
    #
    ## load the config and the cache
    #cfg = loadConfig()
    #
    #cacheName = 'trilep_analysis'
    #if small: cacheName += '_small'
    #cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    #
    #
    #fileset = {
    #    'topW_v3': fileset_2018['topW_v3'],
    #    'TTW': fileset_2018['TTW'],
    #    'TTZ': fileset_2018['TTZ'],
    #    'TTH': fileset_2018['TTH'],
    #    'diboson': fileset_2018['diboson'],
    #    'ttbar': fileset_2018['top2l'], # like 20 events (10x signal)
    #    'DY': fileset_2018['DY'], # like 20 events (10x signal)
    #}

    #fileset = make_small(fileset, small)
    #
    #add_processes_to_output(fileset, desired_output)

    ## add some histograms that we defined in the processor
    ## everything else is taken the default_accumulators.py
    #from processor.default_accumulators import mass_axis, dataset_axis
    #desired_output.update({
    #    "dilep_mass": hist.Hist("Counts", dataset_axis, mass_axis),
    #})

    #histograms = sorted(list(desired_output.keys()))

    #exe_args = {
    #    'workers': 16,
    #    'function_args': {'flatten': False},
    #    "schema": NanoAODSchema,
    #}
    #exe = processor.futures_executor
    #
    #if not overwrite:
    #    cache.load()
    #
    #if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
    #    output = cache.get('simple_output')
    #
    #else:
    #    print ("I'm running now")
    #    
    #    output = processor.run_uproot_job(
    #        fileset,
    #        "Events",
    #        trilep_analysis(year=year, variations=variations, accumulator=desired_output),
    #        exe,
    #        exe_args,
    #        chunksize=250000,
    #    )
    #    
    #    cache['fileset']        = fileset
    #    cache['cfg']            = cfg
    #    cache['histograms']     = histograms
    #    cache['simple_output']  = output
    #    cache.dump()


    #lines = ['entry']
    #lines += [
    #        'filter',
    #        'lepveto',
    #        'trilep',
    #        'p_T(lep0)>25',
    #        'p_T(lep1)>20',
    #        'trigger',
    #        'offZ',
    #        'MET>50',
    #        'N_jet>2',
    #        'N_central>1',
    #        'N_btag>0',
    #        'N_fwd>0',
    #    ]


