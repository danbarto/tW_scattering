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

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx


class SS_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False):
        self.variations = variations
        self.year = year
        self.evaluate = evaluate
        self.training = training
        self.dump = dump
        
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)

        self.nonpromptWeight = NonpromptWeight(year=year)
        self.chargeflipWeight = charge_flip(year=year)
        
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
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            ## Generated leptons
            gen_lep = ev.GenL
            leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
            trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ## Get the leptons. This has changed a couple of times now, but we are using fakeable objects as baseline leptons.
        ## Muons
        mu_v     = Collections(ev, "Muon", "vetoTTH", year=year).get()  # these include all muons, tight and fakeable
        mu_t     = Collections(ev, "Muon", "tightSSTTH", year=year).get()
        mu_f     = Collections(ev, "Muon", "fakeableSSTTH", year=year).get()
        muon     = ak.concatenate([mu_t, mu_f], axis=1)
        
        ## Electrons
        el_v        = Collections(ev, "Electron", "vetoTTH", year=year).get()
        el_t        = Collections(ev, "Electron", "tightSSTTH", year=year).get()
        el_f        = Collections(ev, "Electron", "fakeableSSTTH", year=year).get()
        electron    = ak.concatenate([el_t, el_f], axis=1)
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            el_t_p  = prompt(el_t)
            el_t_np = nonprompt(el_t)
            el_f_p  = prompt(el_f)
            el_f_np = nonprompt(el_f)
            mu_t_p  = prompt(mu_t)
            mu_t_np = nonprompt(mu_t)
            mu_f_p  = prompt(mu_f)
            mu_f_np = nonprompt(mu_f)

            is_flipped = ( (el_t_p.matched_gen.pdgId*(-1) == el_t_p.pdgId) & (abs(el_t_p.pdgId) == 11) )
            el_t_p_cc  = el_t_p[~is_flipped]  # this is tight, prompt, and charge consistent
            el_t_p_cf  = el_t_p[is_flipped]  # this is tight, prompt, and charge flipped

        ## Merge electrons and muons. These are fakeable leptons now
        lepton   = ak.concatenate([muon, electron], axis=1)
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]

        dilepton_mass = (leading_lepton+trailing_lepton).mass
        dilepton_pt = (leading_lepton+trailing_lepton).pt
        dilepton_dR = delta_r(leading_lepton, trailing_lepton)
        
        lepton_pdgId_pt_ordered = ak.fill_none(ak.pad_none(lepton[ak.argsort(lepton.pt, ascending=False)].pdgId, 2, clip=True), 0)
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            n_nonprompt = getNonPromptFromFlavour(electron) + getNonPromptFromFlavour(muon)
            n_chargeflip = getChargeFlips(electron, ev.GenPart) + getChargeFlips(muon, ev.GenPart)
            gp = ev.GenPart
            gp_e = gp[((abs(gp.pdgId)==11)&(gp.status==1)&((gp.statusFlags&(1<<0))==1)&(gp.statusFlags&(1<<8)==256))]
            gp_m = gp[((abs(gp.pdgId)==13)&(gp.status==1)&((gp.statusFlags&(1<<0))==1)&(gp.statusFlags&(1<<8)==256))]
            n_gen_lep = ak.num(gp_e) + ak.num(gp_m)
        else:
            n_gen_lep = np.zeros(len(ev))

        LL = (n_gen_lep > 2)  # this is the classifier for LL events (should mainly be ttZ/tZ/WZ...)

        mt_lep_met = mt(lepton.pt, lepton.phi, ev.MET.pt, ev.MET.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)

        ## Tau and other stuff
        tau       = getTaus(ev)
        track     = getIsoTracks(ev)

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
        
        high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

        bl          = cross(lepton, high_score_btag)
        bl_dR       = delta_r(bl['0'], bl['1'])
        min_bl_dR   = ak.min(bl_dR, axis=1)

        ## forward jets
        j_fwd = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator

        # try to get either the most forward light jet, or if there's more than one with eta>1.7, the highest pt one
        most_fwd = light[ak.argsort(abs(light.eta))][:,0:1]
        #most_fwd = light[ak.singletons(ak.argmax(abs(light.eta)))]
        best_fwd = ak.concatenate([j_fwd, most_fwd], axis=1)[:,0:1]
        
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
        
        # define the weight
        weight = Weights( len(ev) )

        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            
            # PU weight
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        

        cutflow     = Cutflow(output, ev, weight=weight)

        # slightly restructured
        # calculate everything from loose, require two tights on top
        # since n_tight == n_loose == 2, the tight and loose leptons are the same in the end

        # in this selection we'll get events with exactly two fakeable+tight and two loose leptons.
        sel = Selection(
            dataset = dataset,
            events = ev,
            year = self.year,
            ele = electron,
            ele_veto = el_v,
            mu = muon,
            mu_veto = mu_v,
            jet_all = jet,
            jet_central = central,
            jet_btag = btag,
            jet_fwd = fwd,
            jet_light = light,
            met = ev.MET,
        )
        
        baseline = sel.dilep_baseline(cutflow=cutflow, SS=True, omit=['N_fwd>0'])
        baseline_OS = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0'])  # this is for charge flip estimation
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):

            BL = (baseline & ((ak.num(el_t_p_cc)+ak.num(mu_t_p))==2))  # this is the MC baseline for events with two tight prompt leptons
            BL_incl = (baseline) # this is the MC baseline for events with two fakeable+tight leptons
            np_est_sel_mc = (baseline & \
                ((((ak.num(el_t_p_cc)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p_cc)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
            np_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
            np_est_sel_data = (baseline & ~baseline)  # this has to be false

            cf_est_sel_mc = (baseline_OS & ((ak.num(el_t_p)+ak.num(mu_t_p))==2))
            cf_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_p_cf))>=1) )  # two tight leptons, at least one electron charge flip
            cf_est_sel_data = (baseline & ~baseline)  # this has to be false

            weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')
            weight_cf_mc = self.chargeflipWeight.flip_weight(el_t_p)

        else:
            BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2))

            np_est_sel_mc = (baseline & ~baseline)
            np_obs_sel_mc = (baseline & ~baseline)
            np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)==1) & (ak.num(el_f)+ak.num(mu_f)==1) )

            cf_est_sel_mc = (baseline & ~baseline)
            cf_obs_sel_mc = (baseline & ~baseline)
            cf_est_sel_data = (baseline_OS & ((ak.num(el_t)+ak.num(mu_t))==2) )

            weight_np_mc = np.zeros(len(ev))
            weight_cf_mc = np.zeros(len(ev))

        weight_BL = weight.weight()[BL]  # this is just a shortened weight list for the two prompt selection
        weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data')
        weight_cf_data = self.chargeflipWeight.flip_weight(el_t)

        out_sel = (BL | np_est_sel_mc | cf_est_sel_mc)

        #def fill_multiple_np(hist, arrays):
        #    fill_multiple(
        #        hist,
        #        datasets=[dataset, "np_est_mc", "np_obs_mc", "np_est_data", "cf_est_mc", "cf_obs_mc", "cf_est_data"],
        #        arrays=arrays,
        #        selections=[BL, np_est_sel_mc, np_obs_sel_mc, np_est_sel_data, cf_est_sel_mc, cf_obs_sel_mc, cf_est_sel_data],
        #        weights=[
        #            weight_BL,
        #            weight.weight()[np_est_sel_mc]*weight_np_mc[np_est_sel_mc],
        #            weight.weight()[np_obs_sel_mc],
        #            weight.weight()[np_est_sel_data]*weight_np_data[np_est_sel_data],
        #            weight.weight()[cf_est_sel_mc]*weight_cf_mc[cf_est_sel_mc],
        #            weight.weight()[cf_obs_sel_mc],
        #            weight.weight()[cf_est_sel_data]*weight_cf_data[cf_est_sel_data],
        #        ],
        #    )
        dummy = (np.ones(len(ev))==1)
        def fill_multiple_np(hist, arrays, add_sel=dummy):
            #reg_sel = [BL, np_est_sel_mc, np_obs_sel_mc, np_est_sel_data, cf_est_sel_mc, cf_obs_sel_mc, cf_est_sel_data],
            #print ('len', len(reg_sel[0]))
            #print ('sel', reg_sel[0])
            reg_sel = [BL&add_sel, np_est_sel_mc&add_sel, np_obs_sel_mc&add_sel, np_est_sel_data&add_sel, cf_est_sel_mc&add_sel, cf_obs_sel_mc&add_sel, cf_est_sel_data&add_sel],
            fill_multiple(
                hist,
                datasets=[dataset, "np_est_mc", "np_obs_mc", "np_est_data", "cf_est_mc", "cf_obs_mc", "cf_est_data"],
                arrays=arrays,
                selections=reg_sel[0],  # no idea where the additional dimension is coming from...
                weights=[
                    weight.weight()[reg_sel[0][0]],
                    #weight_BL,
                    weight.weight()[reg_sel[0][1]]*weight_np_mc[reg_sel[0][1]],
                    weight.weight()[reg_sel[0][2]],
                    weight.weight()[reg_sel[0][3]]*weight_np_data[reg_sel[0][3]],
                    weight.weight()[reg_sel[0][4]]*weight_cf_mc[reg_sel[0][4]],
                    weight.weight()[reg_sel[0][5]],
                    weight.weight()[reg_sel[0][6]]*weight_cf_data[reg_sel[0][6]],
                ],
            )

        if self.evaluate or self.dump:
            # define the inputs to the NN
            # this is super stupid. there must be a better way.
            # used a np.stack which is ok performance wise. pandas data frame seems to be slow and memory inefficient
            #FIXME no n_b, n_fwd back in v13/v14 of the DNN
            NN_inputs_d = {
                'n_jet':            ak.to_numpy(ak.num(jet)),
                'n_b':              ak.to_numpy(ak.num(btag)),
                'n_fwd':            ak.to_numpy(ak.num(fwd)),
                'n_tau':            ak.to_numpy(ak.num(tau)),
                'n_track':          ak.to_numpy(ak.num(track)),
                'st':               ak.to_numpy(st),
                'met':              ak.to_numpy(ev.MET.pt),
                'mjj_max':          ak.to_numpy(ak.fill_none(ak.max(mjf, axis=1),0)),
                'delta_eta_jj':     ak.to_numpy(pad_and_flatten(delta_eta)),
                'lead_lep_pt':      ak.to_numpy(pad_and_flatten(leading_lepton.pt)),
                'lead_lep_eta':     ak.to_numpy(pad_and_flatten(leading_lepton.eta)),
                'sublead_lep_pt':   ak.to_numpy(pad_and_flatten(trailing_lepton.pt)),
                'sublead_lep_eta':  ak.to_numpy(pad_and_flatten(trailing_lepton.eta)),
                'dilepton_mass':    ak.to_numpy(pad_and_flatten(dilepton_mass)),
                'dilepton_pt':      ak.to_numpy(pad_and_flatten(dilepton_pt)),
                'fwd_jet_pt':       ak.to_numpy(pad_and_flatten(best_fwd.pt)),
                'fwd_jet_p':        ak.to_numpy(pad_and_flatten(best_fwd.p)),
                'fwd_jet_eta':      ak.to_numpy(pad_and_flatten(best_fwd.eta)),
                'lead_jet_pt':      ak.to_numpy(pad_and_flatten(jet[:, 0:1].pt)),
                'sublead_jet_pt':   ak.to_numpy(pad_and_flatten(jet[:, 1:2].pt)),
                'lead_jet_eta':     ak.to_numpy(pad_and_flatten(jet[:, 0:1].eta)),
                'sublead_jet_eta':  ak.to_numpy(pad_and_flatten(jet[:, 1:2].eta)),
                'lead_btag_pt':     ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1].pt)),
                'sublead_btag_pt':  ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2].pt)),
                'lead_btag_eta':    ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1].eta)),
                'sublead_btag_eta': ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2].eta)),
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

                model, scaler = load_onnx_model(self.training)

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

                ##k.clear_session()

                #FIXME below needs to be fixed again with changed NN evaluation. Should work now
                #output['node'].fill(dataset=dataset, multiplicity=best_score[BL] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL)
                #output['node'].fill(dataset=dataset, multiplicity=best_score[BL], weight=weight_BL)

                fill_multiple_np(output['node'], {'multiplicity':best_score})
                fill_multiple_np(output['node0_score_incl'], {'score':NN_pred[:,0]})
                #output['node0_score_incl'].fill(dataset=dataset, score=NN_pred[:,0][BL], weight=weight_BL)
                
                fill_multiple_np(output['node0_score'], {'score':NN_pred[:,0]}, add_sel=(best_score==0))
                fill_multiple_np(output['node1_score'], {'score':NN_pred[:,1]}, add_sel=(best_score==1))
                fill_multiple_np(output['node2_score'], {'score':NN_pred[:,2]}, add_sel=(best_score==2))
                fill_multiple_np(output['node3_score'], {'score':NN_pred[:,3]}, add_sel=(best_score==3))
                fill_multiple_np(output['node4_score'], {'score':NN_pred[:,4]}, add_sel=(best_score==4))
                #output['node0_score'].fill(dataset=dataset, score=NN_pred[((best_score==0)&BL)][:,0], weight=weight.weight()[((best_score==0)&BL)])
                #output['node1_score'].fill(dataset=dataset, score=NN_pred[((best_score==1)&BL)][:,1], weight=weight.weight()[((best_score==1)&BL)])
                #output['node2_score'].fill(dataset=dataset, score=NN_pred[((best_score==2)&BL)][:,2], weight=weight.weight()[((best_score==2)&BL)])
                #output['node3_score'].fill(dataset=dataset, score=NN_pred[((best_score==3)&BL)][:,3], weight=weight.weight()[((best_score==3)&BL)])
                #output['node4_score'].fill(dataset=dataset, score=NN_pred[((best_score==4)&BL)][:,4], weight=weight.weight()[((best_score==4)&BL)])

                #SR_sel_pp = ((best_score==0) & ak.flatten((leading_lepton[BL].pdgId<0)))
                #SR_sel_mm = ((best_score==0) & ak.flatten((leading_lepton[BL].pdgId>0)))
                #leading_lepton_BL = leading_lepton[BL]

                #output['lead_lep_SR_pp'].fill(
                #    dataset = dataset,
                #    pt  = ak.to_numpy(ak.flatten(leading_lepton_BL[SR_sel_pp].pt)),
                #    weight = weight_BL[SR_sel_pp]
                #)

                #output['lead_lep_SR_mm'].fill(
                #    dataset = dataset,
                #    pt  = ak.to_numpy(ak.flatten(leading_lepton_BL[SR_sel_mm].pt)),
                #    weight = weight_BL[SR_sel_mm]
                #)

                del model
                del scaler
                del NN_inputs, NN_inputs_scaled, NN_pred

        labels = {'topW_v3': 0, 'TTW':1, 'TTZ': 2, 'TTH': 3, 'ttbar': 4, 'rare':5, 'diboson':6}  # these should be all?
        if dataset in labels:
            label_mult = labels[dataset]
        else:
            label_mult = 7  # data or anything else

        if self.dump:
            output['label']     += processor.column_accumulator(np.ones(len(ev[out_sel])) * label_mult)
            output['SS']        += processor.column_accumulator(ak.to_numpy(BL[out_sel]))
            output['OS']        += processor.column_accumulator(ak.to_numpy(cf_est_sel_mc[out_sel]))
            output['AR']        += processor.column_accumulator(ak.to_numpy(np_est_sel_mc[out_sel]))
            output['LL']        += processor.column_accumulator(ak.to_numpy(LL[out_sel]))
            output['weight']    += processor.column_accumulator(ak.to_numpy(weight.weight()[out_sel]))
            output['weight_np'] += processor.column_accumulator(ak.to_numpy(weight_np_mc[out_sel]))
            output['weight_cf'] += processor.column_accumulator(ak.to_numpy(weight_cf_mc[out_sel]))

        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight_BL)
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight_BL)
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL], weight=weight_BL)
        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight_BL)
        fill_multiple_np(output['N_ele'], {'multiplicity':ak.num(electron)})
        fill_multiple_np(output['N_mu'],  {'multiplicity':ak.num(muon)})
        fill_multiple_np(output['N_fwd'], {'multiplicity':ak.num(fwd)})
        output['ST'].fill(dataset=dataset, ht=st[BL], weight=weight_BL)
        output['HT'].fill(dataset=dataset, ht=ht[BL], weight=weight_BL)

        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight_BL)
            output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight_BL)
            output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight_BL)
            output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight_BL)
            output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight_BL)
            output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight_BL)
            output['chargeFlip_vs_nonprompt'].fill(dataset=dataset, n1=n_chargeflip[BL], n2=n_nonprompt[BL], n_ele=ak.num(electron)[BL], weight=weight_BL)

        fill_multiple_np(output['MET'], {'pt':ev.MET.pt, 'phi':ev.MET.phi})

        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
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
        
        fill_multiple_np(
            output['lead_lep'],
            {
                'pt':  pad_and_flatten(leading_lepton.pt),
                'eta': pad_and_flatten(leading_lepton.eta),
                'phi': pad_and_flatten(leading_lepton.phi),
            },
        )

        output['trail_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
            weight = weight_BL
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
            
        output['high_p_fwd_p'].fill(dataset=dataset, p = ak.flatten(best_fwd[BL].p), weight = weight_BL)
        
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
    argParser.add_argument('--training', action='store', default='v8', help="Which training to use?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = not args.keep
    small       = args.small
    verysmall   = args.verysmall
    year        = int(args.year)
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'SS_analysis_%s'%year
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    

    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.3_dilep/', year='UL%s'%year)
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
        ##'ttbar': fileset_all['ttbar1l'],
        ##'ttbar': fileset_all['ttbar2l'],
        'ttbar': fileset_all['top'],
        'MuonEG': fileset_all['MuonEG'],
        'DoubleMuon': fileset_all['DoubleMuon'],
        'EGamma': fileset_all['EGamma'],
        ####'topW_full_EFT': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_UL17_v7/*.root'),
        ####'topW_NLO': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_SMEFTatNLO_weight_RunIIAutumn18_NANO_UL17_v7/*.root'),
    }
    
    fileset = make_small(fileset, small, n_max=10)

    if verysmall:
        fileset = {'topW_v3': fileset['topW_v3'], 'MuonEG': fileset['MuonEG']}

    #fileset = make_small(fileset, small)
    
    add_processes_to_output(fileset, desired_output)

    if args.dump:
        variables = [
            'n_jet',
            'n_b',
            'n_fwd',
            'n_tau',
            'n_track',
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
            'label',
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
        "node0_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node4_score": hist.Hist("Counts", dataset_axis, score_axis),
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
            SS_analysis(year=year, variations=variations, accumulator=desired_output, evaluate=args.evaluate, training=args.training, dump=args.dump),
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
                df_out.to_hdf('multiclass_input_%s_v2.h5'%year, key='df', format='table', mode='w')
        else:
            print ("Loading DF")
            df_out = pd.read_hdf('multiclass_input_%s_v2.h5'%year)

    
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
        'rare': '#525B76',
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
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3'],
         save=os.path.expandvars('$TWHOME/dump/ML_node'),
        )

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars('$TWHOME/dump/ML_node0_score'),
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

