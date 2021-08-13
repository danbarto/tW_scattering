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

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import btag_scalefactor
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx


class trilep_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False):
        self.variations = variations
        self.year = year
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
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
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
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            # tight electrons
            el_t_p  = prompt(el_t)
            el_t_np = nonprompt(el_t)
            # fakeable electrons
            el_f_p  = prompt(el_f)
            el_f_np = nonprompt(el_f)
            # loose/veto electrons
            el_v_p  = prompt(el_v)

            mu_t_p  = prompt(mu_t)
            mu_t_np = nonprompt(mu_t)

            mu_f_p  = prompt(mu_f)
            mu_f_np = nonprompt(mu_f)

            mu_v_p  = prompt(mu_v)

        ## Merge electrons and muons. These are fakeable leptons now
        lepton   = ak.concatenate([muon, electron], axis=1)
        lead_leptons = lepton[ak.argsort(lepton.p4.pt)][:,:3]
        leading_lepton_idx = ak.singletons(ak.argmax(lepton.p4.pt, axis=1))
        leading_lepton = lepton[leading_lepton_idx]
        trailing_lepton_idx = ak.singletons(ak.argmin(lepton.p4.pt, axis=1))
        trailing_lepton = lepton[trailing_lepton_idx]

        dilepton_mass = (leading_lepton.p4 + trailing_lepton.p4).mass
        dilepton_pt = (leading_lepton.p4 + trailing_lepton.p4).pt
        #dilepton_dR = delta_r(leading_lepton, trailing_lepton)
        dilepton_dR = leading_lepton.p4.delta_r(trailing_lepton.p4)
        
        lepton_pdgId_pt_ordered = ak.fill_none(ak.pad_none(lepton[ak.argsort(lepton.p4.pt, ascending=False)].pdgId, 2, clip=True), 0)
        
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

        mt_lep_met = mt(lepton.p4.pt, lepton.p4.phi, ev.MET.pt, ev.MET.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)

        ## Tau and other stuff
        tau       = getTaus(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] 
        tau       = tau[~match(tau, electron, deltaRCut=0.4)]

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


        #################
        ### Variables ###
        #################
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
        
        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))  # FIXME this needs to be evaluated for loose, too
        
        cutflow     = Cutflow(output, ev, weight=weight)

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
            met = ev.MET,
        )

        baseline = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0'])
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            # The baseline selection is at least three loose leptons, at least two tight with SS.
            # For the way we estimate the background, I need to ask for the tight leptons to be prompt
            # Can I allow for loose fakes?

            BL = (baseline & ((ak.num(el_t_p)+ak.num(mu_t_p))>=3) & ((ak.num(el_v)+ak.num(mu_v))>=3) )  # 
            BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))>=3) & ((ak.num(el_v)+ak.num(mu_v))>=3) )

            np_est_sel_mc = (baseline & \
                ((((ak.num(el_t_p)+ak.num(mu_t_p))>=1) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=2)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=3)) | (((ak.num(el_t_p)+ak.num(mu_t_p))>=2) & ((ak.num(el_f_np)+ak.num(mu_f_np))>=1)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
            np_obs_sel_mc = (baseline & ( (ak.num(el_t)+ak.num(mu_t))>=3) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
            np_est_sel_data = (baseline & ~baseline)  # this has to be false

            weight_np_mc = self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')

        else:
            BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))>=2))

            BL_incl = BL

            np_est_sel_mc = (baseline & ~baseline)
            np_obs_sel_mc = (baseline & ~baseline)
            np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)>=1) & (ak.num(el_f)+ak.num(mu_f)>=1) )

            weight_np_mc = np.zeros(len(ev))

        weight_BL = weight.weight()[BL]  # this is just a shortened weight list for the two prompt selection
        weight_np_data = self.nonpromptWeight.get(el_f, mu_f, meas='data')

        out_sel = (BL | np_est_sel_mc)
        
        dummy = (np.ones(len(ev))==1)
        def fill_multiple_np(hist, arrays, add_sel=dummy):
            reg_sel = [BL&add_sel, BL_incl&add_sel, np_est_sel_mc&add_sel, np_obs_sel_mc&add_sel, np_est_sel_data&add_sel],
            fill_multiple(
                hist,
                datasets=[
                    dataset, # only prompt contribution from process
                    dataset+"_incl", # everything from process (inclusive MC truth)
                    "np_est_mc", # MC based NP estimate
                    "np_obs_mc", # MC based NP observation
                    "np_est_data",
                ],
                arrays=arrays,
                selections=reg_sel[0],  # no idea where the additional dimension is coming from...
                weights=[
                    weight.weight()[reg_sel[0][0]],
                    weight.weight()[reg_sel[0][1]],
                    weight.weight()[reg_sel[0][2]]*weight_np_mc[reg_sel[0][2]],
                    weight.weight()[reg_sel[0][3]],
                    weight.weight()[reg_sel[0][4]]*weight_np_data[reg_sel[0][4]],
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
                #'n_track':          ak.to_numpy(ak.num(track)),
                'st':               ak.to_numpy(st),
                'met':              ak.to_numpy(ev.MET.pt),
                'mjj_max':          ak.to_numpy(ak.fill_none(ak.max(mjf, axis=1),0)),
                'delta_eta_jj':     ak.to_numpy(pad_and_flatten(delta_eta)),
                'lead_lep_pt':      ak.to_numpy(pad_and_flatten(lead_leptons[:,0:1].p4.pt)),
                'lead_lep_eta':     ak.to_numpy(pad_and_flatten(lead_leptons[:,0:1].p4.eta)),
                'sublead_lep_pt':   ak.to_numpy(pad_and_flatten(lead_leptons[:,1:2].p4.pt)),
                'sublead_lep_eta':  ak.to_numpy(pad_and_flatten(lead_leptons[:,1:2].p4.eta)),
                'trail_lep_pt':     ak.to_numpy(pad_and_flatten(lead_leptons[:,2:3].p4.pt)),
                'trail_lep_eta':    ak.to_numpy(pad_and_flatten(lead_leptons[:,2:3].p4.eta)),
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
            output['weight_np'] += processor.column_accumulator(ak.to_numpy(weight_np_mc[out_sel]))

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
    year        = int(args.year)
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'trilep_analysis_%s'%year
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    

    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.3_trilep/', year='UL%s'%year)
    #fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/', year=2018)
    
    fileset = {
        'topW_v3': fileset_all['topW_NLO'],
        'topW_old': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.4_trilep/ProjectMetis_TTW*JetsToLNuEWK_5f_NLO_v2_RunIIAutumn18_NANO_v4/*.root'),
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
        #'ttbar': fileset_all['ttbar2l'],
        'ttbar': fileset_all['top'],
        'XG': fileset_all['XG'],
        #'MuonEG': fileset_all['MuonEG'],
        #'DoubleMuon': fileset_all['DoubleMuon'],
        #'EGamma': fileset_all['EGamma'],
        ####'topW_full_EFT': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_UL17_v7/*.root'),
        ####'topW_NLO': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_SMEFTatNLO_weight_RunIIAutumn18_NANO_UL17_v7/*.root'),
    }
    
    fileset = make_small(fileset, small, n_max=10)

    if verysmall:
        fileset = {'topW_v3': fileset['topW_v3'], 'MuonEG': fileset['MuonEG'], 'ttbar': fileset['ttbar']}

    #fileset = make_small(fileset, small)
    
    add_processes_to_output(fileset, desired_output)

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
            'trail_lep_pt',
            'trail_lep_eta',
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
        "node4_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
        "node0_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node1_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node2_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node3_score": hist.Hist("Counts", dataset_axis, score_axis),
        "node4_score": hist.Hist("Counts", dataset_axis, score_axis),
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
                df_out.to_hdf('multiclass_input_%s_trilep_v2.h5'%year, key='df', format='table', mode='w')
        else:
            print ("Loading DF")
            df_out = pd.read_hdf('multiclass_input_%s_trilep_v2.h5'%year)

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


