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

from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import *
from Tools.ttH_lepton_scalefactors import *
from Tools.selections import Selection
from Tools.nonprompt_weight import NonpromptWeight

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx

class SS_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}):
        self.variations = variations
        self.year = year
        
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
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            ## Generated leptons
            gen_lep = ev.GenL
            leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
            trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ## Muons
        mu_v     = Collections(ev, "Muon", "vetoTTH").get()  # these include all muons, tight and fakeable
        mu_t     = Collections(ev, "Muon", "tightSSTTH").get()
        mu_f     = Collections(ev, "Muon", "fakeableSSTTH").get()
        muon     = ak.concatenate([mu_t, mu_f], axis=1)
        
        ## Electrons
        el_v        = Collections(ev, "Electron", "vetoTTH").get()
        el_t        = Collections(ev, "Electron", "tightSSTTH").get()
        el_f        = Collections(ev, "Electron", "fakeableSSTTH").get()
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
            
        dimuon     = choose(muon, 2)
        dielectron = choose(electron, 2)

        ## Merge electrons and muons - this should work better now in ak1
        dilepton = cross(muon, electron)
        SSlepton = ak.any((dilepton['0'].charge * dilepton['1'].charge)>0, axis=1)

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
        
        baseline = sel.dilep_baseline(cutflow=cutflow, SS=True)
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):

            BL = (baseline & ((ak.num(el_t_p)+ak.num(mu_t_p))==2))  # this is the MC baseline for events with two tight prompt leptons
            BL_incl = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2)) # this is the MC baseline for events with two tight leptons
            np_est_sel_mc = (baseline & \
                ((((ak.num(el_t_p)+ak.num(mu_t_p))==1) & ((ak.num(el_f_np)+ak.num(mu_f_np))==1)) | (((ak.num(el_t_p)+ak.num(mu_t_p))==0) & ((ak.num(el_f_np)+ak.num(mu_f_np))==2)) ))  # no overlap between tight and nonprompt, and veto on additional leptons. this should be enough
            np_obs_sel_mc = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2) & ((ak.num(el_t_np)+ak.num(mu_t_np))>=1) )  # two tight leptons, at least one nonprompt
            np_est_sel_data = (baseline & ~baseline)  # this has to be false

        else:
            BL = (baseline & ((ak.num(el_t)+ak.num(mu_t))==2))
            np_est_sel_mc = (baseline & ~baseline)
            np_obs_sel_mc = (baseline & ~baseline)
            np_est_sel_data = (baseline & (ak.num(el_t)+ak.num(mu_t)==1) & (ak.num(el_f)+ak.num(mu_f)==1) )

        weight_BL = weight.weight()[BL]        

        if False:
            # define the inputs to the NN
            # this is super stupid. there must be a better way.
            NN_inputs = np.stack([
                ak.to_numpy(ak.num(jet[BL])),
                ak.to_numpy(ak.num(tau[BL])),
                ak.to_numpy(ak.num(track[BL])),
                ak.to_numpy(st[BL]),
                ak.to_numpy(ev.MET[BL].pt),
                ak.to_numpy(ak.max(mjf[BL], axis=1)),
                ak.to_numpy(pad_and_flatten(delta_eta[BL])),
                ak.to_numpy(pad_and_flatten(leading_lepton[BL].pt)),
                ak.to_numpy(pad_and_flatten(leading_lepton[BL].eta)),
                ak.to_numpy(pad_and_flatten(trailing_lepton[BL].pt)),
                ak.to_numpy(pad_and_flatten(trailing_lepton[BL].eta)),
                ak.to_numpy(pad_and_flatten(dilepton_mass[BL])),
                ak.to_numpy(pad_and_flatten(dilepton_pt[BL])),
                ak.to_numpy(pad_and_flatten(j_fwd[BL].pt)),
                ak.to_numpy(pad_and_flatten(j_fwd[BL].p)),
                ak.to_numpy(pad_and_flatten(j_fwd[BL].eta)),
                ak.to_numpy(pad_and_flatten(jet[:, 0:1][BL].pt)),
                ak.to_numpy(pad_and_flatten(jet[:, 1:2][BL].pt)),
                ak.to_numpy(pad_and_flatten(jet[:, 0:1][BL].eta)),
                ak.to_numpy(pad_and_flatten(jet[:, 1:2][BL].eta)),
                ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1][BL].pt)),
                ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2][BL].pt)),
                ak.to_numpy(pad_and_flatten(high_score_btag[:, 0:1][BL].eta)),
                ak.to_numpy(pad_and_flatten(high_score_btag[:, 1:2][BL].eta)),
                ak.to_numpy(min_bl_dR[BL]),
                ak.to_numpy(min_mt_lep_met[BL]),
            ])

            NN_inputs = np.moveaxis(NN_inputs, 0, 1)

            model, scaler = load_onnx_model('v8')

            try:
                NN_inputs_scaled = scaler.transform(NN_inputs)

                NN_pred    = predict_onnx(model, NN_inputs_scaled)

                best_score = np.argmax(NN_pred, axis=1)


            except ValueError:
                #print ("Empty NN_inputs")
                NN_pred = np.array([])
                best_score = np.array([])
                NN_inputs_scaled = NN_inputs

            #k.clear_session()

            output['node'].fill(dataset=dataset, multiplicity=best_score, weight=weight_BL)

            output['node0_score_incl'].fill(dataset=dataset, score=NN_pred[:,0] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL)
            output['node0_score'].fill(dataset=dataset, score=NN_pred[best_score==0][:,0] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL[best_score==0])
            output['node1_score'].fill(dataset=dataset, score=NN_pred[best_score==1][:,1] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL[best_score==1])
            output['node2_score'].fill(dataset=dataset, score=NN_pred[best_score==2][:,2] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL[best_score==2])
            output['node3_score'].fill(dataset=dataset, score=NN_pred[best_score==3][:,3] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL[best_score==3])
            output['node4_score'].fill(dataset=dataset, score=NN_pred[best_score==4][:,4] if np.shape(NN_pred)[0]>0 else np.array([]), weight=weight_BL[best_score==4])

            SR_sel_pp = ((best_score==0) & ak.flatten((leading_lepton[BL].pdgId<0)))
            SR_sel_mm = ((best_score==0) & ak.flatten((leading_lepton[BL].pdgId>0)))
            leading_lepton_BL = leading_lepton[BL]

            output['lead_lep_SR_pp'].fill(
                dataset = dataset,
                pt  = ak.to_numpy(ak.flatten(leading_lepton_BL[SR_sel_pp].pt)),
                weight = weight_BL[SR_sel_pp]
            )

            output['lead_lep_SR_mm'].fill(
                dataset = dataset,
                pt  = ak.to_numpy(ak.flatten(leading_lepton_BL[SR_sel_mm].pt)),
                weight = weight_BL[SR_sel_mm]
            )

            del model
            del scaler
            del NN_inputs, NN_inputs_scaled, NN_pred

        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight_BL)
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight_BL)
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight_BL)
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL], weight=weight_BL)
        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight_BL)
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight_BL)
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight_BL)
        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL], weight=weight_BL)
        output['ST'].fill(dataset=dataset, pt=st[BL], weight=weight_BL)
        output['HT'].fill(dataset=dataset, pt=ht[BL], weight=weight_BL)

        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight_BL)
            output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight_BL)
            output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight_BL)
            output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight_BL)
            output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight_BL)
            output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight_BL)
            output['chargeFlip_vs_nonprompt'].fill(dataset=dataset, n1=n_chargeflip[BL], n2=n_nonprompt[BL], n_ele=ak.num(electron)[BL], weight=weight_BL)


        # How to package filling hists like this into a function?
        #output['MET'].fill(
        #    dataset = dataset,
        #    pt  = ev.MET[BL].pt,
        #    phi  = ev.MET[BL].phi,
        #    weight = weight_BL
        #)

        #if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
        #    output['MET'].fill(
        #        dataset="np_est",
        #        pt  = ev.MET[np_est_sel].pt,
        #        phi  = ev.MET[np_est_sel].phi,
        #        weight = weight.weight()[np_est_sel]*(self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')[np_est_sel])
        #    )
        #    output['MET'].fill(
        #        dataset="np_obs",
        #        pt  = ev.MET[np_obs_sel].pt,
        #        phi  = ev.MET[np_obs_sel].phi,
        #        weight = weight.weight()[np_obs_sel]
        #    )
        
        fill_multiple(
            output['MET'],
            datasets=[dataset, "np_est_mc", "np_obs_mc", "np_est_data"],
            arrays={'pt':ev.MET.pt, 'phi':ev.MET.phi},
            selections=[BL, np_est_sel_mc, np_obs_sel_mc, np_est_sel_data],
            weights=[
                weight_BL,
                #weight.weight()[np_est_sel_mc]*(self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')[np_est_sel_mc]),
                #weight.weight()[np_obs_sel_mc],
                weight.weight()[np_est_sel_mc]*(self.nonpromptWeight.get(el_f_np[np_est_sel_mc], mu_f_np[np_est_sel_mc], meas='TT')),
                weight.weight()[np_obs_sel_mc],
                weight.weight()[np_est_sel_data]*(self.nonpromptWeight.get(el_f_np, mu_f_np, meas='TT')[np_est_sel_data]), ## will need to be replaced
            ],
        )

        output['N_ele'].fill(dataset="np_est_mc", multiplicity=ak.num(electron)[np_est_sel_mc], weight=weight.weight()[np_est_sel_mc]*(self.nonpromptWeight.get(el_f_np[np_est_sel_mc], mu_f_np[np_est_sel_mc], meas='TT')) )
        output['N_ele'].fill(dataset="np_obs_mc", multiplicity=ak.num(electron)[np_obs_sel_mc], weight=weight.weight()[np_obs_sel_mc] )

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
        
        output['lead_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
            weight = weight_BL
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
        
        
        output['fwd_jet'].fill(
            dataset = dataset,
            pt  = ak.flatten(j_fwd[BL].pt),
            eta = ak.flatten(j_fwd[BL].eta),
            phi = ak.flatten(j_fwd[BL].phi),
            weight = weight_BL
        )
            
        output['high_p_fwd_p'].fill(dataset=dataset, p = ak.flatten(j_fwd[BL].p), weight = weight_BL)
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import get_babies
    from processor.default_accumulators import *

    overwrite = True
    small = True
    save = True

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'SS_analysis'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    year = 2018

    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.2_dilep/', year='UL2018')
    #fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.3/', year=2018)
    
    fileset = {
        ##'topW_v3': fileset_all['topW_NLO'],
        #'topW_v3': fileset_all['topW_v3'],
        ##'topW_EFT_mix': fileset_all['topW_EFT'],
        #'topW_EFT_cp8': fileset_all['topW_EFT_cp8'],
        #'topW_EFT_mix': fileset_all['topW_EFT_mix'],
        #'TTW': fileset_all['TTW'],
        #'TTZ': fileset_all['TTZ'],
        #'TTH': fileset_all['TTH'],
        #'diboson': fileset_all['diboson'],
        #'triboson': fileset_all['triboson'],
        ##'wpwp': fileset_all['wpwp'],
        #'TTTT': fileset_all['TTTT'],
        'ttbar': fileset_all['ttbar1l'],
        #'ttbar': fileset_all['ttbar'],
        ##'MuonEG': fileset_all['MuonEG_Run2018'],
        ##'DoubleMuon': fileset_all['DoubleMuon_Run2018'],
        ##'EGamma': fileset_all['EGamma_Run2018'],
        #'MuonEG': fileset_all['MuonEG'],
        #'DoubleMuon': fileset_all['DoubleMuon'],
        #'EGamma': fileset_all['EGamma'],
        ##'topW_full_EFT': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_UL17_v7/*.root'),
        ##'topW_NLO': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_SMEFTatNLO_weight_RunIIAutumn18_NANO_UL17_v7/*.root'),
    }
    
    fileset = make_small(fileset, small, n_max=10)
    #fileset = make_small(fileset, small)
    
    add_processes_to_output(fileset, desired_output)

    exe_args = {
        'workers': 12,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
    }
    exe = processor.futures_executor

    # add some histograms that we defined in the processor
    # everything else is taken the default_accumulators.py
    from processor.default_accumulators import multiplicity_axis, dataset_axis, score_axis, pt_axis
    desired_output.update({
        "ST": hist.Hist("Counts", dataset_axis, pt_axis),
        "HT": hist.Hist("Counts", dataset_axis, pt_axis),
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
            SS_analysis(year=year, variations=variations, accumulator=desired_output),
            exe,
            exe_args,
            chunksize=100000,
            #chunksize=250000,
        )
        
        if save:
            cache['fileset']        = fileset
            cache['cfg']            = cfg
            cache['histograms']     = histograms
            cache['simple_output']  = output
            cache.dump()

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
        'ttbar': r'$t\bar{t}$',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'ttbar': '#1982C4',
    }

    makePlot(output, 'node', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars('$TWHOME/dump/ML_node'),
        )


    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars('$TWHOME/dump/ML_node0_score'),
        )

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score', shape=True, ymax=0.35,
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'ttbar', 'TTH', 'TTZ'],
         save=os.path.expandvars('$TWHOME/dump/ML_node0_score_shape'),
        )

