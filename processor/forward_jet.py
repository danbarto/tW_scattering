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
from Tools.cutflow import *
from Tools.config_helpers import loadConfig, make_small, data_pattern #, zip_run_lumi_event
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.trigger_scalefactors import *
from Tools.ttH_lepton_scalefactors import *
from Tools.selections import Selection, get_pt
from Tools.helpers import mt

import warnings
warnings.filterwarnings("ignore")


class forwardJetAnalyzer(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False, era=None):
        self.variations = variations
        self.year = year
        self.era = era
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)
        
        self.triggerSF = triggerSF(year=year)
        
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
        
        ## Jets
        jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var='pt_nom')
        jet       = jet[ak.argsort(jet.p4.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
        jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        
        central   = jet[(abs(jet.eta)<2.4)]
        btag      = getBTagsDeepFlavB(jet, era=era, year=self.year) # should study working point for DeepJet
        light     = getBTagsDeepFlavB(jet, era=era, year=self.year, invert=True)
        light_central = light[(abs(light.eta)<2.5)]
        fwd       = getFwdJet(light)
        fwd_noPU  = getFwdJet(light, puId=False)
        
        ## forward jets
        high_p_fwd   = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator
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
        
        ## MET -> can switch to puppi MET
        #met_pt  = ev.MET.T1_pt
        #met_phi = ev.MET.phi
        met = getMET(ev, pt_var='pt_nom')
        met_pt = met.pt
        met_phi = met.phi

        ## other variables
        ht = ak.sum(jet.p4.pt, axis=1)
        st = met_pt + ht + ak.sum(muon.p4.pt, axis=1) + ak.sum(electron.p4.pt, axis=1)
        ht_central = ak.sum(central.p4.pt, axis=1)
        
        tau       = getTaus(ev)
        track     = getIsoTracks(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] # remove taus that overlap with muons
        tau       = tau[~match(tau, electron, deltaRCut=0.4)] # remove taus that overlap with electrons
        
        bl          = cross(lepton, high_score_btag)
        bl_dR       = delta_r(bl['0'], bl['1'])
        min_bl_dR   = ak.min(bl_dR, axis=1)

        #mt_lep_met = mt(lepton.p4.pt, lepton.phi, ev.MET.T1_pt, ev.MET.phi)
        mt_lep_met = mt(lepton.p4.pt, lepton.phi, met_pt, met_phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)
        
        if not re.search(data_pattern, dataset):
            gen = ev.GenPart
            gen_photon = gen[gen.pdgId==22]
            external_conversions = external_conversion(lepton, gen_photon)
            conversion_veto = ((ak.num(external_conversions))==0)
            conversion_req = ((ak.num(external_conversions))>0)
               
        # define the weight
        weight = Weights( len(ev) )
        
        if not re.search(data_pattern, dataset):
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light_central))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
            
            weight.add("trigger", self.triggerSF.get(electron, muon))
            
        
        cutflow     = Cutflow(output, ev, weight=weight)

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
            #met = ev.MET,
        )
#        BL = sel.dilep_baseline(cutflow=cutflow, SS=False)
        BL = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_central>0', 'N_fwd>0'])
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
            
        BL_diele = ( BL & (ak.num(electron, axis=1)==2) )
        BL_1ele = ( BL & (ak.num(electron, axis=1)==1) )
        BL_0ele = ( BL & (ak.num(electron, axis=1)==0) )
        
        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight.weight()[BL])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight.weight()[BL])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight.weight()[BL])
        output['N_tau'].fill(dataset=dataset, multiplicity=ak.num(tau)[BL], weight=weight.weight()[BL])
        
        output['N_track'].fill(dataset=dataset, multiplicity=ak.num(track)[BL], weight=weight.weight()[BL])
        
        
#        BL_minusNb = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_btag>0'])
        BL_minusNb = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_btag=0','N_central>0', 'N_fwd>0'])
        
        if dataset=='XG':
            BL_minusNb = (BL_minusNb & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusNb = (BL_minusNb & conversion_veto)
            
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])

        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight.weight()[BL])
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])

#        BL_minusFwd = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0'])
        BL_minusFwd = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0', 'N_central>0'])           
        if dataset=='XG':
            BL_minusFwd = (BL_minusFwd & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusFwd = (BL_minusFwd & conversion_veto)

        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL_minusFwd], weight=weight.weight()[BL_minusFwd])
        
        output['dilep_pt'].fill(dataset=dataset, pt=ak.flatten(dilepton_pt[BL]), weight=weight.weight()[BL])
        
        output['dilep_mass'].fill(dataset=dataset, mass=ak.flatten(dilepton_mass[BL]), weight=weight.weight()[BL])
        
#        BL_mjf = BL & (ak.num(fwd)>0)
#        BL_bldr = BL
        BL_mjf = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_central>0'])
        BL_bldr = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0'])
        
        output['mjf_max'].fill(dataset=dataset, mass=mjf_max[BL_mjf], weight=weight.weight()[BL_mjf])
        output['deltaEta'].fill(dataset=dataset, eta=ak.flatten(deltaEta[BL_mjf]), weight=weight.weight()[BL_mjf])
        output['min_bl_dR'].fill(dataset=dataset, eta=min_bl_dR[BL_bldr], weight=weight.weight()[BL_bldr])
        output['min_mt_lep_met'].fill(dataset=dataset, pt=min_mt_lep_met[BL], weight=weight.weight()[BL])
        
        output['leading_jet_pt'].fill(dataset=dataset, pt=ak.flatten(jet[:, 0:1][BL].p4.pt), weight=weight.weight()[BL])
        output['subleading_jet_pt'].fill(dataset=dataset, pt=ak.flatten(jet[:, 1:2][BL].p4.pt), weight=weight.weight()[BL])
        output['leading_jet_eta'].fill(dataset=dataset, eta=ak.flatten(jet[:, 0:1][BL].p4.eta), weight=weight.weight()[BL])
        output['subleading_jet_eta'].fill(dataset=dataset, eta=ak.flatten(jet[:, 1:2][BL].p4.eta), weight=weight.weight()[BL])
#
        '''
        output['leading_btag_pt'].fill(dataset=dataset, pt=ak.flatten(high_score_btag[:, 0:1][BL].p4.pt), weight=weight.weight()[BL])
        output['subleading_btag_pt'].fill(dataset=dataset, pt=ak.flatten(high_score_btag[:, 1:2][BL].p4.pt), weight=weight.weight()[BL])
        output['leading_btag_eta'].fill(dataset=dataset, eta=ak.flatten(high_score_btag[:, 0:1][BL].p4.eta), weight=weight.weight()[BL])
        output['subleading_btag_eta'].fill(dataset=dataset, eta=ak.flatten(high_score_btag[:, 1:2][BL].p4.eta), weight=weight.weight()[BL])
        '''
#        
#        BL_minusMET = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['MET>50'])
        BL_minusMET = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['MET>50', 'N_central>0','N_fwd>0'])
        if dataset=='XG':
            BL_minusMET = (BL_minusMET & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusMET = (BL_minusMET & conversion_veto)

        output['MET'].fill(
            dataset = dataset,
            pt  = met[BL_minusMET].pt,
            phi  = met[BL_minusMET].phi,
            weight = weight.weight()[BL_minusMET]
        )
        
        #output['electron'].fill(
        #    dataset = dataset,
        #    pt  = ak.to_numpy(ak.flatten(electron[BL].pt)),
        #    eta = ak.to_numpy(ak.flatten(electron[BL].eta)),
        #    phi = ak.to_numpy(ak.flatten(electron[BL].phi)),
        #    weight = weight.weight()[BL]
        #)
        #
        #output['muon'].fill(
        #    dataset = dataset,
        #    pt  = ak.to_numpy(ak.flatten(muon[BL].pt)),
        #    eta = ak.to_numpy(ak.flatten(muon[BL].eta)),
        #    phi = ak.to_numpy(ak.flatten(muon[BL].phi)),
        #    weight = weight.weight()[BL]
        #)
        
        output['lead_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].p4.pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL].p4.eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['trail_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].p4.pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].p4.eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
               
        output['lead_lep_2ele'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL_diele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL_diele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL_diele].phi)),
            weight = weight.weight()[BL_diele]
        )
        
        output['trail_lep_2ele'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL_diele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL_diele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL_diele].phi)),
            weight = weight.weight()[BL_diele]
        )
            
        output['lead_lep_2mu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL_0ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL_0ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL_0ele].phi)),
            weight = weight.weight()[BL_0ele]
        )
        
        output['trail_lep_2mu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL_0ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL_0ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL_0ele].phi)),
            weight = weight.weight()[BL_0ele]
        )
        
        
        output['lead_lep_elemu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL_1ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL_1ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL_1ele].phi)),
            weight = weight.weight()[BL_1ele]
        )
        
        output['trail_lep_elemu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL_1ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL_1ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL_1ele].phi)),
            weight = weight.weight()[BL_1ele]
        )

        output['electron_elemu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(electron[BL_1ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(electron[BL_1ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(electron[BL_1ele].phi)),
            weight = weight.weight()[BL_1ele]
        )

        output['muon_elemu'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(muon[BL_1ele].p4.pt)),
            eta = ak.to_numpy(ak.flatten(muon[BL_1ele].p4.eta)),
            phi = ak.to_numpy(ak.flatten(muon[BL_1ele].phi)),
            weight = weight.weight()[BL_1ele]
        )
        
#        BL_bldr = BL & (ak.num(fwd)>0)
        BL_bldr = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_central>0'])
        
        output['fwd_jet'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_p_fwd[BL_bldr].p4.pt),
            eta = ak.flatten(high_p_fwd[BL_bldr].p4.eta),
            phi = ak.flatten(high_p_fwd[BL_bldr].phi),
            weight = weight.weight()[BL_bldr]
        )
#        
        '''
        output['b1'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_score_btag[:, 0:1][BL].p4.pt),
            eta = ak.flatten(high_score_btag[:, 0:1][BL].p4.eta),
            phi = ak.flatten(high_score_btag[:, 0:1][BL].phi),
            weight = weight.weight()[BL]
        )
        
        
        output['b2'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_score_btag[:, 1:2][BL].p4.pt),
            eta = ak.flatten(high_score_btag[:, 1:2][BL].p4.eta),
            phi = ak.flatten(high_score_btag[:, 1:2][BL].phi),
            weight = weight.weight()[BL]
        )
        '''
#        
        BL = BL & (ak.num(light)>0)
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(light.p4.pt[:, 0:1][BL]),
            eta = ak.flatten(light.p4.eta[:, 0:1][BL]),
            phi = ak.flatten(light.phi[:, 0:1][BL]),
            weight = weight.weight()[BL]
        )
        
        BL = sel.dilep_baseline(cutflow=cutflow, SS=False)
        
        output['j2'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 1:2][BL].p4.pt),
            eta = ak.flatten(jet[:, 1:2][BL].p4.eta),
            phi = ak.flatten(jet[:, 1:2][BL].phi),
            weight = weight.weight()[BL]
        )
        
        output['j3'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 2:3][BL].p4.pt),
            eta = ak.flatten(jet[:, 2:3][BL].p4.eta),
            phi = ak.flatten(jet[:, 2:3][BL].phi),
            weight = weight.weight()[BL]
        )

        if re.search(data_pattern, dataset):
            #rle = ak.to_numpy(ak.zip([ev.run, ev.luminosityBlock, ev.event]))
            run_ = ak.to_numpy(ev.run)
            lumi_ = ak.to_numpy(ev.luminosityBlock)
            event_ = ak.to_numpy(ev.event)
            output['%s_run'%dataset] += processor.column_accumulator(run_[BL])
            output['%s_lumi'%dataset] += processor.column_accumulator(lumi_[BL])
            output['%s_event'%dataset] += processor.column_accumulator(event_[BL])
        
        # Now, take care of systematic unceratinties
        if not re.search(data_pattern, dataset):
            alljets = getJets(ev, minPt=0, maxEta=4.7, pt_var='pt_nom')
            alljets = alljets[(alljets.jetId>1)]
            for var in self.variations:
                
                # get the collections that change with the variations
                met = getMET(ev, pt_var='pt_nom')
                #jet = alljets
                #jet = getPtEtaPhi(alljets, pt_var='pt_nom')
                jet = getJets(ev, minPt=0, maxEta=4.7, pt_var='pt_nom')
                jet = jet[(jet.p4.pt>25)]
                jet = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
                jet = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

                central   = jet[(abs(jet.p4.eta)<2.4)]
                btag      = getBTagsDeepFlavB(jet, year=self.year) # should study working point for DeepJet
                light     = getBTagsDeepFlavB(jet, year=self.year, invert=True)
                fwd       = getFwdJet(light)
                fwd_noPU  = getFwdJet(light, puId=False)        
        
                ## forward jets
                high_p_fwd   = fwd[ak.singletons(ak.argmax(fwd.p4.p, axis=1))] # highest momentum spectator
                high_pt_fwd  = fwd[ak.singletons(ak.argmax(fwd.p4.pt, axis=1))]  # highest transverse momentum spectator
                high_eta_fwd = fwd[ak.singletons(ak.argmax(abs(fwd.p4.eta), axis=1))] # most forward spectator
        
                ## Get the two leading b-jets in terms of btag score
                high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

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
#                BL = sel.dilep_baseline(cutflow=cutflow, SS=False)
                BL = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0', 'N_central>0'])
          
                if dataset=='XG':
                    BL = (BL & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL = (BL & conversion_veto)

                #BL = sel.dilep_baseline(SS=False)

                # get the modified selection -> more difficult
                #selection.add('N_jet>2_'+var, (ak.num(jet.pt)>=3)) # stupid bug here...
                #selection.add('N_btag=2_'+var,      (ak.num(btag)==2) ) 
                #selection.add('N_central>1_'+var,   (ak.num(central)>=2) )
                #selection.add('N_fwd>0_'+var,       (ak.num(fwd)>=1) )
                #selection.add('MET>30_'+var, (getattr(ev.MET, var)>30) )

                ### Don't change the selection for now...
                #bl_reqs = os_reqs + ['N_jet>2_'+var, 'MET>30_'+var, 'N_btag=2_'+var, 'N_central>1_'+var, 'N_fwd>0_'+var]
                #bl_reqs_d = { sel: True for sel in bl_reqs }
                #BL = selection.require(**bl_reqs_d)

                # the OS selection remains unchanged
                output['N_jet_'+var].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight.weight()[BL])
                
#                BL_minusFwd = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0'])
                BL_minusFwd = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_fwd>0', 'N_central>0'])
                if dataset=='XG':
                    BL_minusFwd = (BL_minusFwd & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusFwd = (BL_minusFwd & conversion_veto)
                    
                output['N_fwd_'+var].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL_minusFwd], weight=weight.weight()[BL_minusFwd])
                
#                BL_minusNb = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_btag>0'])
                BL_minusNb = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_btag=0','N_fwd>0', 'N_central>0'])
                if dataset=='XG':
                    BL_minusNb = (BL_minusNb & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusNb = (BL_minusNb & conversion_veto)
                    
                output['N_b_'+var].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])
                output['N_central_'+var].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight.weight()[BL])


                # We don't need to redo all plots with variations. E.g., just add uncertainties to the jet plots.
                BL = BL & (ak.num(light)>0)
                output['j1_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(light.p4.pt[:, 0:1][BL]),
                    eta = ak.flatten(light.p4.eta[:, 0:1][BL]),
                    phi = ak.flatten(light.phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
                BL = sel.dilep_baseline(cutflow=cutflow, SS=False)
#
                '''
                output['b1_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(high_score_btag[:, 0:1].p4.pt[:, 0:1][BL]),
                    eta = ak.flatten(high_score_btag[:, 0:1].p4.eta[:, 0:1][BL]),
                    phi = ak.flatten(high_score_btag[:, 0:1].phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
                '''
#    
#                BL_bldr = BL & (ak.num(fwd)>0)
                BL_bldr = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['N_central>0'])
                
                output['fwd_jet_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(high_p_fwd[BL_bldr].p4.pt),
                    #p   = ak.flatten(high_p_fwd[BL].p),
                    eta = ak.flatten(high_p_fwd[BL_bldr].p4.eta),
                    phi = ak.flatten(high_p_fwd[BL_bldr].phi),
                    weight = weight.weight()[BL_bldr]
                ) 
                
#                BL_minusMET = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['MET>50'])
                BL_minusMET = sel.dilep_baseline(cutflow=cutflow, SS=False, omit=['MET>50','N_fwd>0', 'N_central>0'])
                if dataset=='XG':
                    BL_minusMET = (BL_minusMET & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusMET = (BL_minusMET & conversion_veto)
                   
                #output['MET_'+var].fill(
                #    dataset = dataset,
                #    #pt  = getattr(ev.MET, var)[BL_minusMET],
                #    pt  = ev.MET[BL_minusMET].T1_pt,
                #    phi  = ev.MET[BL_minusMET].phi,
                #    weight = weight.weight()[BL_minusMET]
                #)
                
                output['MET_'+var].fill(
                    dataset = dataset,
                    #pt  = getattr(ev.MET, var)[BL_minusMET],
                    pt  = met[BL_minusMET].pt,
                    phi  = met[BL_minusMET].phi,
                    weight = weight.weight()[BL_minusMET]
                )
        
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
    argParser.add_argument('--year', action='store', default='2016', help="Which year to run on?")
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
    
    cacheName = 'forward_OS_%s%s'%(year,era)
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    in_path = '/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.5.2_dilep/'

    fileset_all = get_babies(in_path, year='UL%s%s'%(year,era))
    fileset = {
        #'tW_scattering': fileset_all['tW_scattering'],
        'topW_v3': fileset_all['topW_NLO'],
        #'topW_v3': fileset_all['topW_v3'],
        #'ttbar': fileset_all['ttbar2l'], # dilepton ttbar should be enough for this study.
        'ttbar': fileset_all['top'], # dilepton ttbar should be enough for this study.
        'MuonEG': fileset_all['MuonEG'],
        'DoubleMuon': fileset_all['DoubleMuon'],
        'EGamma': fileset_all['EGamma'],
        'DoubleEG': fileset_all['DoubleEG'],
        'SingleMuon': fileset_all['SingleMuon'],
        'SingleElectron': fileset_all['SingleElectron'],
        'diboson': fileset_all['diboson'],
        'TTXnoW': fileset_all['TTXnoW'],
        'TTW': fileset_all['TTW'],
        #'WZ': fileset_all['WZ'],
        'DY': fileset_all['DY'],
        'XG': fileset_all['XG'],
    }

    fileset = make_small(fileset, small, n_max=10)

    add_processes_to_output(fileset, desired_output)
    for rle in ['run', 'lumi', 'event']:
        desired_output.update({
                'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'SingleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'SingleElectron_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                "N_tau":                hist.Hist("Counts", dataset_axis, multiplicity_axis),
                "N_track":              hist.Hist("Counts", dataset_axis, multiplicity_axis),
                "dilep_pt":             hist.Hist("Counts", dataset_axis, pt_axis),
                "dilep_mass":           hist.Hist("Counts", dataset_axis, mass_axis),
                "deltaEta":             hist.Hist("Counts", dataset_axis, eta_axis),
                "mjf_max":              hist.Hist("Counts", dataset_axis, ext_mass_axis),
                "min_bl_dR":            hist.Hist("Counts", dataset_axis, eta_axis),
                "min_mt_lep_met":       hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_jet_pt":       hist.Hist("Counts", dataset_axis, pt_axis),
                "subleading_jet_pt":    hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_jet_eta":      hist.Hist("Counts", dataset_axis, eta_axis),
                "subleading_jet_eta":   hist.Hist("Counts", dataset_axis, eta_axis),
            
                "leading_btag_pt":      hist.Hist("Counts", dataset_axis, pt_axis),
                "subleading_btag_pt":   hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_btag_eta":     hist.Hist("Counts", dataset_axis, eta_axis),
                "subleading_btag_eta":  hist.Hist("Counts", dataset_axis, eta_axis),
                "lead_lep_2ele":        hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "trail_lep_2ele":       hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "lead_lep_2mu":         hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "trail_lep_2mu":        hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "lead_lep_elemu":       hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "trail_lep_elemu":      hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "electron_elemu":       hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "muon_elemu":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "dR_mu_genphoton":      hist.Hist("Counts", dataset_axis, eta_axis),
                
             })

    histograms = sorted(list(desired_output.keys()))

    
    if not overwrite:
        cache.load()
    
    
    if local:
        exe_args = {
            'workers': 12,
            "schema": NanoAODSchema,
        }
        exe = processor.futures_executor

    else:
        from Tools.helpers import get_scheduler_address
        from dask.distributed import Client, progress

        scheduler_address = get_scheduler_address()
        c = Client(scheduler_address)

        exe_args = {
            'client': c,
            "schema": NanoAODSchema,
        }
        exe = processor.dask_executor

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            forwardJetAnalyzer(year=year, variations=['pt_jesTotalDown', 'pt_jesTotalUp'], accumulator=desired_output, dump=args.dump, era=era ),  # not using variations now
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    # this is a check for double counting.

    '''
    em = zip_run_lumi_event(output, 'MuonEG')
    e  = zip_run_lumi_event(output, 'EGamma')
    mm = zip_run_lumi_event(output, 'DoubleMuon')
    print ("Total events from MuonEG:", len(em))
    print ("Total events from EGamma:", len(e))
    print ("Total events from DoubleMuon:", len(mm))
    em_mm = np.intersect1d(em, mm)
    print ("Overlap MuonEG/DoubleMuon:", len(em_mm))
    # print (em_mm)
    e_mm = np.intersect1d(e, mm)
    print ("Overlap EGamma/DoubleMuon:", len(e_mm))
    # print (e_mm)
    em_e = np.intersect1d(em, e)
    print ("Overlap MuonEG/EGamma:", len(em_e))
    # print (em_e)'''
