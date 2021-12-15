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
from Tools.config_helpers import loadConfig, make_small, data_pattern #,  zip_run_lumi_event
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.ttH_lepton_scalefactors import *
from Tools.selections import Selection, get_pt
from Tools.helpers import mt

import warnings
warnings.filterwarnings("ignore")



class forwardJetAnalyzer(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}, evaluate=False, training='v8', dump=False, era=None):
        self.variations = variations
        self.era = era
        self.year = year
        
        self.btagSF = btag_scalefactor(year)
        
        self.leptonSF = LeptonSF(year=year)
        
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
        second_lepton = lepton[~(trailing_lepton_idx & leading_lepton_idx)]
         
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
        
        ## forward jets
        high_p_fwd   = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator
        high_pt_fwd  = fwd[ak.singletons(ak.argmax(fwd.pt_nom, axis=1))]  # highest transverse momentum spectator
        high_eta_fwd = fwd[ak.singletons(ak.argmax(abs(fwd.eta), axis=1))] # most forward spectator
        
        ## Get the two leading b-jets in terms of btag score
        high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]
        
        jf          = cross(high_p_fwd, jet)
        mjf         = (jf['0']+jf['1']).mass
        deltaEta    = abs(high_p_fwd.eta - jf[ak.singletons(ak.argmax(mjf, axis=1))]['1'].eta)
        #deltaEta   = abs(jf['0'].eta - jf['1'].eta)
        deltaEtaMax = ak.max(deltaEta, axis=1)
        mjf_max     = ak.max(mjf, axis=1)
        
        jj          = choose(jet, 2)
        mjj_max     = ak.max((jj['0']+jj['1']).mass, axis=1)
        
        ## MET -> can switch to puppi MET
        met_pt  = ev.MET.pt
        met_phi = ev.MET.phi

        ## other variables
        ht = ak.sum(jet.pt, axis=1)
        st = met_pt + ht + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)
        lt = met_pt + ak.sum(muon.pt, axis=1) + ak.sum(electron.pt, axis=1)
        ht_central = ak.sum(central.pt, axis=1)
        
        track     = getIsoTracks(ev)
        tau       = getTaus(ev)
        tau       = tau[~match(tau, muon, deltaRCut=0.4)] 
        tau       = tau[~match(tau, electron, deltaRCut=0.4)]
        
        bl          = cross(lepton, high_score_btag)
        bl_dR       = delta_r(bl['0'], bl['1'])
        min_bl_dR   = ak.min(bl_dR, axis=1)

        mt_lep_met = mt(lepton.pt, lepton.phi, ev.MET.pt, ev.MET.phi)
        min_mt_lep_met = ak.min(mt_lep_met, axis=1)
        dilepton_dR = delta_r(leading_lepton, trailing_lepton)
        
               
        
        # define the weight
        weight = Weights( len(ev) )
        
        if not re.search(data_pattern, dataset):
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
           # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
            
        if not re.search(data_pattern, dataset):
            gen = ev.GenPart
            gen_photon = gen[gen.pdgId==22]
            external_conversions = external_conversion(lepton, gen_photon)
            conversion_veto = ((ak.num(external_conversions))==0)
            conversion_req = ((ak.num(external_conversions))>0)
        
        
        cutflow     = Cutflow(output, ev, weight=weight)

        sel = Selection(
            dataset = dataset,
            events = ev,
            year = self.year,
            ele = electron,
            ele_veto = vetoelectron,
            mu = muon,
            mu_veto = vetomuon,
            jet_all = jet,
            jet_central = central,
            jet_btag = btag,
            jet_fwd = fwd,
            met = ev.MET,
        )
        
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0', 'SFOS>1'])
        
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
            
        
        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvs, weight=weight.weight()[BL])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[BL].npvsGood, weight=weight.weight()[BL])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[BL], weight=weight.weight()[BL])
        
        BL_minusNb = sel.trilep_baseline(omit=['N_btag>0']) 
#        BL_minusNb = sel.trilep_baseline(omit=['N_btag>0','N_fwd>0', 'N_central>0','SFOS>1'])   
          
        if dataset=='XG':
            BL_minusNb = (BL_minusNb & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusNb = (BL_minusNb & conversion_veto)
            
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])

        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight.weight()[BL])
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[BL], weight=weight.weight()[BL])
        
        output['ST'].fill(dataset=dataset, ht=st[BL], weight=weight.weight()[BL])
        output['HT'].fill(dataset=dataset, ht=ht[BL], weight=weight.weight()[BL])
        output['LT'].fill(dataset=dataset, ht=lt[BL], weight=weight.weight()[BL])
        
        vetolepton   = ak.concatenate([vetomuon, vetoelectron], axis=1)    
        trilep = choose3(lepton, 3)
        #trilep_m = trilep.mass
        trilep_m = ak.max(trilep.mass, axis=1)
        
           
        dimu_veto = choose(vetomuon,2)
        diele_veto = choose(vetoelectron,2) 
        OS_dimu_veto = dimu_veto[(dimu_veto['0'].charge*dimu_veto['1'].charge < 0)]
        OS_diele_veto = diele_veto[(diele_veto['0'].charge*diele_veto['1'].charge < 0)]
        
        OS_dimuon_bestZmumu = OS_dimu_veto[ak.singletons(ak.argmin(abs(OS_dimu_veto.mass-91.2), axis=1))]
        OS_dielectron_bestZee = OS_diele_veto[ak.singletons(ak.argmin(abs(OS_diele_veto.mass-91.2), axis=1))]
        OS_dilepton_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.mass, OS_dielectron_bestZee.mass], axis=1), 1, clip=True), -1)
        
        OS_dilepton_pt = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_bestZmumu.pt, OS_dielectron_bestZee.pt], axis=1), 1, clip=True), -1)
        
        OS_dilepton_all_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimu_veto.mass, OS_diele_veto.mass], axis=1), 1, clip=True), -1)
        
        SFOS = ak.concatenate([OS_diele_veto, OS_dimu_veto], axis=1)
        OS_dimu_veto2 = OS_dimu_veto[ak.num(SFOS)>1]
        OS_diele_veto2 = OS_diele_veto[ak.num(SFOS)>1]
        OS_dimuon_worstZmumu = OS_dimu_veto[ak.singletons(ak.argmax(abs(OS_dimu_veto.mass-91.2), axis=1))]
        OS_dielectron_worstZee = OS_diele_veto[ak.singletons(ak.argmax(abs(OS_diele_veto.mass-91.2), axis=1))]
        OS_dilepton_worst_mass = ak.fill_none(ak.pad_none(ak.concatenate([OS_dimuon_worstZmumu.mass, OS_dielectron_worstZee.mass], axis=1), 1, clip=True), -1) 
        
        OS_min_mass = ak.fill_none(ak.min(ak.concatenate([OS_dimu_veto.mass, OS_diele_veto.mass], axis=1), axis=1),0)
        
        

        
        BL_omitOffZ = sel.trilep_baseline(omit=['offZ'])
        BL_omitOnZ = sel.trilep_baseline(omit=['onZ'])
#        BL_omitOffZ = sel.trilep_baseline(omit=['offZ','SFOS>1'])
#        BL_omitOnZ = sel.trilep_baseline(omit=['m3l_onZ','SFOS>1'])

        if dataset=='XG':
            BL_omitOffZ = (BL_omitOffZ & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_omitOffZ = (BL_omitOffZ & conversion_veto)
          
        if dataset=='XG':
            BL_omitOnZ = (BL_omitOnZ & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_omitOnZ = (BL_omitOnZ & conversion_veto)
            
        output['min_mass_SFOS'].fill(dataset=dataset, mass=(OS_min_mass[BL_omitOnZ]), weight=weight.weight()[BL_omitOnZ])    
        output['onZ_pt'].fill(dataset=dataset, pt=ak.flatten(OS_dilepton_pt[BL]), weight=weight.weight()[BL])
        output['M3l'].fill(dataset=dataset, mass=(trilep_m[BL]), weight=weight.weight()[BL])
        output['M_ll'].fill(dataset=dataset, mass=ak.flatten(OS_dilepton_mass[BL_omitOnZ]), weight=weight.weight()[BL_omitOnZ])
        #output['M_ll_all'].fill(dataset=dataset, mass=ak.flatten(OS_dilepton_all_mass[BL_omitOffZ]), weight=weight.weight()[BL_omitOffZ])
        
        BL_omitOffZ = sel.trilep_baseline(omit=['offZ'])
        if dataset=='XG':
            BL_omitOffZ = (BL_omitOffZ & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_omitOffZ = (BL_omitOffZ & conversion_veto)
        
        #output['M_ll_worst'].fill(dataset=dataset, mass=ak.flatten(OS_dilepton_worst_mass[BL_omitOffZ]), weight=weight.weight()[BL_omitOffZ])

        output['N_tau'].fill(dataset=dataset, multiplicity=ak.num(tau)[BL], weight=weight.weight()[BL])
        output['N_track'].fill(dataset=dataset, multiplicity=ak.num(track)[BL], weight=weight.weight()[BL])
        
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_central>0','SFOS>1'])
        if dataset=='XG':
            BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_central>0','SFOS>1'])
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
        
        output['mjf_max'].fill(dataset=dataset, mass=mjf_max[BL], weight=weight.weight()[BL])
        output['deltaEta'].fill(dataset=dataset, eta=ak.flatten(deltaEta[BL]), weight=weight.weight()[BL])
        
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0','SFOS>1'])
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
        
        output['min_bl_dR'].fill(dataset=dataset, eta=min_bl_dR[BL], weight=weight.weight()[BL])
        output['min_mt_lep_met'].fill(dataset=dataset, pt=min_mt_lep_met[BL], weight=weight.weight()[BL])
        
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
            
        output['leading_jet_pt'].fill(dataset=dataset, pt=ak.flatten(jet[:, 0:1][BL].pt), weight=weight.weight()[BL])
        
        output['leading_jet_eta'].fill(dataset=dataset, eta=ak.flatten(jet[:, 0:1][BL].eta), weight=weight.weight()[BL])
#        '''
        output['subleading_jet_pt'].fill(dataset=dataset, pt=ak.flatten(jet[:, 1:2][BL].pt), weight=weight.weight()[BL])
        output['subleading_jet_eta'].fill(dataset=dataset, eta=ak.flatten(jet[:, 1:2][BL].eta), weight=weight.weight()[BL])
        
        output['leading_btag_pt'].fill(dataset=dataset, pt=ak.flatten(high_score_btag[:, 0:1][BL].pt), weight=weight.weight()[BL])
        output['subleading_btag_pt'].fill(dataset=dataset, pt=ak.flatten(high_score_btag[:, 1:2][BL].pt), weight=weight.weight()[BL])
        output['leading_btag_eta'].fill(dataset=dataset, eta=ak.flatten(high_score_btag[:, 0:1][BL].eta), weight=weight.weight()[BL])
        output['subleading_btag_eta'].fill(dataset=dataset, eta=ak.flatten(high_score_btag[:, 1:2][BL].eta), weight=weight.weight()[BL])
#        '''        
        BL_minusFwd = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0'])
#        BL_minusFwd = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
        if dataset=='XG':
            BL_minusFwd = (BL_minusFwd & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusFwd = (BL_minusFwd & conversion_veto)
        
        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL_minusFwd], weight=weight.weight()[BL_minusFwd])
        
        BL_minusMET = sel.trilep_baseline(cutflow=cutflow, omit=['MET>50'])
#        BL_minusMET = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
        if dataset=='XG':
            BL_minusMET = (BL_minusMET & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL_minusMET = (BL_minusMET & conversion_veto)
            
        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[BL_minusMET].pt,
            phi  = ev.MET[BL_minusMET].phi,
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
            pt  = ak.to_numpy(ak.flatten(leading_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(leading_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(leading_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['trail_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        
        output['second_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(second_lepton[BL].pt)),
            eta = ak.to_numpy(ak.flatten(second_lepton[BL].eta)),
            phi = ak.to_numpy(ak.flatten(second_lepton[BL].phi)),
            weight = weight.weight()[BL]
        )
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_central>0','SFOS>1'])
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
            
        output['fwd_jet'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_p_fwd[BL].pt_nom),
            eta = ak.flatten(high_p_fwd[BL].eta),
            phi = ak.flatten(high_p_fwd[BL].phi),
            weight = weight.weight()[BL]
        )
        BL = sel.trilep_baseline(cutflow=cutflow)
#        BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
        if dataset=='XG':
            BL = (BL & conversion_req)
        elif dataset=='ttbar' or dataset=='DY':
            BL = (BL & conversion_veto)
#        '''        
        output['b1'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_score_btag[:, 0:1][BL].pt_nom),
            eta = ak.flatten(high_score_btag[:, 0:1][BL].eta),
            phi = ak.flatten(high_score_btag[:, 0:1][BL].phi),
            weight = weight.weight()[BL]
        )
        
        output['b2'].fill(
            dataset = dataset,
            pt  = ak.flatten(high_score_btag[:, 1:2][BL].pt_nom),
            eta = ak.flatten(high_score_btag[:, 1:2][BL].eta),
            phi = ak.flatten(high_score_btag[:, 1:2][BL].phi),
            weight = weight.weight()[BL]
        )
#        '''
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
            eta = ak.flatten(jet.eta[:, 0:1][BL]),
            phi = ak.flatten(jet.phi[:, 0:1][BL]),
            weight = weight.weight()[BL]
        )
#        '''
        output['j2'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 1:2][BL].pt_nom),
            eta = ak.flatten(jet[:, 1:2][BL].eta),
            phi = ak.flatten(jet[:, 1:2][BL].phi),
            weight = weight.weight()[BL]
        )
        
        output['j3'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet[:, 2:3][BL].pt_nom),
            eta = ak.flatten(jet[:, 2:3][BL].eta),
            phi = ak.flatten(jet[:, 2:3][BL].phi),
            weight = weight.weight()[BL]
        )
#        '''

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
            alljets = getJets(ev, minPt=0, maxEta=4.7)
            alljets = alljets[(alljets.jetId>1)]
            for var in self.variations:
                # get the collections that change with the variations
                
                jet       = getJets(ev, minPt=25, maxEta=4.7, pt_var='pt_nom')
                jet       = jet[ak.argsort(jet.p4.pt, ascending=False)] # need to sort wrt smeared and recorrected jet pt
                jet       = jet[~match(jet, muon, deltaRCut=0.4)] # remove jets that overlap with muons
                jet       = jet[~match(jet, electron, deltaRCut=0.4)] # remove jets that overlap with electrons

                central   = jet[(abs(jet.eta)<2.4)]
                btag      = getBTagsDeepFlavB(jet, year=self.year) # should study working point for DeepJet
                light     = getBTagsDeepFlavB(jet, year=self.year, invert=True)
                fwd       = getFwdJet(light)
                fwd_noPU  = getFwdJet(light, puId=False)
        
                ## forward jets
                high_p_fwd   = fwd[ak.singletons(ak.argmax(fwd.p, axis=1))] # highest momentum spectator
                high_pt_fwd  = fwd[ak.singletons(ak.argmax(fwd.pt, axis=1))]  # highest transverse momentum spectator
                high_eta_fwd = fwd[ak.singletons(ak.argmax(abs(fwd.eta), axis=1))] # most forward spectator
        
                ## Get the two leading b-jets in terms of btag score
                high_score_btag = central[ak.argsort(central.btagDeepFlavB)][:,:2]

                met = ev.MET
                #met['pt'] = getattr(met, var)

                sel = Selection(
                    dataset = dataset,
                    events = ev,
                    year = self.year,
                    ele = electron,
                    ele_veto = vetoelectron,
                    mu = muon,
                    mu_veto = vetomuon,
                    jet_all = jet,
                    jet_central = central,
                    jet_btag = btag,
                    jet_fwd = fwd,
                    met = met,
                )

                BL = sel.trilep_baseline(cutflow=cutflow)
#                BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
                if dataset=='XG':
                    BL = (BL & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL = (BL & conversion_veto)

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
                                 
                BL_minusFwd = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0'])
#                BL_minusFwd = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
                if dataset=='XG':
                    BL_minusFwd = (BL_minusFwd & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusFwd = (BL_minusFwd & conversion_veto)
                
                output['N_fwd_'+var].fill(dataset=dataset, multiplicity=ak.num(fwd)[BL_minusFwd], weight=weight.weight()[BL_minusFwd])
                BL_minusNb = sel.trilep_baseline(omit=['N_btag>0'])
                if dataset=='XG':
                    BL_minusNb = (BL_minusNb & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusNb = (BL_minusNb & conversion_veto)
                    
                output['N_b_'+var].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])
                output['N_central_'+var].fill(dataset=dataset, multiplicity=ak.num(central)[BL], weight=weight.weight()[BL])


                # We don't need to redo all plots with variations. E.g., just add uncertainties to the jet plots.
                output['j1_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(jet.pt[:, 0:1][BL]),
                    eta = ak.flatten(jet.eta[:, 0:1][BL]),
                    phi = ak.flatten(jet.phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
#                '''                                 
                output['j2_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(jet.pt[:, 0:1][BL]),
                    eta = ak.flatten(jet.eta[:, 0:1][BL]),
                    phi = ak.flatten(jet.phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
                output['j3_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(jet.pt[:, 0:1][BL]),
                    eta = ak.flatten(jet.eta[:, 0:1][BL]),
                    phi = ak.flatten(jet.phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
                
                output['b1_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(high_score_btag[:, 0:1].pt[:, 0:1][BL]),
                    eta = ak.flatten(high_score_btag[:, 0:1].eta[:, 0:1][BL]),
                    phi = ak.flatten(high_score_btag[:, 0:1].phi[:, 0:1][BL]),
                    weight = weight.weight()[BL]
                )
#                '''
                BL = sel.trilep_baseline(cutflow=cutflow)
#                BL = sel.trilep_baseline(cutflow=cutflow, omit=['N_central>0','SFOS>1'])
                if dataset=='XG':
                    BL = (BL & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL = (BL & conversion_veto)
                
                output['fwd_jet_'+var].fill(
                    dataset = dataset,
                    pt  = ak.flatten(high_p_fwd[BL].pt),
                    #p   = ak.flatten(high_p_fwd[BL].p),
                    eta = ak.flatten(high_p_fwd[BL].eta),
                    phi = ak.flatten(high_p_fwd[BL].phi),
                    weight = weight.weight()[BL]
                )

                BL_minusMET = sel.trilep_baseline(cutflow=cutflow, omit=['MET>50'])
#                BL_minusMET = sel.trilep_baseline(cutflow=cutflow, omit=['N_fwd>0', 'N_central>0','SFOS>1'])
                if dataset=='XG':
                    BL_minusMET = (BL_minusMET & conversion_req)
                elif dataset=='ttbar' or dataset=='DY':
                    BL_minusMET = (BL_minusMET & conversion_veto)
                    
                output['MET_'+var].fill(
                    dataset = dataset,
                    #pt  = getattr(ev.MET, var)[BL_minusMET],
                    pt  = ev.MET[BL_minusMET].pt,
                    phi  = ev.MET[BL_minusMET].phi,
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
    
    #verysmall=True
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
    
    cacheName = 'onZ_nobreq_2016'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    in_path = '/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.5.2_trilep/'

    fileset_all = get_babies(in_path, year='UL%s%s'%(year,era))
   
    fileset = {
        #'tW_scattering': fileset_all['tW_scattering'],
        'topW_v3': fileset_all['topW_NLO'],
        #'topW_v3': fileset_all['topW_v3'],
        #'ttbar': fileset_all['ttbar2l'], # dilepton ttbar should be enough for this study.
        'ttbar': fileset_all['top'], # dilepton ttbar should be enough for this study.
        
        'MuonEG': fileset_all['MuonEG'],
        'DoubleMuon': fileset_all['DoubleMuon'],
        'EGamma': fileset_all['EGamma'], #DoubleEG for 2017, EGamma for 2018
        'SingleElectron': fileset_all['SingleElectron'],
        'SingleMuon': fileset_all['SingleMuon'],
        
        'diboson': fileset_all['diboson'],
        'TTXnoW': fileset_all['TTXnoW'],
        'TTW': fileset_all['TTW'],
        #'WZ': fileset_all['WZ'],
        'DY': fileset_all['DY'],
        'XG': fileset_all['XG'],
    }

    fileset = make_small(fileset, small, 1)
      
    add_processes_to_output(fileset, desired_output)
    for rle in ['run', 'lumi', 'event']:
        desired_output.update({
                'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'SingleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'SingleElectron_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                "M_ll": hist.Hist("Counts", dataset_axis, mass_axis),
                "M3l": hist.Hist("Counts", dataset_axis, ext_mass_axis),
                "ST": hist.Hist("Counts", dataset_axis, ht_axis),
                "HT": hist.Hist("Counts", dataset_axis, ht_axis),
                "LT": hist.Hist("Counts", dataset_axis, ht_axis),
                "onZ_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "min_mass_SFOS": hist.Hist("Counts", dataset_axis, mass_axis),
                "second_lep":          hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                "N_tau": hist.Hist("Counts", dataset_axis, multiplicity_axis),
                "N_track": hist.Hist("Counts", dataset_axis, multiplicity_axis),
                "deltaEta": hist.Hist("Counts", dataset_axis, delta_eta_axis),
                "mjf_max": hist.Hist("Counts", dataset_axis, ext_mass_axis),
                "min_bl_dR": hist.Hist("Counts", dataset_axis, eta_axis),
                "min_mt_lep_met": hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_jet_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "subleading_jet_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_jet_eta": hist.Hist("Counts", dataset_axis, eta_axis),
                "subleading_jet_eta": hist.Hist("Counts", dataset_axis, eta_axis),
                "leading_btag_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "subleading_btag_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "leading_btag_eta": hist.Hist("Counts", dataset_axis, eta_axis),
                "subleading_btag_eta": hist.Hist("Counts", dataset_axis, eta_axis),
                "M_ll_worst": hist.Hist("Counts", dataset_axis, mass_axis),
                "M_ll_all": hist.Hist("Counts", dataset_axis, mass_axis),
            
             })

    histograms = sorted(list(desired_output.keys()))

    
    if not overwrite:
        cache.load()
    
    
    if local:
        exe_args = {
            'workers': 12,
            'function_args': {'flatten': False},
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
            'function_args': {'flatten': False},
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
            forwardJetAnalyzer(year=year, variations=['pt_jesTotalDown', 'pt_jesTotalUp'], accumulator=desired_output),  # not using variations now
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
