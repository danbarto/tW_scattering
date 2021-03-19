import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import awkward1 as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

from Tools.objects import *
from Tools.basic_objects import *
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import *
from Tools.ttH_lepton_scalefactors import *

os.environ['KERAS_BACKEND'] = 'theano'
#from keras.models import load_model
from ML.multiclassifier import load_model


class SS_analysis(processor.ProcessorABC):
    def __init__(self, year=2016, variations=[], accumulator={}):
        self.variations = variations
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
        
        ## Generated leptons
        gen_lep = ev.GenL
        leading_gen_lep = gen_lep[ak.singletons(ak.argmax(gen_lep.pt, axis=1))]
        trailing_gen_lep = gen_lep[ak.singletons(ak.argmin(gen_lep.pt, axis=1))]

        ## Muons
        muon     = Collections(ev, "Muon", "tightSSTTH").get()
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightSSTTH").get()
        vetoelectron = Collections(ev, "Electron", "vetoTTH").get()
        dielectron   = choose(electron, 2)
        SSelectron   = ak.any((dielectron['0'].charge * dielectron['1'].charge)>0, axis=1)
        leading_electron_idx = ak.singletons(ak.argmax(electron.pt, axis=1))
        leading_electron = electron[leading_electron_idx]
        
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
        
        ## event selectors
        filters   = getFilters(ev, year=self.year, dataset=dataset)
        triggers  = getTriggers(ev,
            ak.flatten(lepton_pdgId_pt_ordered[:,0:1]),
            ak.flatten(lepton_pdgId_pt_ordered[:,1:2]), year=self.year, dataset=dataset)
        
        dilep     = ((ak.num(electron) + ak.num(muon))==2)
        pos_charge = ((ak.sum(electron.pdgId, axis=1) + ak.sum(muon.pdgId, axis=1))<0)
        neg_charge = ((ak.sum(electron.pdgId, axis=1) + ak.sum(muon.pdgId, axis=1))>0)
        lep0pt    = ((ak.num(electron[(electron.pt>30)]) + ak.num(muon[(muon.pt>30)]))>0)
        lep0pt_40 = ((ak.num(electron[(electron.pt>40)]) + ak.num(muon[(muon.pt>40)]))>0)
        lep0pt_100 = ((ak.num(electron[(electron.pt>100)]) + ak.num(muon[(muon.pt>100)]))>0)
        lep1pt    = ((ak.num(electron[(electron.pt>20)]) + ak.num(muon[(muon.pt>20)]))>1)
        lep1pt_30 = ((ak.num(electron[(electron.pt>30)]) + ak.num(muon[(muon.pt>30)]))>1)
        lepveto   = ((ak.num(vetoelectron) + ak.num(vetomuon))==2)
        
        # define the weight
        weight = Weights( len(ev) )
        
        #mult = 1
        #if dataset=='inclusive': mult = 0.0478/47.448
        #if dataset=='plus': mult = 0.0036/7.205

        if not dataset=='MuonEG':
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
            #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        
        selection = PackedSelection()
        selection.add('lepveto',       lepveto)
        selection.add('dilep',         dilep )
        selection.add('filter',        (filters) )
        selection.add('trigger',       (triggers) )
        selection.add('p_T(lep0)>30',  lep0pt )
        selection.add('p_T(lep0)>40',  lep0pt_40 )
        selection.add('p_T(lep1)>20',  lep1pt )
        selection.add('p_T(lep1)>30',  lep1pt_30 )
        selection.add('SS',            ( SSlepton | SSelectron | SSmuon) )
        selection.add('pos',           ( pos_charge ) )
        selection.add('neg',           ( neg_charge ) )
        selection.add('N_jet>3',       (ak.num(jet)>=4) )
        selection.add('N_jet>4',       (ak.num(jet)>=5) )
        selection.add('N_central>2',   (ak.num(central)>=3) )
        selection.add('N_central>3',   (ak.num(central)>=4) )
        selection.add('N_btag>0',      (ak.num(btag)>=1) )
        selection.add('MET>50',        (ev.MET.pt>50) )
        selection.add('ST',            (st>600) )
        selection.add('N_fwd>0',       (ak.num(fwd)>=1 ))
        selection.add('delta_eta',     (ak.any(delta_eta>2, axis=1) ) )
        selection.add('fwd_p>500',     (ak.any(j_fwd.p>500, axis=1) ) )
        
        ss_reqs = ['lepveto', 'dilep', 'SS', 'filter', 'p_T(lep0)>30', 'p_T(lep1)>20', 'trigger', 'N_jet>3', 'N_central>2', 'N_btag>0']
        bl_reqs = ss_reqs + ['N_fwd>0', 'N_jet>4', 'N_central>3', 'ST', 'MET>50', 'delta_eta']
        sr_reqs = bl_reqs + ['fwd_p>500', 'p_T(lep0)>40', 'p_T(lep1)>30']

        ss_reqs_d = { sel: True for sel in ss_reqs }
        ss_selection = selection.require(**ss_reqs_d)
        bl_reqs_d = { sel: True for sel in bl_reqs }
        BL = selection.require(**bl_reqs_d)
        sr_reqs_d = { sel: True for sel in sr_reqs }
        SR = selection.require(**sr_reqs_d)

        cutflow     = Cutflow(output, ev, weight=weight)
        cutflow_reqs_d = {}
        for req in sr_reqs:
            cutflow_reqs_d.update({req: True})
            cutflow.addRow( req, selection.require(**cutflow_reqs_d) )
        

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

            model, scaler = load_model('v6')

            #print (np.shape(NN_inputs))
            try:
                NN_inputs_scaled = scaler.transform(NN_inputs)

                NN_pred    = model.predict( NN_inputs_scaled )

                best_score = np.argmax(NN_pred, axis=1)

            except ValueError:
                #print ("Empty NN_inputs")
                NN_pred = np.array([])
                best_score = np.array([])

            output['node'].fill(dataset=dataset, multiplicity=best_score, weight=weight.weight()[BL])


        # first, make a few super inclusive plots
        output['PV_npvs'].fill(dataset=dataset, multiplicity=ev.PV[ss_selection].npvs, weight=weight.weight()[ss_selection])
        output['PV_npvsGood'].fill(dataset=dataset, multiplicity=ev.PV[ss_selection].npvsGood, weight=weight.weight()[ss_selection])
        output['N_jet'].fill(dataset=dataset, multiplicity=ak.num(jet)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_central'].fill(dataset=dataset, multiplicity=ak.num(central)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_ele'].fill(dataset=dataset, multiplicity=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_mu'].fill(dataset=dataset, multiplicity=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        output['N_fwd'].fill(dataset=dataset, multiplicity=ak.num(fwd)[ss_selection], weight=weight.weight()[ss_selection])
        output['nLepFromTop'].fill(dataset=dataset, multiplicity=ev[BL].nLepFromTop, weight=weight.weight()[BL])
        output['nLepFromTau'].fill(dataset=dataset, multiplicity=ev.nLepFromTau[BL], weight=weight.weight()[BL])
        output['nLepFromZ'].fill(dataset=dataset, multiplicity=ev.nLepFromZ[BL], weight=weight.weight()[BL])
        output['nLepFromW'].fill(dataset=dataset, multiplicity=ev.nLepFromW[BL], weight=weight.weight()[BL])
        output['nGenTau'].fill(dataset=dataset, multiplicity=ev.nGenTau[BL], weight=weight.weight()[BL])
        output['nGenL'].fill(dataset=dataset, multiplicity=ak.num(ev.GenL[BL], axis=1), weight=weight.weight()[BL])
        output['chargeFlip_vs_nonprompt'].fill(dataset=dataset, n1=n_chargeflip[ss_selection], n2=n_nonprompt[ss_selection], n_ele=ak.num(electron)[ss_selection], weight=weight.weight()[ss_selection])
        
        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[ss_selection].pt,
            phi  = ev.MET[ss_selection].phi,
            weight = weight.weight()[ss_selection]
        )

        output['lead_gen_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(leading_gen_lep[BL].pt)),
            eta = ak.to_numpy(ak.flatten(leading_gen_lep[BL].eta)),
            phi = ak.to_numpy(ak.flatten(leading_gen_lep[BL].phi)),
            weight = weight.weight()[BL]
        )

        output['trail_gen_lep'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].pt)),
            eta = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].eta)),
            phi = ak.to_numpy(ak.flatten(trailing_gen_lep[BL].phi)),
            weight = weight.weight()[BL]
        )
        
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
        
        output['j1'].fill(
            dataset = dataset,
            pt  = ak.flatten(jet.pt_nom[:, 0:1][BL]),
            eta = ak.flatten(jet.eta[:, 0:1][BL]),
            phi = ak.flatten(jet.phi[:, 0:1][BL]),
            weight = weight.weight()[BL]
        )
        
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
        
        
        output['fwd_jet'].fill(
            dataset = dataset,
            pt  = ak.flatten(j_fwd[BL].pt),
            eta = ak.flatten(j_fwd[BL].eta),
            phi = ak.flatten(j_fwd[BL].phi),
            weight = weight.weight()[BL]
        )
            
        output['high_p_fwd_p'].fill(dataset=dataset, p = ak.flatten(j_fwd[BL].p), weight = weight.weight()[BL])
        
        return output

    def postprocess(self, accumulator):
        return accumulator




if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.default_accumulators import *

    overwrite = True
    small = True
    
    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'SS_analysis'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    year = 2018
    
    fileset = {
        'topW_v3': fileset_2018['topW_v3'],
        'TTW': fileset_2018['TTW'],
        'TTZ': fileset_2018['TTZ'],
        'TTH': fileset_2018['TTH'],
        'diboson': fileset_2018['diboson'],
        'ttbar': fileset_2018['ttbar'],
    }
    
    fileset = make_small(fileset, small)
    
    add_processes_to_output(fileset, desired_output)

    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
    }
    exe = processor.futures_executor

    # add some histograms that we defined in the processor
    # everything else is taken the default_accumulators.py
    from processor.default_accumulators import multiplicity_axis, dataset_axis
    desired_output.update({
        "node": hist.Hist("Counts", dataset_axis, multiplicity_axis),
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
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    
    import warnings
    warnings.filterwarnings('ignore')
    
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
    
    my_labels = {
        'topW_v3': 'top-W scat.',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'ttbar': r'$t\bar{t}$',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'ttbar': '#1982C4',
    }

    makePlot(output, 'node', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'$p_{T}$ (lead lep) (GeV)',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar', 'topW_v3'],
         save='/home/users/dspitzba/public_html/tW_scattering/dump/ML_node',
        )
