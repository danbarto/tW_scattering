import os
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
from Tools.helpers import pad_and_flatten, mt
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import *
from Tools.ttH_lepton_scalefactors import *
from Tools.selections import Selection

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx

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
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
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
            #weight.add("weight", ev.genWeight*cfg['lumi'][self.year]*mult)
            
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light))
            
            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
        

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
        
        BL = sel.dilep_baseline(cutflow=cutflow, SS=True)

        weight_BL = weight.weight()[BL]        

        if True:
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
        
        output['MET'].fill(
            dataset = dataset,
            pt  = ev.MET[BL].pt,
            phi  = ev.MET[BL].phi,
            weight = weight_BL
        )

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
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.default_accumulators import *

    overwrite = True
    small = True
    save = False

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'SS_analysis'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    year = 2018
    
    fileset = {
        'topW_v3': fileset_2018['topW_v3'],
        'topW_EFT_cp8': fileset_2018['topW_EFT_cp8'],
        'topW_EFT_mix': fileset_2018['topW_EFT_mix'],
        'TTW': fileset_2018['TTW'],
        'TTZ': fileset_2018['TTZ'],
        'TTH': fileset_2018['TTH'],
        'diboson': fileset_2018['diboson'],
        #'wpwp': fileset_2018['wpwp'],
        'TTTT': fileset_2018['TTTT'],
        'ttbar': fileset_2018['ttbar'],
        'MuonEG': fileset_2018['MuonEG'],
        'DoubleMuon': fileset_2018['DoubleMuon'],
        'EGamma': fileset_2018['EGamma'],
        #'topW_full_EFT': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_NLO_RunIIAutumn18_NANO_UL17_v7/*.root'),
        #'topW_NLO': glob.glob('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.2.5/ProjectMetis_TTWJetsToLNuEWK_5f_SMEFTatNLO_weight_RunIIAutumn18_NANO_UL17_v7/*.root'),
    }
    
    fileset = make_small(fileset, small)
    
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

