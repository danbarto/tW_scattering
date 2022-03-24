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
from Tools.config_helpers import loadConfig, make_small
from Tools.triggers import *
from Tools.btag_scalefactors import *
from Tools.ttH_lepton_scalefactors import *
from Tools.selections import Selection

import warnings
warnings.filterwarnings("ignore")

def zip_rle(output, dataset):
    return ak.to_numpy(
        ak.zip([
            output['%s_run'%dataset].value.astype(int),
            output['%s_lumi'%dataset].value.astype(int),
            output['%s_event'%dataset].value.astype(int),
            ]))

class forwardJetAnalyzer(processor.ProcessorABC):
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
        
        ## Muons
        muon     = Collections(ev, "Muon", "tightSSTTH").get()
        vetomuon = Collections(ev, "Muon", "vetoTTH").get()
        dimuon   = choose(muon, 2)
        SSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)>0, axis=1)
        OSmuon   = ak.any((dimuon['0'].charge * dimuon['1'].charge)<0, axis=1)
        leading_muon_idx = ak.singletons(ak.argmax(muon.pt, axis=1))
        leading_muon = muon[leading_muon_idx]
        
        ## Electrons
        electron     = Collections(ev, "Electron", "tightSSTTH").get()
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
        
        
       
        
        # define the weight
        weight = Weights( len(ev) )
        
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            # lumi weight
            weight.add("weight", ev.weight*cfg['lumi'][self.year])
                        
            # PU weight - not in the babies...
            weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
            
            
            # b-tag SFs
            weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='central', c_direction='central'))

            # lepton SFs
            weight.add("lepton", self.leptonSF.get(electron, muon))
            
        
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
        
        BL = sel.dilep_baseline(SS=False)

        BL_minusNb = sel.dilep_baseline(SS=False, omit=['N_btag>0'])
        output['N_b'].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])
        

        if re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            #rle = ak.to_numpy(ak.zip([ev.run, ev.luminosityBlock, ev.event]))
            run_ = ak.to_numpy(ev.run)
            lumi_ = ak.to_numpy(ev.luminosityBlock)
            event_ = ak.to_numpy(ev.event)
            output['%s_run'%dataset] += processor.column_accumulator(run_[BL])
            output['%s_lumi'%dataset] += processor.column_accumulator(lumi_[BL])
            output['%s_event'%dataset] += processor.column_accumulator(event_[BL])
        
        # Now, take care of systematic unceratinties
        if not re.search(re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma'), dataset):
            alljets = getJets(ev, minPt=0, maxEta=4.7)
            alljets = alljets[(alljets.jetId>1)]
            for var in self.variations:
                # get the collections that change with the variations
                
                btag      = getBTagsDeepFlavB(jet, year=self.year) # should study working point for DeepJet
                weight = Weights( len(ev) )
                weight.add("weight", ev.weight*cfg['lumi'][self.year])
                weight.add("PU", ev.puWeight, weightUp=ev.puWeightUp, weightDown=ev.puWeightDown, shift=False)
                if var=='centralUp':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='central', c_direction='up'))
                elif var=='centralDown':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='central', c_direction='down'))
                elif var=='upCentral':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='up', c_direction='central'))
                elif var=='downCentral':
                    weight.add("btag", self.btagSF.Method1a(btag, light, b_direction='down', c_direction='central'))
                    
                weight.add("lepton", self.leptonSF.get(electron, muon))
                met = ev.MET
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

                BL = sel.dilep_baseline(SS=False)


                BL_minusNb = sel.dilep_baseline(SS=False,omit=['N_btag>0'])
                output['N_b_'+var].fill(dataset=dataset, multiplicity=ak.num(btag)[BL_minusNb], weight=weight.weight()[BL_minusNb])
                        

        
        return output

    def postprocess(self, accumulator):
        return accumulator



if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import * # fileset_2018 #, fileset_2018_small
    from processor.default_accumulators import *

    overwrite = True
    year = 2018
    local = True
    small = False

    # load the config and the cache
    cfg = loadConfig()
    
    cacheName = 'forward_dilep_2018_Nb'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)
    
    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.3_dilep/', year='UL2018')
    
    fileset = {
        #'tW_scattering': fileset_all['tW_scattering'],
        'topW_v3': fileset_all['topW_NLO'],
        #'topW_v3': fileset_all['topW_v3'],
        #'ttbar': fileset_all['ttbar2l'], # dilepton ttbar should be enough for this study.
        'ttbar': fileset_all['top'], # dilepton ttbar should be enough for this study.
        'MuonEG': fileset_all['MuonEG_Run2018'],
        'DoubleMuon': fileset_all['DoubleMuon_Run2018'],
        'EGamma': fileset_all['EGamma_Run2018'],
        'diboson': fileset_all['diboson'],
        'TTXnoW': fileset_all['TTXnoW'],
        'TTW': fileset_all['TTW'],
        #'WZ': fileset_all['WZ'],
        'DY': fileset_all['DY'],
    }


    fileset = make_small(fileset, small, 1)

    add_processes_to_output(fileset, desired_output)
    for rle in ['run', 'lumi', 'event']:
        desired_output.update({
                'MuonEG_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'EGamma_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                'DoubleMuon_%s'%rle: processor.column_accumulator(np.zeros(shape=(0,))),
                "M_ll": hist.Hist("Counts", dataset_axis, mass_axis),
                "M3l": hist.Hist("Counts", dataset_axis, mass_axis),
                "ST": hist.Hist("Counts", dataset_axis, ht_axis),
                "HT": hist.Hist("Counts", dataset_axis, ht_axis),
                "LT": hist.Hist("Counts", dataset_axis, ht_axis),
                "onZ_pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "min_mass_SFOS": hist.Hist("Counts", dataset_axis, mass_axis),
                "second_lep":          hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis),
                
            
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
            forwardJetAnalyzer(year=year, variations=['centralUp', 'centralDown', 'upCentral', 'downCentral'], accumulator=desired_output),  # not using variations now
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

    em = zip_rle(output, 'MuonEG')
    e = zip_rle(output, 'EGamma')
    mm = zip_rle(output, 'DoubleMuon')

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
    # print (em_e)