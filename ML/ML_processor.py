import os
import time
import glob
import re
import pandas as pd
from functools import reduce
from klepto.archives import dir_archive

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea.analysis_objects import JaggedCandidateArray
from coffea.btag_tools import BTagScaleFactor
from coffea import hist
import pandas as pd
import uproot_methods
import uproot
import awkward
import copy

from memory_profiler import profile

from Tools.helpers import loadConfig, getCutFlowTable, mergeArray
from Tools.objects import Collections
from Tools.cutflow import Cutflow

class analysisProcessor(processor.ProcessorABC):
    """Processor used for running the analysis"""
    def __init__(self):
        
        ## load b-tag SFs
        #self.btag_sf = BTagScaleFactor(os.path.expandvars("$TWHOME/data/DeepCSV_102XSF_V1.btag.csv.gz", "reshape")

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 1000, 0, 1000)

        self._accumulator = processor.dict_accumulator({
            'diboson':          processor.defaultdict_accumulator(int),
            'ttbar':            processor.defaultdict_accumulator(int),
            'TTW':              processor.defaultdict_accumulator(int),
            'TTZ':              processor.defaultdict_accumulator(int),
            'TTH':              processor.defaultdict_accumulator(int),
            'TTTT':             processor.defaultdict_accumulator(int),
            'tW_scattering':    processor.defaultdict_accumulator(int),
            'DY':               processor.defaultdict_accumulator(int),
            'totalEvents':      processor.defaultdict_accumulator(int),
            'passedEvents':      processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        dataset = df["dataset"]
        cfg = loadConfig()
        
        ## Muons
        muon = Collections(df, "Muon", "tight").get()
        vetomuon = Collections(df, "Muon", "veto").get()
        dimuon = muon.choose(2)
        SSmuon = ( dimuon[(dimuon.i0.charge * dimuon.i1.charge)>0].counts>0 )
        
        ## Electrons
        electron = Collections(df, "Electron", "tight").get()
        vetoelectron = Collections(df, "Electron", "veto").get()
        dielectron = electron.choose(2)
        SSelectron = ( dielectron[(dielectron.i0.charge * dielectron.i1.charge)>0].counts>0 )

        ## E/Mu cross
        dilepton = electron.cross(muon)
        SSdilepton = ( dilepton[(dilepton.i0.charge * dilepton.i1.charge)>0].counts>0 )
        
        ## how to get leading lepton easily? Do I actually care?
        leading_muon = muon[muon.pt.argmax()]
        leading_electron = electron[electron.pt.argmax()]
        
        lepton = mergeArray(electron, muon)
        '''
        ok so this is getting **really** awkward (pun slightly intended). because the mergeArray function builds a JaggedArray that has a UnionArry as .content, which in turn
        does not work with .argmax(), we need to build a jagged array just holding the pts
        '''
        lepton_pt = awkward.concatenate([electron.pt, muon.pt], axis=1)
        leading_lep_index = lepton_pt.argmax()
        trailing_lep_index = lepton_pt.argmin()

        #leading_lep = lepton[lepton.p4.pt().argmax()]
        #leading_lep_pt = lepton.p4.fPt[:,:1] 
        leading_lep_pt = lepton[leading_lep_index].p4.fPt.max() # taking the max here has no impact, but otherwise code fails
        leading_lep_eta = lepton[leading_lep_index].p4.fEta.max() # taking the max here has no impact, but otherwise code fails

        trailing_lep_pt = lepton[trailing_lep_index].p4.fPt.max() # taking the max here has no impact, but otherwise code fails
        trailing_lep_eta = lepton[trailing_lep_index].p4.fEta.max() # taking the max here has no impact, but otherwise code fails

        ## Jets
        jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            jetId = df['Jet_jetId'].content, # https://twiki.cern.ch/twiki/bin/view/CMS/JetID
            puId = df['Jet_puId'].content, # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetID
            btagDeepB = df['Jet_btagDeepB'].content, # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
        )
        
        jet       = jet[(jet.pt>25) & (jet.jetId>1)]
        jet       = jet[~jet.match(muon, deltaRCut=0.4)] # remove jets that overlap with muons
        jet       = jet[~jet.match(electron, deltaRCut=0.4)] # remove jets that overlap with electrons
        btag      = jet[(jet.btagDeepB>0.4184) & (abs(jet.eta)<2.4)]
        central   = jet[(abs(jet.eta)<2.4)]
        light     = jet[((jet.btagDeepB<0.4184) & (abs(jet.eta)<2.4)) | (abs(jet.eta)>=2.4)]
        lightCentral = jet[(jet.btagDeepB<0.4184) & (abs(jet.eta)<2.4) & (jet.pt>30)]
        fw        = light[abs(light.eta).argmax()] # the most forward light jet
        
        leading_jet = jet[jet.pt.argmax()]
        leading_b = btag[btag.pt.argmax()]

        mass_eb = electron.cross(btag).mass
        mass_mub = muon.cross(btag).mass
        mass_lb = awkward.concatenate([mass_eb, mass_mub], axis=1)
        mlb_min = mass_lb.min()
        mlb_max = mass_lb.max()

        mll = awkward.concatenate([dimuon.mass, dielectron.mass, dilepton.mass], axis=1).max() # max shouldn't matter, again
        ej  = electron.cross(jet)
        muj = muon.cross(jet)
        deltaR_ej = ej.i0.p4.delta_r(ej.i1.p4)
        deltaR_muj = muj.i0.p4.delta_r(muj.i1.p4)

        deltaR_lj = awkward.concatenate([deltaR_ej,deltaR_muj], axis=1)
        deltaR_lj_min = deltaR_lj.min()

        ## MET -> can switch to puppi MET
        met_pt  = df["MET_pt"]
        met_phi = df["MET_phi"]

        ## other variables
        st = df["MET_pt"] + jet.pt.sum() + muon.pt.sum() + electron.pt.sum()
        ht = jet.pt.sum()
        
        light_light = light.choose(2)
        mjj_max = light_light[light_light.mass.argmax()].mass
        
        ## define selections (maybe move to a different file at some point)
        dilep      = ((electron.counts + muon.counts)==2)
        leppt      = (((electron.pt>25).counts + (muon.pt>25).counts)>0)
        lepveto    = ((vetoelectron.counts + vetomuon.counts)==2)
        SS         = (SSelectron | SSmuon | SSdilepton)

        output['totalEvents']['all'] += len(df['weight'])
        
        # Cutflow
        processes = ['tW_scattering', 'TTW', 'TTZ', 'TTH', 'TTTT', 'diboson', 'ttbar', 'DY']
        cutflow = Cutflow(output, df, cfg, processes)
        
        cutflow.addRow( 'dilep',       dilep )
        cutflow.addRow( 'leppt',       leppt )
        cutflow.addRow( 'lepveto',     lepveto )
        #cutflow.addRow( 'SS',          SS )
        cutflow.addRow( 'njet4',       (jet.counts>=4) )
        cutflow.addRow( 'light2',      (light.counts>=2) ) 
        cutflow.addRow( 'nbtag',       btag.counts>0 )
        
        baseline = cutflow.selection

        output['passedEvents']['all'] += len(df['weight'][baseline])
        
        nEventsBaseline = len(df['weight'][baseline])
        signal_label = np.ones(nEventsBaseline) if (df['dataset']=='tW_scattering' or df['dataset']=='TTW') else np.zeros(nEventsBaseline)
        
        df_out = pd.DataFrame({
            'j0_pt':    leading_jet[baseline].pt.flatten(),
            'j0_eta':   leading_jet[baseline].eta.flatten(),
            'l0_pt':    leading_lep_pt[baseline].flatten(),
            'l0_eta':   leading_lep_eta[baseline].flatten(), # this was the problem
            'l1_pt':    trailing_lep_pt[baseline].flatten(),
            'l1_eta':   trailing_lep_eta[baseline].flatten(), # this was the problem
            'st':       st[baseline].flatten(),
            'ht':       ht[baseline].flatten(),
            'njet':     jet.counts[baseline].flatten(),
            'nbtag':    btag.counts[baseline].flatten(),
            'met':      met_pt[baseline].flatten(),
            'mjj_max':  mjj_max[baseline].flatten(),
            'mlb_min':  mlb_min[baseline].flatten(),
            'mlb_max':  mlb_max[baseline].flatten(),
            'deltaR_lj_min': deltaR_lj_min[baseline].flatten(),
            'mll':      mll[baseline].flatten(),
            'signal':   signal_label,
            'weight':   df['weight'][baseline]
        })
        df_out.to_hdf('data/data_X.h5', key='df', format='table', mode='a', append=True)
        
        return output

    def postprocess(self, accumulator):
        return accumulator


# create empty df
df_out = {
    'j0_pt':    [],
    'j0_eta':   [],
    'l0_pt':    [],
    'l0_eta':   [],
    'l1_pt':    [],
    'l1_eta':   [],
    'st':       [],
    'ht':       [],
    'met':      [],
    'njet':     [],
    'nbtag':    [],
    'mjj_max':  [],
    'mlb_min':  [],
    'mlb_max':  [],
    'deltaR_lj_min': [],
    'mll':      [],
    'weight':   [],
    'signal':   [],
}
#df_out = {'spectator_pt': []}


small = False
overwrite = True

# load the config and the cache
cfg = loadConfig()

from processor.samples import fileset, fileset_small, fileset_2l, fileset_SS

if small:
    fileset = {'TTW': fileset_SS['TTW'][:2], 'ttbar':fileset_SS['ttbar'][:2]} # {'tW_scattering': fileset_small['tW_scattering']}
    workers = 4
else:
    fileset = {'TTW': fileset['TTW'], 'ttbar':fileset['ttbar1l']}
    workers = 4

if overwrite:

    # create .h5 file
    df = pd.DataFrame(df_out)
    df.to_hdf( 'data/data_X.h5', key='df', format='table', mode='w')

    output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=analysisProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': workers, 'function_args': {'flatten': False}},
                                      chunksize=500000,
                                     )
    
    check = output['passedEvents']['all'] == len(pd.read_hdf('data/data_X.h5'))
    print ("Analyzed events:", output['totalEvents']['all'])
    print ("Check passed:", check)

test = pd.read_hdf('data/data_X.h5')

print ("Got %s events for signal (ttW)."%len(test[test['signal']==1]))
print ("Got %s events for background (t/tt)."%len(test[test['signal']==0]))

#len(test[test['signal']==1])


