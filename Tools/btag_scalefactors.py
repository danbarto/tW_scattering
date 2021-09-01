try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

import os

from coffea.btag_tools import BTagScaleFactor
from yahist import Hist1D, Hist2D

from Tools.helpers import yahist_1D_lookup


class btag_scalefactor:
    def __init__(self, year, UL=True):
        self.year = year

        if self.year == 2016:
            SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_106XUL17SF_WPonly_V2.csv')  #FIXME. What to do with APV?
            self.btag_sf = BTagScaleFactor(SF_file, "medium", keep_df=False)

            # and load the efficiencies
            self.effs = {
                'b':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL16_b_eff_deepJet.json")),
                'c':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL16_c_eff_deepJet.json")),
                'light': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL16_light_eff_deepJet.json")),
            }
        elif self.year == 2017:
            SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_106XUL17SF_WPonly_V2.csv')
            #SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_106XUL17SF_V2.csv')
            self.btag_sf = BTagScaleFactor(SF_file, "medium", keep_df=False)

            # and load the efficiencies
            self.effs = {
                'b':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL17_b_eff_deepJet.json")),
                'c':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL17_c_eff_deepJet.json")),
                'light': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL17_light_eff_deepJet.json")),
            }
        elif self.year == 2018 and UL:
            SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_106XUL18SF_WPonly.csv')
            self.btag_sf = BTagScaleFactor(SF_file, "medium", keep_df=False)

            # and load the efficiencies
            self.effs = {
                'b':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL18_b_eff_deepJet.json")),
                'c':     Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL18_c_eff_deepJet.json")),
                'light': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Summer20UL18_light_eff_deepJet.json")),
            }
        elif self.year == 2018 and not UL:
            SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_102XSF_V2.csv')
            self.btag_sf = BTagScaleFactor(SF_file, "medium", keep_df=False)

            # and load the efficiencies
            self.effs = {
                'b': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_b_eff_deepJet.json")),
                'c': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_c_eff_deepJet.json")),
                'light': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_light_eff_deepJet.json")),
            }
            
   
    def Method1a(self, tagged, untagged, b_direction='central', c_direction='central'):
        import numpy as np
        '''
        tagged: jet collection of tagged jets
        untagged: jet collection untagged jets
        effs: dictionary of the tagging efficiencies (1D yahist objects)
        btag_sf: coffea b-tag SF object
        '''
        tagged_b = yahist_1D_lookup(self.effs['b'], tagged.pt)*(tagged.hadronFlavour==5)
        tagged_c = yahist_1D_lookup(self.effs['c'], tagged.pt)*(tagged.hadronFlavour==4)
        tagged_light = yahist_1D_lookup(self.effs['light'], tagged.pt)*(tagged.hadronFlavour==0)
        
        tagged_SFs_b = self.btag_sf.eval(b_direction, tagged.hadronFlavour, abs(tagged.eta), tagged.pt )
        tagged_SFs_c = self.btag_sf.eval(c_direction, tagged.hadronFlavour, abs(tagged.eta), tagged.pt )
        tagged_SFs_light = self.btag_sf.eval(c_direction, tagged.hadronFlavour, abs(tagged.eta), tagged.pt )
        
        SFs_c = ((tagged_c/tagged_c)*tagged_SFs_c)
        SFs_b = ((tagged_b/tagged_b)*tagged_SFs_b)
        SFs_light = ((tagged_light/tagged_light)*tagged_SFs_light)
        SFs_c = np.where(np.isnan(SFs_c), 0, SFs_c)
        SFs_b = np.where(np.isnan(SFs_b), 0, SFs_b)
        SFs_light = np.where(np.isnan(SFs_light), 0, SFs_light)
        
        tagged_SFs = SFs_b+SFs_c+SFs_light
        
        untagged_b = yahist_1D_lookup(self.effs['b'], untagged.pt)*(untagged.hadronFlavour==5)
        untagged_c = yahist_1D_lookup(self.effs['c'], untagged.pt)*(untagged.hadronFlavour==4)
        untagged_light = yahist_1D_lookup(self.effs['light'], untagged.pt)*(untagged.hadronFlavour==0)
        
        untagged_SFs_b = self.btag_sf.eval(b_direction, untagged.hadronFlavour, abs(untagged.eta), untagged.pt )
        untagged_SFs_c = self.btag_sf.eval(c_direction, untagged.hadronFlavour, abs(untagged.eta), untagged.pt )
        untagged_SFs_light = self.btag_sf.eval(c_direction, untagged.hadronFlavour, abs(untagged.eta), untagged.pt )
        
        SFs_c = ((untagged_c/untagged_c)*untagged_SFs_c)
        SFs_b = ((untagged_b/untagged_b)*untagged_SFs_b)
        SFs_light = ((untagged_light/untagged_light)*untagged_SFs_light)
        SFs_c = np.where(np.isnan(SFs_c), 0, SFs_c)
        SFs_b = np.where(np.isnan(SFs_b), 0, SFs_b)
        SFs_light = np.where(np.isnan(SFs_light), 0, SFs_light)
        
        untagged_SFs = SFs_b+SFs_c+SFs_light
                                
        tagged_all = (tagged_b+tagged_c+tagged_light)
        untagged_all = (untagged_b+untagged_c+untagged_light)
        
        denom = ak.prod(tagged_all, axis=1) * ak.prod((1-untagged_all), axis=1)
        num = ak.prod(tagged_all*tagged_SFs, axis=1) * ak.prod((1-untagged_all*untagged_SFs), axis=1)
        return num/denom

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16 = btag_scalefactor(year=2016)
    sf17 = btag_scalefactor(year=2017)
    sf18 = btag_scalefactor(year=2018)
    
    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.samples import get_babies
    from Tools.basic_objects import getJets, getBTagsDeepFlavB
    
    import awkward as ak
    
    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.3_dilep/', year='UL2017')
    
    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        fileset_all['TTW'][0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    jet     = getJets(events, minPt=25, maxEta=2.4, pt_var='pt_nom')
    btag    = getBTagsDeepFlavB(jet, year=2017)
    light   = getBTagsDeepFlavB(jet, year=2017, invert=True)

    sf_central = sf17.Method1a(btag, light)
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    #sf_up       = sf18.get(el[sel], mu[sel], variation='up')
    #print ("Mean value of SF (up): %.3f"%ak.mean(sf_up))
    #sf_down     = sf18.get(el[sel], mu[sel], variation='down')
    #print ("Mean value of SF (down): %.3f"%ak.mean(sf_down))

