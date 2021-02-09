import awkward1 as ak
import os

from coffea.btag_tools import BTagScaleFactor
from yahist import Hist1D, Hist2D

def doAwkwardLookup(h, ar):
    '''
    takes a ya_hist histogram (which has a lookup function) and an awkward array.
    '''
    return ak.unflatten(
        h.lookup(
            ak.to_numpy(
                ak.flatten(ar)
            ) 
        ), ak.num(ar) )



class btag_scalefactor:
    def __init__(self, year):
        self.year = year

        if self.year == 2016:
            pass
        elif self.year == 2017:
            pass
        elif self.year == 2018:
            SF_file = os.path.expandvars('$TWHOME/data/btag/DeepJet_102XSF_V2.csv')
            self.btag_sf = BTagScaleFactor(SF_file, "medium", keep_df=False)

            # and load the efficiencies
            self.effs = {
                'b': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_b_eff_deepJet.json")),
                'c': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_c_eff_deepJet.json")),
                'light': Hist1D.from_json(os.path.expandvars("$TWHOME/data/btag/Autumn18_light_eff_deepJet.json")),
            }
            
    def Method1a(self, tagged, untagged):
        '''
        tagged: jet collection of tagged jets
        untagged: jet collection untagged jets
        effs: dictionary of the tagging efficiencies (1D yahist objects)
        btag_sf: coffea b-tag SF object
        '''
        tagged_b = doAwkwardLookup(self.effs['b'], tagged.pt)*(tagged.hadronFlavour==5)
        tagged_c = doAwkwardLookup(self.effs['c'], tagged.pt)*(tagged.hadronFlavour==4)
        tagged_light = doAwkwardLookup(self.effs['light'], tagged.pt)*(tagged.hadronFlavour==0)
        
        tagged_SFs = self.btag_sf.eval('central', tagged.hadronFlavour, abs(tagged.eta), tagged.pt )
        
        untagged_b = doAwkwardLookup(self.effs['b'], untagged.pt)*(untagged.hadronFlavour==5)
        untagged_c = doAwkwardLookup(self.effs['c'], untagged.pt)*(untagged.hadronFlavour==4)
        untagged_light = doAwkwardLookup(self.effs['light'], untagged.pt)*(untagged.hadronFlavour==0)
        
        untagged_SFs = self.btag_sf.eval('central', untagged.hadronFlavour, abs(untagged.eta), untagged.pt )
        
        tagged_all = (tagged_b+tagged_c+tagged_light)
        untagged_all = (untagged_b+untagged_c+untagged_light)
        
        denom = ak.prod(tagged_all, axis=1) * ak.prod((1-untagged_all), axis=1)
        num = ak.prod(tagged_all*tagged_SFs, axis=1) * ak.prod((1-untagged_all*untagged_SFs), axis=1)
        return num/denom
