import os
import awkward as ak
from coffea.lookup_tools import extractor

class triggerSF:

    def __init__(self, year=2016):
        self.year = year
          
        triggerSF_2016 = os.path.expandvars("$TWHOME/data/trigger/TriggerSF_2016.root")
        triggerSF_2017 = os.path.expandvars("$TWHOME/data/trigger/TriggerSF_2017.root")
        triggerSF_2018 = os.path.expandvars("$TWHOME/data/trigger/TriggerSF_2018.root")
        
        
        self.ext = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            self.ext.add_weight_sets(["mumu_2016 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2016])
            
            self.ext.add_weight_sets(["emu_2016 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2016])
            
            self.ext.add_weight_sets(["ee_2016 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2016])
            

        elif self.year == 2017:
            self.ext.add_weight_sets(["mumu_2017 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2017])
            
            self.ext.add_weight_sets(["emu_2017 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2017])
            
            self.ext.add_weight_sets(["ee_2017 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2017])

        elif self.year == 2018:
            self.ext.add_weight_sets(["mumu_2018 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2018])
            
            self.ext.add_weight_sets(["emu_2018 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2018])
            
            self.ext.add_weight_sets(["ee_2018 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2018])


        self.ext.finalize()

        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu): 
        from Tools.helpers import pad_and_flatten
        from Tools.objects import cross, choose 
        import numpy as np
        lep = ak.concatenate([mu, ele], axis=1)
                
        if self.year == 2016:
            
            ee_sf = self.evaluator["ee_2016"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = self.evaluator["emu_2016"](pad_and_flatten(lep[:,0:1].pt), pad_and_flatten(lep[:,1:2].pt))
            mumu_sf = self.evaluator["mumu_2016"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            
            sf = ee_sf * emu_sf * mumu_sf
   

        elif self.year == 2017:
                                          
            ee_sf = self.evaluator["ee_2017"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = self.evaluator["emu_2017"](pad_and_flatten(lep[:,0:1].pt), pad_and_flatten(lep[:,1:2].pt))
            mumu_sf = self.evaluator["mumu_2017"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            
            sf = ee_sf * emu_sf * mumu_sf
               
        elif self.year == 2018:
            
            ee_sf = self.evaluator["ee_2018"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = self.evaluator["emu_2018"](pad_and_flatten(lep[:,0:1].pt), pad_and_flatten(lep[:,1:2].pt))
            mumu_sf = self.evaluator["mumu_2018"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            
            sf = ee_sf * emu_sf * mumu_sf 
                
        ee_mm_em = ak.concatenate([choose(ele), choose(mu), cross(mu, ele)], axis=1)
        sf = (pad_and_flatten(sf*(ee_mm_em.pt/ee_mm_em.pt)))
        sf = (sf/sf)*sf
        sf = (np.where(np.isnan(sf), 1, sf))
        
        return sf

    def values(self):

        return 0



if __name__ == '__main__':
    sf16 = triggerSF(year=2016)
    sf17 = triggerSF(year=2017)
    sf18 = triggerSF(year=2018)
    
    

    print("Evaluators found for 2016:")
    for key in sf16.evaluator.keys():
        print("%s:"%key, sf16.evaluator[key])

    print("Evaluators found for 2017:")
    for key in sf17.evaluator.keys():
        print("%s:"%key, sf17.evaluator[key])

    print("Evaluators found for 2018:")
    for key in sf18.evaluator.keys():
        print("%s:"%key, sf18.evaluator[key])
        
        
