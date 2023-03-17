import os
import awkward as ak
from coffea.lookup_tools import extractor

class triggerSF:

    def __init__(self, year=2016):
        self.year = year
          
        triggerSF_2016 = os.path.expandvars("analysis/Tools/data/trigger/TriggerSF_2016.root")
        triggerSF_2017 = os.path.expandvars("analysis/Tools/data/trigger/TriggerSF_2017.root")
        triggerSF_2018 = os.path.expandvars("analysis/Tools/data/trigger/TriggerSF_2018.root")
        
        
        self.ext = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            self.ext.add_weight_sets([
                "mumu_2016 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2016,
                "mumu_2016_error h2D_SF_mumu_lepABpt_FullError_error %s"%triggerSF_2016,

                "emu_2016 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2016,
                "emu_2016_error h2D_SF_emu_lepABpt_FullError_error %s"%triggerSF_2016,

                "ee_2016 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2016,
                "ee_2016_error h2D_SF_ee_lepABpt_FullError_error %s"%triggerSF_2016,

            ])
            

        elif self.year == 2017:
            self.ext.add_weight_sets([
                "mumu_2017 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2017,
                "mumu_2017_error h2D_SF_mumu_lepABpt_FullError_error %s"%triggerSF_2017,

                "emu_2017 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2017,
                "emu_2017_error h2D_SF_emu_lepABpt_FullError_error %s"%triggerSF_2017,

                "ee_2017 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2017,
                "ee_2017_error h2D_SF_ee_lepABpt_FullError_error %s"%triggerSF_2017,

            ])
        elif self.year == 2018:
            self.ext.add_weight_sets([
                "mumu_2018 h2D_SF_mumu_lepABpt_FullError %s"%triggerSF_2018,
                "mumu_2018_error h2D_SF_mumu_lepABpt_FullError_error %s"%triggerSF_2018,

                "emu_2018 h2D_SF_emu_lepABpt_FullError %s"%triggerSF_2018,
                "emu_2018_error h2D_SF_emu_lepABpt_FullError_error %s"%triggerSF_2018,

                "ee_2018 h2D_SF_ee_lepABpt_FullError %s"%triggerSF_2018,
                "ee_2018_error h2D_SF_ee_lepABpt_FullError_error %s"%triggerSF_2018,

            ])


        self.ext.finalize()

        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu, variation='central'):
        multiplier = {'central': 0, 'up': 1, 'down': -1}

        from analysis.Tools.helpers import pad_and_flatten
        from analysis.Tools.objects import cross, choose
        import numpy as np
        # get a lepton collection
        lep = ak.concatenate([mu, ele], axis=1)
        # now sort them
        lep = lep[ak.argsort(lep.pt, ascending=False)]
        l0_is_ele = (abs(pad_and_flatten(lep[:,0:1].pdgId))==11)
        l1_is_ele = (abs(pad_and_flatten(lep[:,1:2].pdgId))==11)
                
        if self.year == 2016:
            
            ee_sf = self.evaluator["ee_2016"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            ee_sf = ee_sf + multiplier[variation]*self.evaluator["ee_2016_error"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            emu_sf = self.evaluator["emu_2016"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = emu_sf + multiplier[variation]*self.evaluator["emu_2016_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            mumu_sf = self.evaluator["mumu_2016"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))            
            mumu_sf = mumu_sf + multiplier[variation]*self.evaluator["mumu_2016_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            
        elif self.year == 2017:
                                          
            ee_sf = self.evaluator["ee_2017"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            ee_sf = ee_sf + multiplier[variation]*self.evaluator["ee_2017_error"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            emu_sf = self.evaluator["emu_2017"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = emu_sf + multiplier[variation]*self.evaluator["emu_2017_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            mumu_sf = self.evaluator["mumu_2017"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))            
            mumu_sf = mumu_sf + multiplier[variation]*self.evaluator["mumu_2017_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            
        elif self.year == 2018:
            
            ee_sf = self.evaluator["ee_2018"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            ee_sf = ee_sf + multiplier[variation]*self.evaluator["ee_2018_error"](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            emu_sf = self.evaluator["emu_2018"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
            emu_sf = emu_sf + multiplier[variation]*self.evaluator["emu_2018_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

            mumu_sf = self.evaluator["mumu_2018"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))            
            mumu_sf = mumu_sf + multiplier[variation]*self.evaluator["mumu_2018_error"](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
            

        sf = (ee_sf*(l0_is_ele&l1_is_ele)) + (emu_sf*(l0_is_ele^l1_is_ele)) + (mumu_sf*(~l0_is_ele&~l1_is_ele))
        #ee_mm_em = ak.concatenate([choose(ele), choose(mu), cross(mu, ele)], axis=1)
        #sf = (pad_and_flatten(sf*(ee_mm_em.pt/ee_mm_em.pt)))
        #sf = (sf/sf)*sf
        #sf = (np.where(np.isnan(sf), 1, sf))
        
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
        
    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.objects import Collections
    from Tools.config_helpers import loadConfig, make_small, load_yaml, data_path
    from Tools.helpers import get_samples
    from Tools.nano_mapping import make_fileset
    
    import awkward as ak

    samples = get_samples("samples_UL18.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    fileset = make_fileset(['TTW'], samples, year='UL18', skim=True, small=True, n_max=1)
    filelist = fileset[list(fileset.keys())[0]]

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        filelist[0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
    mu  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()

    sel = ((ak.num(el)+ak.num(mu))>1)

    sf_central  = sf18.get(el[sel], mu[sel])
    sf_up       = sf18.get(el[sel], mu[sel], variation='up')
    sf_down     = sf18.get(el[sel], mu[sel], variation='down')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    print ("Mean value of SF (up): %.3f"%ak.mean(sf_up))
    print ("Mean value of SF (down): %.3f"%ak.mean(sf_down))
