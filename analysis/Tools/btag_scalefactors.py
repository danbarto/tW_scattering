try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

import os

import correctionlib
from coffea.lookup_tools import extractor
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

class btag_scalefactor:
    def __init__(self, year, era=None, UL=True):
        self.year = year

        self.ext = extractor()

        if self.year == 2016:
            if era=='APV':
                # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16preVFP
                SF_file = os.path.join(here, 'jsonpog-integration/POG/BTV/2016preVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "btagging.json.gz"))

                # and load the efficiencies
                self.ext.add_weight_sets([f"* * {here}/data/btag/deepJet_eff_Summer20UL16APV.json"])

            else:
                # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16postVFP
                SF_file = os.path.join(here, 'jsonpog-integration/POG/BTV/2016postVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "btagging.json.gz"))

                # and load the efficiencies
                self.ext.add_weight_sets([f"* * {here}/data/btag/deepJet_eff_Summer20UL16.json"])

        elif self.year == 2017:
            SF_file = os.path.join(here, 'jsonpog-integration/POG/BTV/2017_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "btagging.json.gz"))

            # and load the efficiencies
            self.ext.add_weight_sets([f"* * {here}/data/btag/deepJet_eff_Summer20UL17.json"])

        elif self.year == 2018:
            SF_file = os.path.join(here, 'jsonpog-integration/POG/BTV/2018_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "btagging.json.gz"))

            # and load the efficiencies
            self.ext.add_weight_sets([f"* * {here}/data/btag/deepJet_eff_Summer20UL18.json"])

        self.ext.finalize()
        self.effs = self.ext.make_evaluator()


    def lookup(self, flav, eta, pt, var='central', WP='M', light=True):
        postfix = 'incl' if light else 'mujets'
        return ak.unflatten(
            self.reader["deepJet_%s"%postfix].evaluate(
                var,
                WP,
                ak.to_numpy(ak.flatten(flav)),
                np.abs(ak.to_numpy(ak.flatten(eta))),
                ak.to_numpy(ak.flatten(pt)),
            ),
            ak.num(flav),
        )


    def Method1a(self, tagged, untagged, b_direction='central', c_direction='central'):
        import numpy as np
        '''
        tagged: jet collection of tagged jets
        untagged: jet collection untagged jets
        https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods
        '''
        # slightly changing structure because of the json correctionlib
        tagged_b = tagged[tagged.hadronFlavour==5]
        tagged_c = tagged[tagged.hadronFlavour==4]
        tagged_light = tagged[tagged.hadronFlavour==0]

        untagged_b = untagged[untagged.hadronFlavour==5]
        untagged_c = untagged[untagged.hadronFlavour==4]
        untagged_light = untagged[untagged.hadronFlavour==0]

        tagged_eff_b = self.effs['eff/b_value'](tagged_b.pt)
        tagged_eff_c = self.effs['eff/c_value'](tagged_c.pt)
        tagged_eff_light = self.effs['eff/light_value'](tagged_light.pt)

        tagged_SFs_b = self.lookup(
            tagged_b.hadronFlavour,
            abs(tagged_b.eta),
            tagged_b.pt,
            var = b_direction,
            light = False,
        )

        tagged_SFs_c = self.lookup(
            tagged_c.hadronFlavour,
            abs(tagged_c.eta),
            tagged_c.pt,
            var = c_direction,
            light = False,
        )

        tagged_SFs_light = self.lookup(
            tagged_light.hadronFlavour,
            abs(tagged_light.eta),
            tagged_light.pt,
            var = c_direction,
            light = True,
        )

        tagged_SFs = ak.concatenate([tagged_SFs_b, tagged_SFs_c, tagged_SFs_light], axis=1)
        
        untagged_eff_b = self.effs['eff/b_value'](untagged_b.pt)
        untagged_eff_c = self.effs['eff/c_value'](untagged_c.pt)
        untagged_eff_light = self.effs['eff/light_value'](untagged_light.pt)

        untagged_SFs_b = self.lookup(
            untagged_b.hadronFlavour,
            abs(untagged_b.eta),
            untagged_b.pt,
            var = b_direction,
            light = False,
        )

        untagged_SFs_c = self.lookup(
            untagged_c.hadronFlavour,
            abs(untagged_c.eta),
            untagged_c.pt,
            var = c_direction,
            light = False,
        )

        untagged_SFs_light = self.lookup(
            untagged_light.hadronFlavour,
            abs(untagged_light.eta),
            untagged_light.pt,
            var = c_direction,
            light = True,
        )

        untagged_SFs = ak.concatenate([untagged_SFs_b, untagged_SFs_c, untagged_SFs_light], axis=1)

        tagged_eff_all = ak.concatenate([tagged_eff_b, tagged_eff_c, tagged_eff_light], axis=1)
        untagged_eff_all = ak.concatenate([untagged_eff_b, untagged_eff_c, untagged_eff_light], axis=1)

        denom = ak.prod(tagged_eff_all, axis=1) * ak.prod((1-untagged_eff_all), axis=1)
        num = ak.prod(tagged_eff_all*tagged_SFs, axis=1) * ak.prod((1-untagged_eff_all*untagged_SFs), axis=1)
        return num/denom

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16 = btag_scalefactor(year=2016, era='APV')
    sf16 = btag_scalefactor(year=2016)
    sf17 = btag_scalefactor(year=2017)
    sf18 = btag_scalefactor(year=2018)
    
    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.helpers import get_samples
    from Tools.basic_objects import getJets, getBTagsDeepFlavB
    from Tools.config_helpers import loadConfig, make_small, load_yaml, data_path
    from Tools.nano_mapping import make_fileset

    samples = get_samples("samples_UL18.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    fileset = make_fileset(
        ['TTW'],
        samples,
        year='UL18',
        skim='topW_v0.7.0_dilep',
        small=True,
        buaf='local',
        merged=True,
        n_max=1)
    filelist = fileset[list(fileset.keys())[0]]

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        filelist[0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    jet     = getJets(events, minPt=25, maxEta=2.4, pt_var='pt_nom')
    btag    = getBTagsDeepFlavB(jet, year=2017)
    light   = getBTagsDeepFlavB(jet, year=2017, invert=True)

    import time
    #start_time = time.time()
    #print ("Using old csv reader")
    #sf_central = sf17.Method1a_old(btag, light)
    #print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    #print ("Took %.2f s"%(time.time()-start_time))

    start_time = time.time()
    print ("Using correctionlib")
    sf_central = sf17.Method1a(btag, light)
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    print ("Took %.2f s"%(time.time()-start_time))
