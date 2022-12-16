try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

import os

import correctionlib
from coffea.lookup_tools import extractor
import numpy as np

class tau_scalefactor:
    def __init__(self, year, era=None, UL=True):
        self.year = year

        self.ext = extractor()

        if self.year == 2016:
            if era=='APV':
                # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16preVFP
                SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/TAU/2016preVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

            else:
                # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16postVFP
                SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/TAU/2016postVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        elif self.year == 2017:
            SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/TAU/2017_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        elif self.year == 2018:
            SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/TAU/2018_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        self.ext.finalize()
        self.effs = self.ext.make_evaluator()


    def lookup(self, pt, decay_mode, genmatch, var='nom', WP='loose'):
        # sf1 = corr1.evaluate(pt,dm,1,wp,"nom","pt")
        return ak.unflatten(
            self.reader["DeepTau2017v2p1VSjet"].evaluate(
                pt,
                decay_mode,
                genmatch,
                WP,
                var,
                "pt",
                )
        )

    def get(self, tau, var='nom', WP='loose'):
        return ak.prod(
            self.lookup(tau.pt, tau.decayMode, tau.genPartFlav, var=var, WP=WP),
            axis=1,
        )

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16 = tau_scalefactor(year=2016, era='APV')
    sf16 = tau_scalefactor(year=2016)
    sf17 = tau_scalefactor(year=2017)
    sf18 = tau_scalefactor(year=2018)
    
    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.helpers import get_samples
    from Tools.basic_objects import getTaus
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

    tau     = getTaus(events)

    import time

    start_time = time.time()
    print ("Using correctionlib")
    sf_central = sf17.get(tau, var='nom', WP='loose')
    print ("Took %.2f s"%(time.time()-start_time))
