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
                SF_file = os.path.expandvars('analysis/Tools/jsonpog-integration/POG/TAU/2016preVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

            else:
                # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16postVFP
                SF_file = os.path.expandvars('analysis/Tools/jsonpog-integration/POG/TAU/2016postVFP_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        elif self.year == 2017:
            SF_file = os.path.expandvars('analysis/Tools/jsonpog-integration/POG/TAU/2017_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        elif self.year == 2018:
            SF_file = os.path.expandvars('analysis/Tools/jsonpog-integration/POG/TAU/2018_UL/')
            self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "tau.json.gz"))

        self.ext.finalize()
        self.effs = self.ext.make_evaluator()


    def lookup(self, pt, decay_mode, genmatch, var='nom', WP='Loose'):
        # sf1 = corr1.evaluate(pt,dm,1,wp,"nom","pt")
        return ak.unflatten(
            self.reader["DeepTau2017v2p1VSjet"].evaluate(
                ak.to_numpy(ak.flatten(pt)),
                ak.to_numpy(ak.flatten(decay_mode)),
                ak.to_numpy(ak.flatten(genmatch)),
                WP,
                var,
                "pt",
                ),
            ak.num(pt),
        )

    def get(self, tau, var='nom', WP='Loose'):
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
    import hist

    from analysis.Tools.basic_objects import getTaus
    from analysis.Tools.samples import Samples

    samples = Samples.from_yaml(f'analysis/Tools/data/samples_v0_8_0_SS.yaml')

    fileset = samples.get_fileset(year='UL17', group='TTW')

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        fileset[list(fileset.keys())[0]][0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    tau     = getTaus(events)

    import time

    start_time = time.time()
    print ("Using correctionlib")
    sf_central  = sf17.get(tau[(ak.num(tau, axis=1)>0)], var='nom', WP='Loose')
    sf_up       = sf17.get(tau[(ak.num(tau, axis=1)>0)], var='up', WP='Loose')
    sf_down     = sf17.get(tau[(ak.num(tau, axis=1)>0)], var='down', WP='Loose')
    print ("Took %.2f s"%(time.time()-start_time))

    weight_axis = hist.axis.Regular(30, 0.8, 1.1, name="weight_ax", label="weight", underflow=True, overflow=True)

    h_central = hist.Hist(weight_axis)
    h_central.fill(sf_central)
    h_central.show()

    h_up = hist.Hist(weight_axis)
    h_up.fill(sf_up)

    h_down = hist.Hist(weight_axis)
    h_down.fill(sf_down)
