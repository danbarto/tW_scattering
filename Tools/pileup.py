import os
import awkward as ak
import numpy as np

import correctionlib

class pileup:
    
    def __init__(self, year, UL=True, era=None):
        self.weight = {}
        self.UL = UL

        if UL:
            if year == 2016:
                if era=='APV':
                    SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/LUM/2016preVFP_UL/')
                    self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "puWeights.json.gz"))['Collisions16_UltraLegacy_goldenJSON']

                else:
                    SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/LUM/2016postVFP_UL/')
                    self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "puWeights.json.gz"))['Collisions16_UltraLegacy_goldenJSON']
            elif year == 2017:
                SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/LUM/2017_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "puWeights.json.gz"))['Collisions17_UltraLegacy_goldenJSON']
            elif year == 2018:
                SF_file = os.path.expandvars('$TWHOME/jsonpog-integration/POG/LUM/2018_UL/')
                self.reader = correctionlib.CorrectionSet.from_file(os.path.join(SF_file, "puWeights.json.gz"))['Collisions18_UltraLegacy_goldenJSON']

        else:
            if year == 2016:
                mc          = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer16_mc.json'))
                data        = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer16_data.json'))
                data_up     = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer16_data_up.json'))
                data_down   = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer16_data_down.json'))
            elif year == 2017:
                mc          = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Fall17_mc.json'))
                data        = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Fall17_data.json'))
                data_up     = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Fall17_data_up.json'))
                data_down   = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Fall17_data_down.json'))
            elif year == 2018:
                mc          = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Autumn18_mc.json'))
                data        = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Autumn18_data.json'))
                data_up     = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Autumn18_data_up.json'))
                data_down   = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Autumn18_data_down.json'))

            self.weight['central']  = Hist1D.from_bincounts(np.nan_to_num(data.counts/mc.counts,nan=1,posinf=1, neginf=1), data.edges)
            self.weight['up']       = Hist1D.from_bincounts(np.nan_to_num(data_up.counts/mc.counts,nan=1,posinf=1, neginf=1), data.edges)
            self.weight['down']     = Hist1D.from_bincounts(np.nan_to_num(data_down.counts/mc.counts,nan=1,posinf=1, neginf=1), data.edges)

    def reweight(self, nvtx, to='central'):
        if self.UL:
            if to == 'central': to = 'nominal'
            return self.reader.evaluate(nvtx, to)
        else:
            return self.weight[to].lookup(ak.to_numpy(nvtx))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    pu = pileup(2018, UL=True)

    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.helpers import get_samples
    from Tools.objects import Collections
    
    import awkward as ak
    samples = get_samples("samples_UL18.yaml")
    f_in    = samples['/ceph/cms/store/user/dspitzba/ProjectMetis/TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO_RunIISummer20UL18_NanoAODv9_NANO_v14/']['files'][0]

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        f_in,
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    print ("Weights from 1-100")
    pu_weight = pu.reweight(np.arange(0,100))
    print (pu_weight)

    print ("Example weights for some events.")
    pu_weight = pu.reweight(events.Pileup.nTrueInt.to_numpy())
    print (pu_weight)


