import os
import awkward as ak
import numpy as np

from yahist import Hist1D
# FIXME there should be a correctionlib version of this now

class pileup:
    
    def __init__(self, year, UL=True):
        self.weight = {}

        if UL:
            if year == 2016:
                pass
            elif year == 2017:
                pass
            elif year == 2018:
                mc          = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_mc.json'))
                data        = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data.json'))
                data_up     = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data_up.json'))
                data_down   = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data_down.json'))
                
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
        return self.weight[to].lookup(ak.to_numpy(nvtx))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    pu = pileup(2018, UL=True)


    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.samples import get_babies
    from Tools.objects import Collections
    
    import awkward as ak
    
    fileset_all = get_babies('/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.3.3_dilep/', year='UL2018')
    
    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        fileset_all['TTW'][0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    print ("Weights from 1-100")
    pu_weight = pu.reweight(np.arange(0,100))
    print (pu_weight)

    print ("Example weights for some events.")
    pu_weight = pu.reweight(events.Pileup.nTrueInt)
    print (pu_weight)


