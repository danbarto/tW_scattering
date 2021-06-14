import os
import awkward as ak

from yahist import Hist1D

class pileup:
    
    def __init__(self, year):
        if year == 2016:
            pass
        elif year == 2017:
            pass
        elif year == 2018:
            self.weight = {}
            
            mc          = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_mc.json'))
            data        = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data.json'))
            data_up     = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data_up.json'))
            data_down   = Hist1D.from_json(os.path.expandvars('$TWHOME/data/PU/Summer20UL18_data_down.json'))
            
            self.weight['central']  = data.divide(mc)
            self.weight['up']       = data_up.divide(mc)
            self.weight['down']     = data_down.divide(mc)

    def reweight(self, nvtx, to='central'):
        return self.weight[to].lookup(nvtx)
