import os

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from Tools.helpers import yahist_2D_lookup
import gzip
import pickle
 
class charge_flip:

    def __init__(self, path):
        self.path = path
        with gzip.open(self.path) as fin:
            self.ratio= pickle.load(fin)
    
    def flip_weight(self, electron):

        f_1 = yahist_2D_lookup(self.ratio, electron.pt[:,0:1], abs(electron.eta[:,0:1]))
        f_2 = yahist_2D_lookup(self.ratio, electron.pt[:,1:2], abs(electron.eta[:,1:2]))

        # I'm using ak.prod and ak.sum to replace empty arrays by 1 and 0, respectively
        weight = ak.sum(f_1/(1-f_1), axis=1)*ak.prod(1-f_2/(1-f_2), axis=1) + ak.sum(f_2/(1-f_2), axis=1)*ak.prod(1-f_1/(1-f_1), axis=1)

        return weight
