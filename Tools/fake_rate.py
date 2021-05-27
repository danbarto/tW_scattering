import os
import awkward as ak

from Tools.helpers import yahist_2D_lookup
import gzip
import pickle
 
class fake_rate:

    def __init__(self, path):
        self.path = path
        self.ratio = pickle.load(open(self.path, 'rb'))
#         with gzip.open(self.path) as fin:
#             self.ratio= pickle.load(fin)
    
    def FR_weight(self, lepton):
        f_1 = yahist_2D_lookup(self.ratio, lepton.conePt[:,0:1], abs(lepton.eta[:,0:1]))
        #breakpoint()
        # I'm using ak.prod and ak.sum to replace empty arrays by 1 and 0, respectively
        weight = ak.sum(f_1/(1-f_1), axis=1)#*ak.prod(1-f_2/(1-f_2), axis=1) + ak.sum(f_2/(1-f_2), axis=1)*ak.prod(1-f_1/(1-f_1), axis=1)

        return weight