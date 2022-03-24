import os

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor
#from Tools.helpers import yahist_2D_lookup
#import gzip
#import pickle
 
class charge_flip:

    def __init__(self, year=2016):

        self.year = year

        self.ext = extractor()

        #FIXME these charge flip rates are still for EOY
        fr = os.path.expandvars("$TWHOME/data/chargeflip/ElectronChargeMisIdRates_era%s_2020Feb13.root"%self.year)

        self.ext.add_weight_sets(["el chargeMisId %s"%fr])

        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

        # For custom measurements
        #self.path = path
        #with gzip.open(self.path) as fin:
        #    self.ratio= pickle.load(fin)
    
    def flip_weight(self, electron):

        f_1 = self.evaluator['el'](electron.pt[:,0:1], abs(electron.eta[:,0:1]))
        f_2 = self.evaluator['el'](electron.pt[:,1:2], abs(electron.eta[:,1:2]))

        # For custom measurements
        #f_1 = yahist_2D_lookup(self.ratio, electron.pt[:,0:1], abs(electron.eta[:,0:1]))
        #f_2 = yahist_2D_lookup(self.ratio, electron.pt[:,1:2], abs(electron.eta[:,1:2]))

        # I'm using ak.prod and ak.sum to replace empty arrays by 1 and 0, respectively
        weight = ak.sum(f_1/(1-f_1), axis=1)*ak.prod(1-f_2/(1-f_2), axis=1) + ak.sum(f_2/(1-f_2), axis=1)*ak.prod(1-f_1/(1-f_1), axis=1)

        return weight



if __name__ == '__main__':
    sf16 = charge_flip(year=2016)
    sf17 = charge_flip(year=2017)
    sf18 = charge_flip(year=2018)
    
    print("Evaluators found for 2016:")
    for key in sf16.evaluator.keys():
        print("%s:"%key, sf16.evaluator[key])

    print("Evaluators found for 2017:")
    for key in sf17.evaluator.keys():
        print("%s:"%key, sf17.evaluator[key])

    print("Evaluators found for 2018:")
    for key in sf18.evaluator.keys():
        print("%s:"%key, sf18.evaluator[key])

