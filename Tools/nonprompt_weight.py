
import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor

class NonpromptWeight:

    def __init__(self, year=2016):
        self.year = year

        self.ext = extractor()
        if self.year == 2016:
            fr = os.path.expandvars("$TWHOME/data/fakerate/fr_2016.root")

        elif self.year == 2017:
            fr = os.path.expandvars("$TWHOME/data/fakerate/fr_2017.root")

        elif self.year == 2018:
            fr = os.path.expandvars("$TWHOME/data/fakerate/fr_2018.root")

        self.ext.add_weight_sets(["el_QCD FR_mva080_el_QCD %s"%fr])
        self.ext.add_weight_sets(["el_QCD_NC FR_mva080_el_QCD_NC %s"%fr])
        self.ext.add_weight_sets(["el_TT FR_mva080_el_TT %s"%fr])
        self.ext.add_weight_sets(["mu_QCD FR_mva085_mu_QCD %s"%fr])
        self.ext.add_weight_sets(["mu_TT FR_mva085_mu_TT %s"%fr])

        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, el, mu, meas='QCD'):

        if meas == 'QCD':
            el_key, mu_key = 'el_QCD_NC', 'mu_QCD'
        elif meas == 'TT':
            el_key, mu_key = 'el_TT', 'mu_TT'

        n_lep   = ak.num(el) + ak.num(mu)
        sign    = (-1)**(n_lep+1)
        el_fr   = evaluator[el_key](el.conePt, np.abs(el.etaSC))
        mu_fr   = evaluator[mu_key](mu.conePt, np.abs(mu.eta))
        fr      = ak.concatenate([el_fr, mu_fr], axis=1)
        return ak.prod(fr, axis=1)*sign


if __name__ == '__main__':
    sf16 = NonpromptWeight(year=2016)
    sf17 = NonpromptWeight(year=2017)
    sf18 = NonpromptWeight(year=2018)
    
    print("Evaluators found for 2016:")
    for key in sf16.evaluator.keys():
        print("%s:"%key, sf16.evaluator[key])

    print("Evaluators found for 2017:")
    for key in sf17.evaluator.keys():
        print("%s:"%key, sf17.evaluator[key])

    print("Evaluators found for 2018:")
    for key in sf18.evaluator.keys():
        print("%s:"%key, sf18.evaluator[key])

