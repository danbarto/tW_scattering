import os
import numpy as np
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor

class NonpromptWeight:

    def __init__(self, year=2016):
        if year == 2016:
            self.year = '2016APV_2016'
        else:
            self.year = year

        self.ext = extractor()

        # FIXME those need updates! Do we still have MC based FRs for closure tests?
        fr = os.path.expandvars("$TWHOME/data/leptons/ttH/fakerate/fr_%s.root"%self.year)
        fr_data = os.path.expandvars("$TWHOME/data/leptons/ttH/fakerate/fr_%s_recorrected.root"%self.year)

        self.ext.add_weight_sets(["el_QCD FR_mva090_el_QCD %s"%fr])
        self.ext.add_weight_sets(["el_QCD_NC FR_mva090_el_QCD_NC %s"%fr])
        self.ext.add_weight_sets(["el_TT FR_mva090_el_TT %s"%fr])
        self.ext.add_weight_sets(["el_data FR_mva090_el_data_comb_NC_recorrected %s"%fr_data])
        self.ext.add_weight_sets(["mu_QCD FR_mva085_mu_QCD %s"%fr])
        self.ext.add_weight_sets(["mu_TT FR_mva085_mu_TT %s"%fr])
        self.ext.add_weight_sets(["mu_data FR_mva085_mu_data_comb_recorrected %s"%fr_data])

        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, el, mu, meas='QCD'):

        if meas == 'QCD':
            el_key, mu_key = 'el_QCD_NC', 'mu_QCD'
        elif meas == 'TT':
            el_key, mu_key = 'el_TT', 'mu_TT'
        elif meas == 'data':
            el_key, mu_key = 'el_data', 'mu_data'

        n_lep   = ak.num(el) + ak.num(mu)
        sign    = (-1)**(n_lep+1)
        el_fr   = self.evaluator[el_key](el.conePt, np.abs(el.etaSC))
        mu_fr   = self.evaluator[mu_key](mu.conePt, np.abs(mu.eta))
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

