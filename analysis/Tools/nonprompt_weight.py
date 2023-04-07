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

        fr = os.path.expandvars("analysis/Tools/data/leptons/ttH/fakerate/fr_%s.root"%self.year)
        fr_data = os.path.expandvars("analysis/Tools/data/leptons/ttH/fakerate/fr_%s_recorrected.root"%self.year)

        self.ext.add_weight_sets(["el_QCD FR_mva090_el_QCD %s"%fr])
        self.ext.add_weight_sets(["el_QCD_NC FR_mva090_el_QCD_NC %s"%fr])
        self.ext.add_weight_sets(["el_TT FR_mva090_el_TT %s"%fr])
        self.ext.add_weight_sets(["el_data FR_mva090_el_data_comb_NC_recorrected %s"%fr_data])
        self.ext.add_weight_sets(["mu_QCD FR_mva085_mu_QCD %s"%fr])
        self.ext.add_weight_sets(["mu_TT FR_mva085_mu_TT %s"%fr])
        self.ext.add_weight_sets(["mu_data FR_mva085_mu_data_comb_recorrected %s"%fr_data])

        for syst in ['_up', '_down', '_be1', '_be2', '_pt1', '_pt2']:
            self.ext.add_weight_sets([f"el_data{syst} FR_mva090_el_data_comb_NC_recorrected{syst} {fr_data}"])
            self.ext.add_weight_sets([f"mu_data{syst} FR_mva085_mu_data_comb_recorrected{syst} {fr_data}"])

        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def closure_syst_el(self, el):
        if self.year == '2016APV_2016':
            return ( (np.abs(el.eta) > 1.5)*0.5 + (np.abs(el.eta) < 1.5)*0.1) + 1.0
        elif self.year == 2017:
            return (np.abs(el.eta) < 3.5)*0.2 + 1.0  # NOTE just need an array that has the same shape as all the electrons
        elif self.year == 2018:
            return ( (np.abs(el.eta) > 1.5)*0.5 + (np.abs(el.eta) < 1.5)*0.1) + 1.0
        else:
            print (f"Don't know what to do with era {self.year}")
            return False

    def closure_syst_mu(self, mu):
        if self.year == '2016APV_2016':
            return (np.abs(mu.eta) < 3.5)*0.05 + 1.0  # NOTE just need an array that has the same shape as all the electrons
        elif self.year == 2017:
            return (np.abs(mu.eta) < 3.5)*0.2 + 1.0  # NOTE just need an array that has the same shape as all the electrons
        elif self.year == 2018:
            return (np.abs(mu.eta) < 3.5)*0.05 + 1.0  # NOTE just need an array that has the same shape as all the electrons
        else:
            print (f"Don't know what to do with era {self.year}")
            return False

    def get(self, el, mu, meas='QCD', variation=''):
        # variation should be el_up, el_be1 etc

        if meas == 'data' or variation:
            el_key, mu_key = 'el_data', 'mu_data'
        elif meas == 'QCD':
            el_key, mu_key = 'el_QCD_NC', 'mu_QCD'
        elif meas == 'TT':
            el_key, mu_key = 'el_TT', 'mu_TT'


        if variation.count('el') and not variation.count('closure'):
            el_key += variation.strip('el')
        elif variation.count('mu') and not variation.count('closure'):
            mu_key += variation.strip('mu')

        n_lep   = ak.num(el) + ak.num(mu)
        sign    = (-1)**(n_lep+1)
        el_fr   = self.evaluator[el_key](el.conePt, np.abs(el.etaSC))
        mu_fr   = self.evaluator[mu_key](mu.conePt, np.abs(mu.eta))

        el_mult = 1
        if variation == 'el_closure_up':
            el_mult = self.closure_syst_el(el)
        elif variation == 'el_closure_down':
            el_mult = 1/self.closure_syst_el(el)

        mu_mult = 1
        if variation == 'mu_closure_up':
            mu_mult = self.closure_syst_mu(mu)
        elif variation == 'mu_closure_down':
            mu_mult = 1/self.closure_syst_mu(mu)

        fr      = ak.concatenate([el_fr/(1-el_fr)*el_mult, mu_fr/(1-mu_fr)*mu_mult], axis=1)
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

    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.objects import Collections

    from Tools.helpers import get_samples
    from Tools.basic_objects import getJets, getBTagsDeepFlavB
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

    el_f  = Collections(events, 'Electron', 'fakeableSSTTH', verbose=1).get()
    el_t  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
#    el    = ak.concatenate([el_f, el_t], axis=1)
    el    = el_f
    mu_f  = Collections(events, 'Muon', 'fakeableSSTTH', verbose=1).get()
    mu_t  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()
#    mu    = ak.concatenate([mu_f, mu_t], axis=1)
    mu    = mu_f

    sel = (((ak.num(el_t)+ak.num(mu_t))==1) & ((ak.num(el_f)+ak.num(mu_f))==1))

    sf_central  = sf18.get(el[sel], mu[sel], variation='', meas='data')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    print ()
    sf_up       = sf18.get(el[sel], mu[sel], variation='el_up')
    print ("Mean value of SF (up, el): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='el_down')
    print ("Mean value of SF (down, el): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='mu_up')
    print ("Mean value of SF (up, mu): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='mu_down')
    print ("Mean value of SF (down, mu): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='el_pt1')
    print ("Mean value of SF (pt1, el): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='el_pt2')
    print ("Mean value of SF (pt2, el): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='el_pt1')
    print ("Mean value of SF (pt1, el): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='el_pt2')
    print ("Mean value of SF (pt2, el): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='mu_pt1')
    print ("Mean value of SF (pt1, mu): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='mu_pt2')
    print ("Mean value of SF (pt2, mu): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='mu_closure_up')
    print ("Mean value of SF (up, mu_closure): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='mu_closure_down')
    print ("Mean value of SF (down, mu_closure): %.3f"%ak.mean(sf_down))
