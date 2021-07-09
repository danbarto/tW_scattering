import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor
import numpy as np

class LeptonSF:

    def __init__(self, year=2016):
        self.year = year

        ele_2016_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2016.root")
        ele_2016_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2016.root")
        ele_2016_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2016_2lss/passttH/egammaEffi_txt_EGM2D.root")
        ele_2016_reco       = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root")
        ele_2016_reco_low   = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_low_RecoSF_Legacy2016.root")
        
        ele_2017_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2017.root")
        ele_2017_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2017.root")
        ele_2017_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2017_2lss/passttH/egammaEffi_txt_EGM2D.root")
        ele_2017_reco       = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO.root")
        ele_2017_reco_low   = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO_lowEt.root")
        
        ele_2018_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2018.root")
        ele_2018_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2018.root")
        ele_2018_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2018_2lss/passttH/egammaEffi_txt_EGM2D.root")
        ele_2018_tight_pt   = os.path.expandvars("$TWHOME/data/leptons/ttH/error/SFttbar_2018_ele_pt.root")
        ele_2018_tight_eta  = os.path.expandvars("$TWHOME/data/leptons/ttH/error/SFttbar_2018_ele_eta.root")
        ele_2018_reco       = os.path.expandvars("$TWHOME/data/leptons/2018_egammaEffi_txt_EGM2D_updatedAll.root")


        muon_2016_loose     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2016.root")
        muon_2016_tight     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2016_2lss/passttH/egammaEffi_txt_EGM2D.root")

        muon_2017_loose     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2017.root")
        muon_2017_tight     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2017_2lss/passttH/egammaEffi_txt_EGM2D.root")

        muon_2018_loose     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2018.root")
        muon_2018_tight     = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2018_2lss/passttH/egammaEffi_txt_EGM2D.root")
        muon_2018_tight_pt  = os.path.expandvars("$TWHOME/data/leptons/ttH/error/SFttbar_2018_muon_pt.root")
        muon_2018_tight_eta = os.path.expandvars("$TWHOME/data/leptons/ttH/error/SFttbar_2018_muon_eta.root")

        
        self.ext = extractor()
        self.ext1D = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            self.ext.add_weight_sets([
                "mu_2016_loose EGamma_SF2D %s"%muon_2016_loose,
                "mu_2016_tight EGamma_SF2D %s"%muon_2016_tight,
       
                "ele_2016_reco EGamma_SF2D %s"%ele_2016_reco,
                "ele_2016_reco_low EGamma_SF2D %s"%ele_2016_reco_low,
                "ele_2016_loose EGamma_SF2D %s"%ele_2016_loose,
                "ele_2016_looseTTH EGamma_SF2D %s"%ele_2016_looseTTH,
                "ele_2016_tight EGamma_SF2D %s"%ele_2016_tight,
            ])
        
        elif self.year == 2017:
            self.ext.add_weight_sets([
                "mu_2017_loose EGamma_SF2D %s"%muon_2017_loose,
                "mu_2017_tight EGamma_SF2D %s"%muon_2017_tight,
       
                "ele_2017_reco EGamma_SF2D %s"%ele_2017_reco,
                "ele_2017_reco_low EGamma_SF2D %s"%ele_2017_reco_low,
                "ele_2017_loose EGamma_SF2D %s"%ele_2017_loose,
                "ele_2017_looseTTH EGamma_SF2D %s"%ele_2017_looseTTH,
                "ele_2017_tight EGamma_SF2D %s"%ele_2017_tight,
            ])

        elif self.year == 2018:
            self.ext.add_weight_sets([
                "mu_2018_loose EGamma_SF2D %s"%muon_2018_loose,
                "mu_2018_tight EGamma_SF2D %s"%muon_2018_tight,

                "ele_2018_reco EGamma_SF2D %s"%ele_2018_reco,
                "ele_2018_loose EGamma_SF2D %s"%ele_2018_loose,
                "ele_2018_looseTTH EGamma_SF2D %s"%ele_2018_looseTTH,
                "ele_2018_tight EGamma_SF2D %s"%ele_2018_tight,

                "ele_2018_tight_pt histo_eff_data %s"%ele_2018_tight_pt,
                "ele_2018_tight_eta histo_eff_data %s"%ele_2018_tight_eta,

                "mu_2018_tight_pt histo_eff_data %s"%muon_2018_tight_pt,
                "mu_2018_tight_eta histo_eff_data %s"%muon_2018_tight_eta,
            ])
        
        
        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu, variation='central'):
        
        if self.year == 2016:
            ele_sf_reco     = self.evaluator["ele_2016_reco"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
            ele_sf_reco_low = self.evaluator["ele_2016_reco_low"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
            ele_sf_loose    = self.evaluator["ele_2016_loose"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_looseTTH = self.evaluator["ele_2016_looseTTH"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_tight    = self.evaluator["ele_2016_tight"](abs(ele.eta + ele.deltaEtaSC), ele.pt)

            mu_sf_loose     = self.evaluator["mu_2016_loose"](abs(mu.eta), mu.pt)
            mu_sf_tight     = self.evaluator["mu_2016_tight"](abs(mu.eta), mu.pt)

            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_reco_low, axis=1) * ak.prod(ele_sf_loose, axis=1) * ak.prod(ele_sf_looseTTH, axis=1) * ak.prod(ele_sf_tight, axis=1) * ak.prod(mu_sf_loose, axis=1) * ak.prod(mu_sf_tight, axis=1)

        elif self.year == 2017:
            ele_sf_reco     = self.evaluator["ele_2017_reco"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
            ele_sf_reco_low = self.evaluator["ele_2017_reco_low"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
            ele_sf_loose    = self.evaluator["ele_2017_loose"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_looseTTH = self.evaluator["ele_2017_looseTTH"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_tight    = self.evaluator["ele_2017_tight"](abs(ele.eta + ele.deltaEtaSC), ele.pt)

            mu_sf_loose     = self.evaluator["mu_2017_loose"](abs(mu.eta), mu.pt)
            mu_sf_tight     = self.evaluator["mu_2017_tight"](abs(mu.eta), mu.pt)

            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_reco_low, axis=1) * ak.prod(ele_sf_loose, axis=1) * ak.prod(ele_sf_looseTTH, axis=1) * ak.prod(ele_sf_tight, axis=1) * ak.prod(mu_sf_loose, axis=1) * ak.prod(mu_sf_tight, axis=1)

        elif self.year == 2018:
            ele_sf_reco     = self.evaluator["ele_2018_reco"](ele.eta, ele.pt)
            ele_sf_loose    = self.evaluator["ele_2018_loose"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_looseTTH = self.evaluator["ele_2018_looseTTH"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
            ele_sf_tight    = self.evaluator["ele_2018_tight"](abs(ele.eta + ele.deltaEtaSC), ele.pt)

            mu_sf_loose     = self.evaluator["mu_2018_loose"](abs(mu.eta), mu.pt)
            mu_sf_tight     = self.evaluator["mu_2018_tight"](abs(mu.eta), mu.pt)

            if not variation=='central':

                ele_sf_tight_err1 = self.evaluator["ele_2018_tight_eta"](abs(ele.eta + ele.deltaEtaSC))
                ele_sf_tight_err2 = self.evaluator["ele_2018_tight_pt"](ele.pt)

                ele_sf_tight_err1 = ak.from_regular(ele_sf_tight_err1[:,:,np.newaxis])
                ele_sf_tight_err2 = ak.from_regular(ele_sf_tight_err2[:,:,np.newaxis])
                ele_sf_tight_err  = ak.max(ak.concatenate([ele_sf_tight_err1, ele_sf_tight_err2], axis=2), axis=2)

                mu_sf_tight_err1 = self.evaluator["mu_2018_tight_eta"](abs(mu.eta))
                mu_sf_tight_err2 = self.evaluator["mu_2018_tight_pt"](mu.pt)

                mu_sf_tight_err1 = ak.from_regular(mu_sf_tight_err1[:,:,np.newaxis])
                mu_sf_tight_err2 = ak.from_regular(mu_sf_tight_err2[:,:,np.newaxis])
                mu_sf_tight_err  = ak.max(ak.concatenate([mu_sf_tight_err1, mu_sf_tight_err2], axis=2), axis=2)

                if variation=='up':
                    ele_sf_tight = ele_sf_tight*ele_sf_tight_err
                    mu_sf_tight = mu_sf_tight*mu_sf_tight_err
                if variation=='down':
                    ele_sf_tight = ele_sf_tight/ele_sf_tight_err
                    mu_sf_tight = mu_sf_tight/mu_sf_tight_err


            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_loose, axis=1) * ak.prod(ele_sf_looseTTH, axis=1) * ak.prod(ele_sf_tight, axis=1) * ak.prod(mu_sf_loose, axis=1) * ak.prod(mu_sf_tight, axis=1)


        return sf

    def values(self):

        return 0
        


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16 = LeptonSF(year=2016)
    sf17 = LeptonSF(year=2017)
    sf18 = LeptonSF(year=2018)
    
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

    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
    mu  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()

    sel = ((ak.num(el)==1)&(ak.num(mu)==1))

    sf_central  = sf18.get(el[sel], mu[sel], variation='central')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    sf_up       = sf18.get(el[sel], mu[sel], variation='up')
    print ("Mean value of SF (up): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='down')
    print ("Mean value of SF (down): %.3f"%ak.mean(sf_down))

