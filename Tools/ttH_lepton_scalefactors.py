import os
import awkward1 as ak
from coffea.lookup_tools import extractor

class LeptonSF:

    def __init__(self, year=2016):
        self.year = year

        ele_2016_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2016.root")
        ele_2016_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2016.root")
        ele_2016_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2016_2lss/passttH/egammaEffi.txt_EGM2D.root")
        ele_2016_reco       = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root")
        ele_2016_reco_low   = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_low_RecoSF_Legacy2016.root")
        
        ele_2017_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2017.root")
        ele_2017_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2017.root")
        ele_2017_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2017_2lss/passttH/egammaEffi.txt_EGM2D.root")
        ele_2017_reco       = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO.root")
        ele_2017_reco_low   = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO_lowEt.root")
        
        ele_2018_loose      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_ele_2018.root")
        ele_2018_looseTTH   = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loosettH_ele_2018.root")
        ele_2018_tight      = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_ele_2018_2lss/passttH/egammaEffi.txt_EGM2D.root")
        ele_2018_reco       = os.path.expandvars("$TWHOME/data/leptons/2018_egammaEffi_txt_EGM2D_updatedAll.root")


        muon_2016_loose = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2016.root")
        muon_2016_tight = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2016_2lss/passttH/egammaEffi.txt_EGM2D.root")

        muon_2017_loose = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2017.root")
        muon_2017_tight = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2017_2lss/passttH/egammaEffi.txt_EGM2D.root")

        muon_2018_loose = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_loose_muon_2018.root")
        muon_2018_tight = os.path.expandvars("$TWHOME/data/leptons/ttH/TnP_ttH_muon_2018_2lss/passttH/egammaEffi.txt_EGM2D.root")

        
        self.ext = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            self.ext.add_weight_sets(["mu_2016_loose EGamma_SF2D %s"%muon_2016_loose])
            self.ext.add_weight_sets(["mu_2016_tight EGamma_SF2D %s"%muon_2016_tight])
       
            self.ext.add_weight_sets(["ele_2016_reco EGamma_SF2D %s"%ele_2016_reco])
            self.ext.add_weight_sets(["ele_2016_reco_low EGamma_SF2D %s"%ele_2016_reco_low])
            self.ext.add_weight_sets(["ele_2016_loose EGamma_SF2D %s"%ele_2016_loose])
            self.ext.add_weight_sets(["ele_2016_looseTTH EGamma_SF2D %s"%ele_2016_looseTTH])
            self.ext.add_weight_sets(["ele_2016_tight EGamma_SF2D %s"%ele_2016_tight])
        
        elif self.year == 2017:
            self.ext.add_weight_sets(["mu_2017_loose EGamma_SF2D %s"%muon_2017_loose])
            self.ext.add_weight_sets(["mu_2017_tight EGamma_SF2D %s"%muon_2017_tight])
       
            self.ext.add_weight_sets(["ele_2017_reco EGamma_SF2D %s"%ele_2017_reco])
            self.ext.add_weight_sets(["ele_2017_reco_low EGamma_SF2D %s"%ele_2017_reco_low])
            self.ext.add_weight_sets(["ele_2017_loose EGamma_SF2D %s"%ele_2017_loose])
            self.ext.add_weight_sets(["ele_2017_looseTTH EGamma_SF2D %s"%ele_2017_looseTTH])
            self.ext.add_weight_sets(["ele_2017_tight EGamma_SF2D %s"%ele_2017_tight])

        elif self.year == 2018:
            self.ext.add_weight_sets(["mu_2018_loose EGamma_SF2D %s"%muon_2018_loose])
            self.ext.add_weight_sets(["mu_2018_tight EGamma_SF2D %s"%muon_2018_tight])
       
            self.ext.add_weight_sets(["ele_2018_reco EGamma_SF2D %s"%ele_2018_reco])
            self.ext.add_weight_sets(["ele_2018_loose EGamma_SF2D %s"%ele_2018_loose])
            self.ext.add_weight_sets(["ele_2018_looseTTH EGamma_SF2D %s"%ele_2018_looseTTH])
            self.ext.add_weight_sets(["ele_2018_tight EGamma_SF2D %s"%ele_2018_tight])
        
        
        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu):
        
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

            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_loose, axis=1) * ak.prod(ele_sf_looseTTH, axis=1) * ak.prod(ele_sf_tight, axis=1) * ak.prod(mu_sf_loose, axis=1) * ak.prod(mu_sf_tight, axis=1)


        return sf

    def values(self):

        return 0
        


if __name__ == '__main__':
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
