import os
import awkward1 as ak
from coffea.lookup_tools import extractor

class LeptonSF:

    def __init__(self, year=2016):
        self.year = year

        electronSF_2016 = os.path.expandvars("$TWHOME/data/leptons/ElectronScaleFactors_Run2016.root")
        electronReco_2016 = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root")
        electronRecoLow_2016 = os.path.expandvars("$TWHOME/data/leptons/2016_EGM2D_BtoH_low_RecoSF_Legacy2016.root")
        
        electronSF_2017 = os.path.expandvars("$TWHOME/data/leptons/ElectronScaleFactors_Run2017.root")
        electronReco_2017 = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO.root")
        electronRecoLow_2017 = os.path.expandvars("$TWHOME/data/leptons/2017_egammaEffi_txt_EGM2D_runBCDEF_passingRECO_lowEt.root")
        
        electronSF_2018 = os.path.expandvars("$TWHOME/data/leptons/ElectronScaleFactors_Run2018.root")
        electronReco_2018 = os.path.expandvars("$TWHOME/data/leptons/2018_egammaEffi_txt_EGM2D_updatedAll.root")
        
        muonID_2016 = os.path.expandvars("$TWHOME/data/leptons/2016_Muon_TnP_NUM_LooseID_DENOM_generalTracks_VAR_map_pt_eta.root")
        muonIso_2016 = os.path.expandvars("$TWHOME/data/leptons/2016_Muon_TnP_NUM_MiniIsoTight_DENOM_LooseID_VAR_map_pt_eta.root")
        
        muonID_2017 = os.path.expandvars("$TWHOME/data/leptons/2017_Muon_RunBCDEF_SF_ID.root")
        muonIso_2017 = os.path.expandvars("$TWHOME/data/leptons/Muon_multiIso_2017.root")
        
        muonID_2018 = os.path.expandvars("$TWHOME/data/leptons/2018_Muon_RunABCD_SF_ID.root")
        muonIso_2018 = os.path.expandvars("$TWHOME/data/leptons/Muon_multiIso_2018.root") # same as 2017
        
        self.ext = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            self.ext.add_weight_sets(["mu_2016_id SF %s"%muonID_2016])
            self.ext.add_weight_sets(["mu_2016_iso SF %s"%muonIso_2016])
       
            self.ext.add_weight_sets(["ele_2016_reco EGamma_SF2D %s"%electronReco_2016])
            self.ext.add_weight_sets(["ele_2016_reco_low EGamma_SF2D %s"%electronRecoLow_2016])
            self.ext.add_weight_sets(["ele_2016_id Run2016_MVATightIP2D3DIDEmu %s"%electronSF_2016])
            self.ext.add_weight_sets(["ele_2016_iso Run2016_MultiIsoEmu %s"%electronSF_2016])
        
        elif self.year == 2017:
            self.ext.add_weight_sets(["mu_2017_id NUM_MediumID_DEN_genTracks_pt_abseta %s"%muonID_2017])
            self.ext.add_weight_sets(["mu_2017_iso TnP_MC_NUM_MultiIsoMCut_DEN_MediumID_PAR_pt_eta %s"%muonIso_2017])
       
            self.ext.add_weight_sets(["ele_2017_reco EGamma_SF2D %s"%electronReco_2017])
            self.ext.add_weight_sets(["ele_2017_reco_low EGamma_SF2D %s"%electronRecoLow_2017])
            self.ext.add_weight_sets(["ele_2017_id Run2017_MVATightIP2D3DIDEmu %s"%electronSF_2017])
            self.ext.add_weight_sets(["ele_2017_iso Run2017_MultiIsoEmuJECv32 %s"%electronSF_2017]) # I gess this is the right one

        elif self.year == 2018:
            self.ext.add_weight_sets(["mu_2018_id NUM_MediumID_DEN_TrackerMuons_pt_abseta %s"%muonID_2018])
            self.ext.add_weight_sets(["mu_2018_iso SF2D %s"%muonIso_2018])
       
            self.ext.add_weight_sets(["ele_2018_reco EGamma_SF2D %s"%electronReco_2018])
            self.ext.add_weight_sets(["ele_2018_id Run2018_MVATightIP2D3DIDEmu %s"%electronSF_2018])
            self.ext.add_weight_sets(["ele_2018_iso Run2018_MultiIsoEmu %s"%electronSF_2018])
        
        
        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu):
        
        if self.year == 2016:
            ele_sf_reco     = self.evaluator["ele_2016_reco"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
            ele_sf_reco_low = self.evaluator["ele_2016_reco_low"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
            ele_sf_id       = self.evaluator["ele_2016_id"](ele.eta + ele.deltaEtaSC, ele.pt)
            ele_sf_iso      = self.evaluator["ele_2016_iso"](ele.eta + ele.deltaEtaSC, ele.pt)

            mu_sf_id        = self.evaluator["mu_2016_id"](mu.pt, abs(mu.eta))
            mu_sf_iso       = self.evaluator["mu_2016_iso"](mu.pt, abs(mu.eta))

            #sf = ele_sf_id.prod() * ele_sf_iso.prod() * ele_sf_reco.prod() * ele_sf_reco_low.prod() * mu_sf_id.prod() * mu_sf_iso.prod()
            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_reco_low, axis=1) * ele_sf_id.prod() * ele_sf_iso.prod() * mu_sf_id.prod() * mu_sf_iso.prod() #FIXME

        elif self.year == 2017:
            ele_sf_reco     = self.evaluator["ele_2017_reco"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
            ele_sf_reco_low = self.evaluator["ele_2017_reco_low"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
            ele_sf_id       = self.evaluator["ele_2017_id"](ele.eta + ele.deltaEtaSC, ele.pt)
            ele_sf_iso      = self.evaluator["ele_2017_iso"](ele.eta + ele.deltaEtaSC, ele.pt)

            mu_sf_id        = self.evaluator["mu_2017_id"](mu.pt, abs(mu.eta))
            mu_sf_iso       = self.evaluator["mu_2017_iso"](mu.pt, abs(mu.eta))

            #sf = ele_sf_id.prod() * ele_sf_iso.prod() * ele_sf_reco.prod() * ele_sf_reco_low.prod() * mu_sf_id.prod() * mu_sf_iso.prod()
            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_reco_low, axis=1) *  ele_sf_id.prod() * ele_sf_iso.prod() * mu_sf_id.prod() * mu_sf_iso.prod() #FIXME

        elif self.year == 2018:
            ele_sf_reco     = self.evaluator["ele_2018_reco"](ele.eta, ele.pt)
            ele_sf_id       = self.evaluator["ele_2018_id"](ele.eta + ele.deltaEtaSC, ele.pt)
            ele_sf_iso      = self.evaluator["ele_2018_iso"](ele.eta + ele.deltaEtaSC, ele.pt)

            mu_sf_id        = self.evaluator["mu_2018_id"](mu.pt, abs(mu.eta))
            mu_sf_iso       = self.evaluator["mu_2018_iso"](abs(mu.eta), mu.pt)

            #sf = ele_sf_id.prod() * ele_sf_iso.prod() * ele_sf_reco.prod() * mu_sf_id.prod() * mu_sf_iso.prod()
            sf = ak.prod(ele_sf_reco, axis=1) * ak.prod(ele_sf_id, axis=1) * ak.prod(ele_sf_iso, axis=1) * ak.prod(mu_sf_id, axis=1) * ak.prod(mu_sf_iso, axis=1)


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
