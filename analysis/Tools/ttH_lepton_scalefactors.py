import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

class LeptonSF:

    def __init__(self, year=2016, era=None):
        self.year = year
        self.era = era
        self.base = os.path.join(here, "data/leptons/ttH/")

        self.ext = extractor()
        self.ext1D = extractor()
        # several histograms can be imported at once using wildcards (*)
        if self.year == 2016:
            if era == "APV":
                self.ext.add_weight_sets([
                    "mu_2016APV_reco NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json"),
                    "mu_2016APV_reco_err NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json"),

                    "mu_2016APV_loose NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json"),
                    "mu_2016APV_loose_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json"),
                    "mu_2016APV_loose_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json"),

                    "mu_2016APV_iso EGamma_SF2D %s"%(self.base+"muon/egammaEffi2016APV_iso_EGM2D.root"),
                    "mu_2016APV_iso_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2016APV_iso_EGM2D.root"),

                    "mu_2016APV_tight EGamma_SF2D %s"%(self.base+"muon/egammaEffi2016APV_EGM2D.root"),
                    "mu_2016APV_tight_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2016APV_EGM2D.root"),

                    "ele_2016APV_reco EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016APV_ptAbove20_EGM2D.root'),
                    "ele_2016APV_reco_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016APV_ptAbove20_EGM2D.root'),

                    "ele_2016APV_reco_low EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016APV_ptBelow20_EGM2D.root'),
                    "ele_2016APV_reco_low_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016APV_ptBelow20_EGM2D.root'),

                    "ele_2016APV_loose EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016APV_recoToloose_EGM2D.root'),
                    "ele_2016APV_loose_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016APV_recoToloose_EGM2D.root'),

                    "ele_2016APV_iso EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016APV_iso_EGM2D.root'),
                    "ele_2016APV_iso_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016APV_iso_EGM2D.root'),

                    "ele_2016APV_tight EGamma_SF2D %s"%(self.base+'elecNEWmva/egammaEffi2016APV_2lss_EGM2D.root'),
                    "ele_2016APV_tight_err EGamma_SF2D_error %s"%(self.base+'elecNEWmva/egammaEffi2016APV_2lss_EGM2D.root'),
                ])
            else:
                self.ext.add_weight_sets([
                    "mu_2016_reco NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json"),
                    "mu_2016_reco_err NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json"),

                    "mu_2016_loose NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json"),
                    "mu_2016_loose_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json"),
                    "mu_2016_loose_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json"),

                    "mu_2016_iso EGamma_SF2D %s"%(self.base+"muon/egammaEffi2016_iso_EGM2D.root"),
                    "mu_2016_iso_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2016_iso_EGM2D.root"),

                    "mu_2016_tight EGamma_SF2D %s"%(self.base+"muon/egammaEffi2016_EGM2D.root"),
                    "mu_2016_tight_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2016_EGM2D.root"),

                    "ele_2016_reco EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016_ptAbove20_EGM2D.root'),
                    "ele_2016_reco_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016_ptAbove20_EGM2D.root'),

                    "ele_2016_reco_low EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016_ptBelow20_EGM2D.root'),
                    "ele_2016_reco_low_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016_ptBelow20_EGM2D.root'),

                    "ele_2016_loose EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016_recoToloose_EGM2D.root'),
                    "ele_2016_loose_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016_recoToloose_EGM2D.root'),

                    "ele_2016_iso EGamma_SF2D %s"%(self.base+'elec/egammaEffi2016_iso_EGM2D.root'),
                    "ele_2016_iso_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2016_iso_EGM2D.root'),

                    "ele_2016_tight EGamma_SF2D %s"%(self.base+'elecNEWmva/egammaEffi2016_2lss_EGM2D.root'),
                    "ele_2016_tight_err EGamma_SF2D_error %s"%(self.base+'elecNEWmva/egammaEffi2016_2lss_EGM2D.root'),
                ])

        elif self.year == 2017:
            self.ext.add_weight_sets([
                "mu_2017_reco NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json"),
                "mu_2017_reco_err NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json"),

                "mu_2017_loose NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json"),
                "mu_2017_loose_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json"),
                "mu_2017_loose_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json"),

                "mu_2017_iso EGamma_SF2D %s"%(self.base+"muon/egammaEffi2017_iso_EGM2D.root"),
                "mu_2017_iso_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2017_iso_EGM2D.root"),

                "mu_2017_tight EGamma_SF2D %s"%(self.base+"muon/egammaEffi2017_EGM2D.root"),
                "mu_2017_tight_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2017_EGM2D.root"),

                "ele_2017_reco EGamma_SF2D %s"%(self.base+'elec/egammaEffi2017_ptAbove20_EGM2D.root'),
                "ele_2017_reco_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2017_ptAbove20_EGM2D.root'),

                "ele_2017_reco_low EGamma_SF2D %s"%(self.base+'elec/egammaEffi2017_ptBelow20_EGM2D.root'),
                "ele_2017_reco_low_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2017_ptBelow20_EGM2D.root'),

                "ele_2017_loose EGamma_SF2D %s"%(self.base+'elec/egammaEffi2017_recoToloose_EGM2D.root'),
                "ele_2017_loose_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2017_recoToloose_EGM2D.root'),

                "ele_2017_iso EGamma_SF2D %s"%(self.base+'elec/egammaEffi2017_iso_EGM2D.root'),
                "ele_2017_iso_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2017_iso_EGM2D.root'),

                "ele_2017_tight EGamma_SF2D %s"%(self.base+'elecNEWmva/egammaEffi2017_2lss_EGM2D.root'),
                "ele_2017_tight_err EGamma_SF2D_error %s"%(self.base+'elecNEWmva/egammaEffi2017_2lss_EGM2D.root'),
            ])

        elif self.year == 2018:
            self.ext.add_weight_sets([
                "mu_2018_reco NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json"),
                "mu_2018_reco_err NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%(self.base+"muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json"),

                "mu_2018_loose NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json"),
                "mu_2018_loose_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json"),
                "mu_2018_loose_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s"%(self.base+"muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json"),

                "mu_2018_iso EGamma_SF2D %s"%(self.base+"muon/egammaEffi2018_iso_EGM2D.root"),
                "mu_2018_iso_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2018_iso_EGM2D.root"),

                "mu_2018_tight EGamma_SF2D %s"%(self.base+"muon/egammaEffi2018_EGM2D.root"),
                "mu_2018_tight_err EGamma_SF2D_error %s"%(self.base+"muon/egammaEffi2018_EGM2D.root"),

                "ele_2018_reco EGamma_SF2D %s"%(self.base+'elec/egammaEffi2018_ptAbove20_EGM2D.root'),
                "ele_2018_reco_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2018_ptAbove20_EGM2D.root'),

                "ele_2018_reco_low EGamma_SF2D %s"%(self.base+'elec/egammaEffi2018_ptBelow20_EGM2D.root'),
                "ele_2018_reco_low_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2018_ptBelow20_EGM2D.root'),

                "ele_2018_loose EGamma_SF2D %s"%(self.base+'elec/egammaEffi2018_recoToloose_EGM2D.root'),
                "ele_2018_loose_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2018_recoToloose_EGM2D.root'),

                "ele_2018_iso EGamma_SF2D %s"%(self.base+'elec/egammaEffi2018_iso_EGM2D.root'),
                "ele_2018_iso_err EGamma_SF2D_error %s"%(self.base+'elec/egammaEffi2018_iso_EGM2D.root'),

                "ele_2018_tight EGamma_SF2D %s"%(self.base+'elecNEWmva/egammaEffi2018_2lss_EGM2D.root'),
                "ele_2018_tight_err EGamma_SF2D_error %s"%(self.base+'elecNEWmva/egammaEffi2018_2lss_EGM2D.root'),
            ])
        
        
        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, ele, mu, variation='central', collection='elemu'):
        sign_mu = 0
        sign_ele = 0
        if variation == 'up':
            if collection.count('ele'):
                sign_ele = 1
            if collection.count('mu'):
                sign_mu = 1
        if variation == 'down':
            if collection.count('ele'):
                sign_ele = -1
            if collection.count('mu'):
                sign_mu = -1
        yearstr = str(self.year)
        if self.era == 'APV':
            yearstr += 'APV'

        # central values
        ele_sf_reco     = self.evaluator[f"ele_{yearstr}_reco"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
        ele_sf_reco_low = self.evaluator[f"ele_{yearstr}_reco_low"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
        ele_sf_loose    = self.evaluator[f"ele_{yearstr}_loose"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
        ele_sf_iso      = self.evaluator[f"ele_{yearstr}_iso"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
        ele_sf_tight    = self.evaluator[f"ele_{yearstr}_tight"](abs(ele.eta + ele.deltaEtaSC), ele.pt)

        mu_sf_reco      = self.evaluator[f"mu_{yearstr}_reco"](abs(mu.eta), mu.pt)[mu.pt<20]
        mu_sf_loose     = self.evaluator[f"mu_{yearstr}_loose"](abs(mu.eta), mu.pt)
        mu_sf_iso       = self.evaluator[f"mu_{yearstr}_iso"](abs(mu.eta), mu.pt)
        mu_sf_tight     = self.evaluator[f"mu_{yearstr}_tight"](abs(mu.eta), mu.pt)

        # errors
        ele_sf_reco_err     = self.evaluator[f"ele_{yearstr}_reco_err"](ele[ele.pt>20].eta, ele[ele.pt>20].pt)
        ele_sf_reco_low_err = self.evaluator[f"ele_{yearstr}_reco_low_err"](ele[ele.pt<=20].eta, ele[ele.pt<=20].pt)
        ele_sf_loose_err    = self.evaluator[f"ele_{yearstr}_loose_err"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
        ele_sf_iso_err      = self.evaluator[f"ele_{yearstr}_iso_err"](abs(ele.eta + ele.deltaEtaSC), ele.pt)
        ele_sf_tight_err    = self.evaluator[f"ele_{yearstr}_tight_err"](abs(ele.eta + ele.deltaEtaSC), ele.pt)

        mu_sf_reco_err      = self.evaluator[f"mu_{yearstr}_reco_err"](abs(mu.eta), mu.pt)[mu.pt<20]
        mu_sf_loose_syst    = self.evaluator[f"mu_{yearstr}_loose_syst"](abs(mu.eta), mu.pt)
        mu_sf_loose_stat    = self.evaluator[f"mu_{yearstr}_loose_syst"](abs(mu.eta), mu.pt)
        mu_sf_iso_err       = self.evaluator[f"mu_{yearstr}_iso_err"](abs(mu.eta), mu.pt)
        mu_sf_tight_err     = self.evaluator[f"mu_{yearstr}_tight_err"](abs(mu.eta), mu.pt)

        mu_sf_loose_err = np.sqrt(mu_sf_loose_syst*mu_sf_loose_syst + mu_sf_loose_stat*mu_sf_loose_stat)

        sf = ak.prod(ele_sf_reco    + sign_ele*ele_sf_reco_err, axis=1) *\
            ak.prod(ele_sf_reco_low + sign_ele*ele_sf_reco_low_err, axis=1) *\
            ak.prod(ele_sf_loose    + sign_ele*ele_sf_loose_err, axis=1) *\
            ak.prod(ele_sf_iso      + sign_ele*ele_sf_iso_err, axis=1) *\
            ak.prod(ele_sf_tight    + sign_ele*ele_sf_tight_err, axis=1) *\
            ak.prod(mu_sf_reco      + ak.sum(sign_mu*mu_sf_reco_err, axis=1), axis=1) *\
            ak.prod(mu_sf_loose     + sign_mu*mu_sf_loose_err, axis=1) *\
            ak.prod(mu_sf_iso       + sign_mu*mu_sf_iso_err, axis=1) *\
            ak.prod(mu_sf_tight     + sign_mu*mu_sf_tight_err, axis=1)

        return sf

    def values(self):

        return 0
        


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16APV = LeptonSF(year=2016, era='APV')
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

    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
    mu  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()

    sel = ((ak.num(el)==1)&(ak.num(mu)==1))

    sf_central  = sf18.get(el[sel], mu[sel], variation='central')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    sf_up       = sf18.get(el[sel], mu[sel], variation='up')
    print ("Mean value of SF (up, all): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='down')
    print ("Mean value of SF (down, all): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='up', collection='ele')
    print ("Mean value of SF (up, ele): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='down', collection='ele')
    print ("Mean value of SF (down, ele): %.3f"%ak.mean(sf_down))

    sf_up       = sf18.get(el[sel], mu[sel], variation='up', collection='mu')
    print ("Mean value of SF (up, mu): %.3f"%ak.mean(sf_up))
    sf_down     = sf18.get(el[sel], mu[sel], variation='down', collection='mu')
    print ("Mean value of SF (down, mu): %.3f"%ak.mean(sf_down))
