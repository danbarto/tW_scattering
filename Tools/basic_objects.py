'''
Standardized object selections for simple objects like jets
'''
import os

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

import numpy as np

from Tools.helpers import get_four_vec_fromPtEtaPhiM

def getPtEtaPhi(coll, pt_var='pt', eta_var='eta', phi_var='phi'):
    return ak.zip({
                    'pt':  getattr(coll, pt_var),
                    'eta': getattr(coll, eta_var),
                    'phi': getattr(coll, phi_var),
                    'p': coll.p, # this is uncorrected....
                    'btagDeepFlavB': coll.btagDeepFlavB,
                    'jetId': coll.jetId,
                    'puId': coll.puId,
                })

def getTaus(ev, WP='veto'):
    if WP == 'veto':
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2#Tau_Identification
        # I guess we can use Tau_idDeepTau2017v2p1VSjet >= 16. This corresponds to loose
        # https://github.com/cms-tau-pog/TauFW/tree/master/PicoProducer/python/analysis#accessing-nanoaod
        #
        return ev.Tau[(ev.Tau.pt > 20) & (abs(ev.Tau.eta) < 2.3) & (ev.Tau.idDeepTau2017v2p1VSjet>=16)]

def getIsoTracks(ev, WP='veto'):
    if WP == 'veto':
        return ev.IsoTrack[(ev.IsoTrack.pt > 10) & (abs(ev.IsoTrack.eta) < 2.4) & ((ev.IsoTrack.pfRelIso03_all < 0.1) | ((ev.IsoTrack.pfRelIso03_all*ev.IsoTrack.pt) < 6))]
 
    
def getFatJets(ev):
    return ev.FatJet[(ev.FatJet.pt>200) & (abs(ev.FatJet.eta)<2.4)]

def getHadronFlavour(jet, hadronFlavour=5):
    return jet[(abs(jet.hadronFlavour)==hadronFlavour)]

def getMET(ev, pt_var='pt'):
    zeros = ak.zeros_like(ev.MET.pt)
    if pt_var == 'pt':
        return get_four_vec_fromPtEtaPhiM(ev.MET, ev.MET.pt, zeros, ev.MET.phi, zeros, copy=False)
    elif pt_var == 'pt_nom':
        return get_four_vec_fromPtEtaPhiM(ev.MET, ev.MET.T1_pt, zeros, ev.MET.T1_phi, zeros, copy=False)
    elif pt_var == 'pt_jesTotalUp':
        return get_four_vec_fromPtEtaPhiM(ev.MET, ev.MET.T1_pt_jesTotalUp, zeros, ev.MET.T1_phi_jesTotalUp, zeros, copy=False)
    elif pt_var == 'pt_jesTotalDown':
        return get_four_vec_fromPtEtaPhiM(ev.MET, ev.MET.T1_pt_jesTotalDown, zeros, ev.MET.T1_phi_jesTotalDown, zeros, copy=False)
    else:
        return get_four_vec_fromPtEtaPhiM(
            ev.MET,
            getattr(ev.MET, pt_var),
            zeros,
            getattr(ev.MET,  pt_var.replace('pt', 'phi'))
        )

def getJets(ev, maxEta=100, minPt=25, pt_var='pt', year=2018):
    jets = ev.Jet[(getattr(ev.Jet, pt_var)>minPt) & (abs(ev.Jet.eta)<maxEta) & (ev.Jet.jetId>1)]
    jets = jets[ak.argsort(getattr(jets, pt_var), ascending=False)]
    jets['p4'] = get_four_vec_fromPtEtaPhiM(jets, getattr(jets, pt_var), jets.eta, jets.phi, jets.mass, copy=False)
    return jets

def getBTagsDeepB(jet, year=2016, invert=False):
    if year == 2016:
        sel = ((jet.btagDeepB>0.6321) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation2016Legacy
    elif year == 2017:
        sel = ((jet.btagDeepB>0.4941) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X
    elif year == 2018:
        sel = ((jet.btagDeepB>0.4184) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
    if invert: sel = ~sel
    return jet[sel]

def getBTagsDeepFlavB(jet, year=2016, era=None, invert=False, UL=True):
    if UL:
        if year == 2016:
            if era=='APV':
                sel = ((jet.btagDeepFlavB>0.2598) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16preVFP
            else:
                sel = ((jet.btagDeepFlavB>0.2489) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation106XUL16postVFP
        elif year == 2017:
            sel = ((jet.btagDeepFlavB>0.3040) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17
        elif year == 2018:
            sel = ((jet.btagDeepFlavB>0.2783) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
    else:
        if year == 2016:
            sel = ((jet.btagDeepFlavB>0.3093) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation2016Legacy
        elif year == 2017:
            sel = ((jet.btagDeepFlavB>0.3033) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X
        elif year == 2018:
            sel = ((jet.btagDeepFlavB>0.2770) & (abs(jet.eta)<2.5)) # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
    if invert: sel = ~sel
    return jet[sel]

def getFwdJet(jet, minPt=40, puId=True):
    minId = 7 if puId else 0
    if puId:
        return jet[(abs(jet.p4.eta)>1.7) & (abs(jet.p4.eta)<4.7) & (jet.p4.pt>minPt) & ( ((jet.puId>=minId) & (jet.p4.pt<50)) | (jet.p4.pt>=50))] # PU jet Id just for jets below 50 GeV
    else:
        return jet[(abs(jet.p4.eta)>1.7) & (abs(jet.p4.eta)<4.7) & (jet.p4.pt>minPt) ]

def getHTags(fatjet, year=2016):
    # 2.5% WP
    # https://indico.cern.ch/event/853828/contributions/3723593/attachments/1977626/3292045/lg-btv-deepak8v2-sf-20200127.pdf#page=4
    if year == 2016:
        return fatjet[(fatjet.deepTagMD_HbbvsQCD > 0.8945)] 
    elif year == 2017:
        return fatjet[(fatjet.deepTagMD_HbbvsQCD > 0.8695)] 
    elif year == 2018:
        return fatjet[(fatjet.deepTagMD_HbbvsQCD > 0.8365)] 

def getWTags(fatjet, year=2016, WP='1p0'):
    # 1% WP
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAK8Tagging2018WPsSFs
    if year == 2016:
        cuts = {'1p0': 0.918, '2p5': 0.763}
        return fatjet[(fatjet.deepTag_WvsQCD > cuts[WP])] 
    elif year == 2017:
        cuts = {'1p0': 0.925, '2p5': 0.772}
        return fatjet[(fatjet.deepTag_WvsQCD > cuts[WP])] 
    elif year == 2018:
        cuts = {'1p0': 0.918, '2p5': 0.762}
        return fatjet[(fatjet.deepTag_WvsQCD > cuts[WP])] # yes, really

def getGenW(df):
    GenW = JaggedCandidateArray.candidatesfromcounts(
            df['nGenW'],
            pt = df['GenW_pt'].content,
            eta = df['GenW_eta'].content,
            phi = df['GenW_phi'].content,
            mass = ((df['GenW_pt']>0)*80).content,
        )
    return GenW

def getGenParts(df):
    GenPart = JaggedCandidateArray.candidatesfromcounts(
        df['nGenPart'],
        pt=df['GenPart_pt'].content,
        eta=df['GenPart_eta'].content,
        phi=df['GenPart_phi'].content,
        mass=df['GenPart_mass'].content,
        pdgId=df['GenPart_pdgId'].content,
        status=df['GenPart_status'].content,
        genPartIdxMother=df['GenPart_genPartIdxMother'].content,
        statusFlags=df['GenPart_statusFlags'].content,
    )
    return GenPart

def getHadW(df):
    # Get hadronically decaying W from the data frame
    # We first get the GenParts that have a mother with abs(PDG ID) = 24 with an abs(PDG ID) < 6.
    # Then, we get the mother GenParts of those. Because we don't want to get the same W bosons twice, we can just require PDG ID < 6 instead of abs(PDG ID) < 6
    GenPart = getGenParts(df)
    return GenPart[GenPart[((GenPart.pdgId<6) & (GenPart.pdgId>0) & (abs(GenPart[GenPart.genPartIdxMother].pdgId)==24))].genPartIdxMother]

def getHadW_fromGenPart(GenPart):
    # We first get the GenParts that have a mother with abs(PDG ID) = 24 with an abs(PDG ID) < 6.
    # Then, we get the mother GenParts of those. Because we don't want to get the same W bosons twice, we can just require PDG ID < 6 instead of abs(PDG ID) < 6
    return GenPart[GenPart[((GenPart.pdgId<6) & (GenPart.pdgId>0) & (abs(GenPart[GenPart.genPartIdxMother].pdgId)==24))].genPartIdxMother]
