'''
Standardized object selection, based on SS(++) analysis
'''
import os

import copy
import numpy as np
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak


from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def delta_phi(first, second):
    return (first.phi - second.phi + np.pi) % (2 * np.pi) - np.pi

def delta_phi_alt(first, second):
    # my version, seems to be faster (and unsigned)
    return np.arccos(np.cos(first.phi - second.phi))

def delta_r2(first, second):
    return (first.eta - second.eta) ** 2 + delta_phi_alt(first, second) ** 2
    
def delta_r(first, second):
    return np.sqrt(delta_r2(first, second))

def delta_r_v2(first, second):
    combs = ak.cartesian([first, second], nested=True)
    return np.sqrt(delta_r2(combs['0'], combs['1']))

def match(first, second, deltaRCut=0.4):
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.any((delta_r2(combs['0'], combs['1'])<drCut2), axis=2)

def match2(first, second, deltaRCut=0.4):
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.any((combs['0'].delta_r2(combs['1'])<drCut2), axis=2)

def match_with_pt(first, second, deltaRCut=0.4, ptCut=0.5):
    '''
    match based on deltaR between first and second, and impose that second.pt > first.pt*ptCut
    '''
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.any(
        (delta_r2(combs['0'], combs['1'])<drCut2) & (combs['1'].pt > ptCut*combs['0'].pt)
        , axis=2)

def choose(first, n=2):
    tmp = ak.combinations(first, n)
    combs = tmp['0']
    for i in range(1,n):
        combs = combs.__add__(tmp[str(i)])
    for i in range(n):
        combs[str(i)] = tmp[str(i)]
    return combs

def choose3(first, n=3):
    from warnings import warn
    warn("Deprecation Warning: The choose3 function will be removed. Use choose(first, n=3) instead.")
    tmp = ak.combinations(first, n)
    combs = (tmp['0'] + tmp['1'] + tmp['2'])
    combs['0'] = tmp['0']
    combs['1'] = tmp['1']
    combs['2'] = tmp['2']
    return combs

def cross(first, second):
    tmp = ak.cartesian([first, second])
    combs = (tmp['0'] + tmp['1'])
    combs['0'] = tmp['0']
    combs['1'] = tmp['1']
    return combs

def getNonPromptFromMatch(obj):
    return ak.num(obj[obj.genPartIdx<0])

def getNonPromptFromFlavour(obj, allow_tau=True):
    # gamma* -> ll is always treated as prompt in NanoAOD
    if allow_tau:
        return ak.num(obj[((obj.genPartFlav!=1) & (obj.genPartFlav!=15))]) # this treats tau->enu / tau->munu as prompt
    else:
        return ak.num(obj[(obj.genPartFlav!=1)])

def getChargeFlips(obj, gen=0):
    # gen is not needed, but keep to not break things
    return ak.num(obj[(obj.matched_gen.pdgId/abs(obj.matched_gen.pdgId) != obj.pdgId/abs(obj.pdgId))])

with open(os.path.expandvars('$TWHOME/data/objects.yaml')) as f:
    obj_def = load(f, Loader=Loader)

prompt    = lambda x: x[((x.genPartFlav==1)|(x.genPartFlav==15))]
#prompt_mask = lambda x: (x.genPartFlav==1)|(x.genPartFlav==15)

nonprompt = lambda x: x[((x.genPartFlav!=1)&(x.genPartFlav!=15)&(x.genPartFlav!=22))]
#nonprompt = lambda x: x[((x.genPartFlav!=1)&(x.genPartFlav!=15)&(x.genPartFlav!=22))]

conversion = lambda x: x[(x.genPartFlav==22)]

chargeflip = lambda x: x[((x.matched_gen.pdgId*(-1) == x.pdgId) & (abs(x.pdgId) == 11))]  # we only care about electron charge flips

def nonprompt_no_conv(reco_lep, gen_photon):
    sel = ((reco_lep.genPartFlav!=1)&(reco_lep.genPartFlav!=15)&(reco_lep.genPartFlav!=22)&\
            ~match_with_pt(reco_lep, gen_photon, deltaRCut=0.3, ptCut=0.5))
    return reco_lep[sel]


class Collections:

    def __init__(self, ev, obj, wp, year=2018, verbose=0):
        self.obj = obj
        self.wp = wp
        if self.wp == None:
            self.selection_dict = {}
        else:
            self.selection_dict = obj_def[self.obj][self.wp]

        self.v = verbose
        self.year = year

        id_level = None
        if wp.lower().count('veto'):
            id_level = 0
        elif wp.lower().count('fake'):
            id_level = 1
        elif wp.lower().count('tight'):
            id_level = 2
        
        if self.obj == "Muon":
            # collections are already there, so we just need to calculate missing ones
            ev['Muon', 'absMiniIso'] = ev.Muon.miniPFRelIso_all*ev.Muon.pt
            ev['Muon', 'ptErrRel']   = ev.Muon.ptErr/ev.Muon.pt

            # this is what we are using:
            # - jetRelIso if the matched jet is within deltaR<0.4, pfRelIso03_all otherwise
            # - btagDeepFlavB discriminator of the matched jet if jet is within deltaR<0.4, 0 otherwise
            # - pt_cone = 0.9*pt of matched jet if jet is within deltaR<0.4, pt/(pt+iso) otherwise

            mask_close = (ak.fill_none(ev.Muon.delta_r(ev.Muon.matched_jet),99)<0.4)*1
            mask_far = ~(ak.fill_none(ev.Muon.delta_r(ev.Muon.matched_jet),99)<0.4)*1

            deepJet = ak.fill_none(ev.Muon.matched_jet.btagDeepFlavB, 0)*mask_close + 0*mask_far
            jetRelIsoV2 = ev.Muon.jetRelIso*mask_close + ev.Muon.pfRelIso03_all*mask_far  # default to 0 if no match
            conePt = 0.9 * ak.fill_none(ev.Muon.matched_jet.pt,0) * mask_close + (ev.Muon.pt*(1 + ev.Muon.miniPFRelIso_all))*mask_far
            #conePt = 0.8 * ak.fill_none(ev.Muon.matched_jet.pt,0) * mask_close + (ev.Muon.pt/(1 + ev.Muon.miniPFRelIso_all))*mask_far

            ev['Muon', 'deepJet'] = ak.copy(deepJet)
            ev['Muon', 'jetRelIsoV2'] = jetRelIsoV2
            ev['Muon', 'conePt'] = conePt
            ev['Muon', 'id'] = ak.ones_like(conePt)*id_level

            self.cand = ev.Muon
            
        elif self.obj == "Electron":
            # calculate new variables. asignment is awkward, but what can you do.
            ev['Electron', 'absMiniIso'] = ev.Electron.miniPFRelIso_all*ev.Electron.pt
            ev['Electron', 'etaSC'] = ev.Electron.eta + ev.Electron.deltaEtaSC

            # the following line is only needed if we do our own matching.
            # right now, we keep using the NanoAOD match, but check the deltaR distance
            # jet_index, mask_match, mask_nomatch = self.matchJets(ev.Electron, ev.Jet)

            # this is what we are using:
            # - jetRelIso if the matched jet is within deltaR<0.4, pfRelIso03_all otherwise
            # - btagDeepFlavB discriminator of the matched jet if jet is within deltaR<0.4, 0 otherwise
            # - pt_cone = 0.9*pt of matched jet if jet is within deltaR<0.4, pt/(pt+iso) otherwise

            mask_close = (ak.fill_none(ev.Electron.delta_r(ev.Electron.matched_jet),99)<0.4)*1
            mask_far = ~(ak.fill_none(ev.Electron.delta_r(ev.Electron.matched_jet),99)<0.4)*1

            deepJet = ak.fill_none(ev.Electron.matched_jet.btagDeepFlavB, 0)*mask_close
            jetRelIsoV2 = ev.Electron.jetRelIso*mask_close + ev.Electron.pfRelIso03_all*mask_far  # default to 0 if no match
            conePt = 0.9 * ak.fill_none(ev.Electron.matched_jet.pt,0) * mask_close + (ev.Electron.pt*(1 + ev.Electron.miniPFRelIso_all))*mask_far
            #conePt = 0.8 * ak.fill_none(ev.Electron.matched_jet.pt,0) * mask_close + (ev.Electron.pt/(1 + ev.Electron.miniPFRelIso_all))*mask_far

            ev['Electron', 'deepJet'] = ak.copy(deepJet)
            ev['Electron', 'jetRelIsoV2'] = jetRelIsoV2
            ev['Electron', 'conePt'] = conePt
            ev['Electron', 'id'] = ak.ones_like(conePt)*id_level
            
            self.cand = ev.Electron
            
        self.getSelection()
        
        if self.obj == "Electron" and self.wp == "tight":
            self.selection = self.selection & self.getElectronMVAID() & self.getIsolation(0.07, 0.78, 8.0) & self.isTriggerSafeNoIso()
            if self.v>0: print (" - custom ID and multi-isolation")

        if self.obj == "Muon" and self.wp == "tight":
            self.selection = self.selection & self.getIsolation(0.11, 0.74, 6.8)
            if self.v>0: print (" - custom multi-isolation")
            #self.selection = self.selection & ak.fill_none(ev.Muon.matched_jet.btagDeepFlavB<0.2770, True)
            #self.selection = self.selection & (ev.Muon.matched_jet.btagDeepFlavB<0.2770)
            #if self.v>0: print (" - deepJet")

        if self.obj == "Electron" and (self.wp == "tightTTH" or self.wp == 'fakeableTTH' or self.wp == "tightSSTTH" or self.wp == 'fakeableSSTTH'):
            self.selection = self.selection & self.getSigmaIEtaIEta()
            if self.v>0: print (" - SigmaIEtaIEta")
            #self.selection = self.selection & ak.fill_none(ev.Electron.matched_jet.btagDeepFlavB<0.2770, True)
            #self.selection = self.selection & (ev.Electron.matched_jet.btagDeepFlavB<0.2770)
            #self.selection = self.selection & (ev.Jet[ev.Electron.jetIdx].btagDeepFlavB<0.2770)
            #if self.v>0: print (" - deepJet")

        if self.obj == 'Muon' and (self.wp == 'fakeableTTH' or self.wp == 'fakeableSSTTH'):
            #self.selection = self.selection & (self.cand.deepJet < self.getThreshold(self.cand.conePt, min_pt=20, max_pt=45, low=0.2770, high=0.0494))
            self.selection = self.selection & (ak.fill_none(ev.Muon.matched_jet.btagDeepFlavB,0) < self.getThreshold(self.cand.conePt, min_pt=20, max_pt=45))
            if self.v>0: print (" - interpolated deepJet")
        
    def getValue(self, var):
        #return np.nan_to_num(getattr(self.cand, var), -999)
        return getattr(self.cand, var)

    def matchJets(self, obj, jet, deltaRCut=0.4):

        combs = ak.cartesian([obj, jet], nested=True)

        jet_index = ak.local_index(delta_r(combs['0'], combs['1']))[delta_r(combs['0'], combs['1'])<0.4]
        jet_index_pad = ak.flatten(
                        ak.fill_none(
                            ak.pad_none(jet_index, target=1, clip=True, axis=2),
                        0),
                    axis=2)

        mask = ak.num(jet_index, axis=2)>0  # a mask for obj with a matched jet
        mask_match = mask*1 + ~mask*0
        mask_nomatch = mask*0 + ~mask*1

        return jet_index_pad, mask_match, mask_nomatch

    
    def getSelection(self):
        self.selection = (self.cand.pt>0)
        if self.wp == None: return
        if self.v>0:
            print ()
            print ("## %s selection for WP %s ##"%(self.obj, self.wp))
        for var in obj_def[self.obj][self.wp].keys():
            #print (var)
            if type(obj_def[self.obj][self.wp][var]) == type(1):
                if self.v>0: print (" - %s == %s"%(var, obj_def[self.obj][self.wp][var]))
                self.selection = self.selection & ( self.getValue(var) == obj_def[self.obj][self.wp][var])
            else:
                extra = obj_def[self.obj][self.wp][var].get('extra')
                if extra=='abs':
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) >= obj_def[self.obj][self.wp][var][self.year]['min'])
                        if self.v>0: print (" - abs(%s) >= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) >= obj_def[self.obj][self.wp][var]['min'])
                        if self.v>0: print (" - abs(%s) >= %s"%(var, obj_def[self.obj][self.wp][var]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) <= obj_def[self.obj][self.wp][var][self.year]['max'])
                        if self.v>0: print (" - abs(%s) <= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['max']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (abs(self.getValue(var)) <= obj_def[self.obj][self.wp][var]['max'])
                        if self.v>0: print (" - abs(%s) <= %s"%(var, obj_def[self.obj][self.wp][var]['max']))
                    except:
                        pass
                else:
                    try:
                        self.selection = self.selection & (self.getValue(var) >= obj_def[self.obj][self.wp][var][self.year]['min'])
                        if self.v>0: print (" - %s >= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) >= obj_def[self.obj][self.wp][var]['min'])
                        if self.v>0: print (" - %s >= %s"%(var, obj_def[self.obj][self.wp][var]['min']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) <= obj_def[self.obj][self.wp][var][self.year]['max'])
                        if self.v>0: print (" - %s <= %s"%(var, obj_def[self.obj][self.wp][var][self.year]['max']))
                    except:
                        pass
                    try:
                        self.selection = self.selection & (self.getValue(var) <= obj_def[self.obj][self.wp][var]['max'])
                        if self.v>0: print (" - %s <= %s"%(var, obj_def[self.obj][self.wp][var]['max']))
                    except:
                        pass
                    
                    
    def get(self):
        if self.v>0: print ("Found %s objects passing the selection"%sum(ak.num(self.cand[self.selection])))
        return self.cand[self.selection]

    def getSigmaIEtaIEta(self):
        return ((abs(self.cand.etaSC)<=1.479) & (self.cand.sieie<0.011)) | ((abs(self.cand.etaSC)>1.479) & (self.cand.sieie<0.030))

    def isTriggerSafeNoIso(self):
        if self.v>0: print (" - trigger safe")
        return ((abs(self.cand.etaSC)<=1.479) & (self.cand.sieie<0.011) & (self.cand.hoe<0.08) & (abs(self.cand.eInvMinusPInv)<0.01) ) | ((abs(self.cand.etaSC)>1.479) & (self.cand.sieie<0.031) & (self.cand.hoe<0.08) & (abs(self.cand.eInvMinusPInv)<0.01))
        
    def getMVAscore(self):
        MVA = np.minimum(np.maximum(self.cand.mvaFall17V2noIso, -1.0 + 1.e-6), 1.0 - 1.e-6)
        return -0.5*np.log(2/(MVA+1)-1)
    
    ## some more involved cuts from SS analysis
    def getElectronMVAID(self):
        # this should be year specific, only 2018 for now
        lowEtaCuts  = 2.597, 4.277, 2.597
        midEtaCuts  = 2.252, 3.152, 2.252
        highEtaCuts = 1.054, 2.359, 1.054
        lowEta      = ( abs(self.cand.etaSC) < 0.8 )
        midEta      = ( (abs(self.cand.etaSC) <= 1.479) & (abs(self.cand.etaSC) >= 0.8) )
        highEta     = ( abs(self.cand.etaSC) > 1.479 )
        lowPt       = ( self.cand.pt < 10 )
        midPt       = ( (self.cand.pt <= 25) & (self.cand.pt >= 10) )
        highPt      = (self.cand.pt > 25)
        
        MVA = self.getMVAscore()
        
        ll = ( lowEta & lowPt & (MVA > lowEtaCuts[2] ) )
        lm = ( lowEta & midPt & (MVA > (lowEtaCuts[0]+(lowEtaCuts[1]-lowEtaCuts[0])/15*(self.cand.pt-10)) ) )
        lh = ( lowEta & highPt & (MVA > lowEtaCuts[1] ) )

        ml = ( midEta & lowPt & (MVA > midEtaCuts[2] ) )
        mm = ( midEta & midPt & (MVA > (midEtaCuts[0]+(midEtaCuts[1]-midEtaCuts[0])/15*(self.cand.pt-10)) ) )
        mh = ( midEta & highPt & (MVA > midEtaCuts[1] ) )

        hl = ( highEta & lowPt & (MVA > highEtaCuts[2] ) )
        hm = ( highEta & midPt & (MVA > (highEtaCuts[0]+(highEtaCuts[1]-highEtaCuts[0])/15*(self.cand.pt-10)) ) )
        hh = ( highEta & highPt & (MVA > highEtaCuts[1] ) )
        if self.v>0: print (" - tight electron MVA ID")
        
        return ( ll | lm | lh | ml | mm | mh | hl | hm | hh )
    
    ## SS isolation
    def getIsolation(self, mini, jet, jetv2 ):
        # again, this is only for 2018 so far
        jetRelIso = 1/(self.cand.jetRelIso+1)
        if self.v>0: print (" - custom multi isolation")
        return ( (self.cand.miniPFRelIso_all < mini) & ( (jetRelIso>jet) | (self.cand.jetPtRelv2>jetv2) ) )

    def getThreshold(self, pt, min_pt=20, max_pt=45, low=None, high=None):
        '''
        get the deepJet threshold for ttH FO muons. default values are for 2018.
        UL values from https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
        '''
        b_low = {2016: 0.3093, 2017: 0.3040, 2018: 0.2783,}
        b_high = {2016: 0.0614, 2017: 0.0532, 2018: 0.0490,}
        if low is None: low = b_low[self.year]
        if high is None: high = b_high[self.year]

        k = (low-high)/(min_pt-max_pt)
        d = low - k*min_pt
        return (pt<min_pt)*low + ((pt>=min_pt)*(pt<max_pt)*(k*pt+d)) + (pt>=max_pt)*high