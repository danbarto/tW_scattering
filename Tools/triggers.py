import numpy as np
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

import re
## happily borrowed from https://github.com/bu-cms/bucoffea/blob/master/bucoffea/helpers/helpers.py

def mask_or(ev, collection, masks):
    """Returns the OR of the masks in the list
    :param ev: NanoEvents
    :type ev: NanoEvents
    :param collection: HLT or Filter
    "type collection: string
    :param masks: Mask names as saved in the df
    :type masks: List
    :return: OR of all masks for each event
    :rtype: array
    """
    # Start with array of False
    decision = ( ak.ones_like(ev.MET.pt)==0 )

    coll = getattr(ev, collection)

    # Flip to true if any is passed
    for t in masks:
        try:
            decision = decision | getattr(coll, t)
        except KeyError:
            continue
    return decision

def mask_and(ev, collection, masks):
    """Returns the AND of the masks in the list
    :param ev: NanoEvents
    :type ev: NanoEvents
    :param collection: HLT or Filter
    "type collection: string
    :param masks: Mask names as saved in the df
    :type masks: List
    :return: OR of all masks for each event
    :rtype: array
    """
    # Start with array of True
    decision = ( ak.ones_like(ev.MET.pt)==1 )

    coll = getattr(ev, collection)

    # Flip to true if any is passed
    for t in masks:
        try:
            decision = decision & getattr(coll, t)
        except KeyError:
            continue
    return decision


def getFilters(ev, year=2018, dataset='None', UL=True):
    # filters, recommendations in https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
    #FIXME Flag_BadPFMuonDzFilter missing in EOY UL?? Should be added.
    if year == 2018:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "ecalBadCalibFilterV2" if not UL else "ecalBadCalibFilter",
        ]
        
        filters_data = filters_MC + ["eeBadScFilter"]
        
    elif year == 2017:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "ecalBadCalibFilterV2" if not UL else "ecalBadCalibFilter",
        ]
        
        filters_data = filters_MC + ["eeBadScFilter"]

    elif year == 2016:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
        ]
        
        filters_data = filters_MC + ["eeBadScFilter"]
        
    if dataset.lower().count('muon') or dataset.lower().count('electron'):
        return mask_and(ev, "Flag", filters_data)
    else:
        return mask_and(ev, "Flag", filters_MC)
        
def getTriggers(ev, leading_pdg, subleading_pdg, year=2018, dataset='None'):
    # these are the MET triggers from the MT2 analysis
    
    triggers = {}
    
    same_flavor = (abs(leading_pdg) == abs(subleading_pdg))
    leading_ele = (abs(leading_pdg) == 11)
    leading_mu  = (abs(leading_pdg) == 13)


    # lepton triggers from here: https://indico.cern.ch/event/718554/contributions/3027981/attachments/1667626/2674497/leptontriggerreview.pdf
    if year == 2018:
        triggers['MuonEG'] = [\
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu27_Ele37_CaloIdL_MW",
            "Mu37_Ele27_CaloIdL_MW",
        ]

        triggers['DoubleMuon'] = [\
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "Mu37_TkMu27",
        ]

        triggers['DoubleEG'] = [\
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "DoubleEle25_CaloIdL_MW",
        ]

        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMET120_PFMHT120_IDTight_PFHT60",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        ]
        
    elif year == 2017:
        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
        ]
        
    elif year == 2016:
        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
        ]
        
    #print (dataset)
    if re.search(re.compile("MuonEG"), dataset):
        #print ("In MuonEG branch")
        trigger = mask_or(ev, "HLT", triggers["MuonEG"])
        #print (sum(trigger & ~same_flavor))
        return (trigger & ~same_flavor)
        #return trigger

    elif re.search(re.compile("DoubleMuon"), dataset):
        #print ("In DoubleMuon branch")
        trigger = mask_or(ev, "HLT", triggers["DoubleMuon"])
        #print (sum(trigger & same_flavor & leading_mu))
        return (trigger & same_flavor & leading_mu)
        #return trigger

    elif re.search(re.compile("DoubleEG|EGamma"), dataset):
        #print ("In EGamma branch")
        trigger = mask_or(ev, "HLT", triggers["DoubleEG"])
        #print (sum(trigger & same_flavor & leading_ele))
        return (trigger & same_flavor & leading_ele)

    else:
        #print ("In MC branch")
        # these triggers aren't fully efficient yet. check if we're missing something.
        mm = (mask_or(ev, "HLT", triggers['DoubleMuon']) & same_flavor & leading_mu)
        ee = (mask_or(ev, "HLT", triggers['DoubleEG']) & same_flavor & leading_ele)
        em = (mask_or(ev, "HLT", triggers['MuonEG']) & ~same_flavor)
        return (mm | ee | em)


    #if re.search(re.compile("MuonEG|DoubleMuon|DoubleEG|EGamma|SingleMuon|SingleElectron"), dataset):  #  dataset.lower().count('muon') or dataset.lower().count('electron'):
    #    return mask_or(ev, "HLT", triggers[dataset])
    #else:
    #    return mask_or(ev, "HLT", triggers['MuonEG'] + triggers['DoubleMuon'] + triggers['DoubleEG'])  # should be OR of all dilepton triggers

