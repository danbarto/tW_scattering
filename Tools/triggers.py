import numpy as np
import awkward1 as ak
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


def getFilters(ev, year=2018, dataset='None'):
    #filters, recommendations in https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
    if year == 2018:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "ecalBadCalibFilterV2"
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
            "ecalBadCalibFilterV2"
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
        
def getTriggers(ev, year=2018, dataset='None'):
    # these are the MET triggers from the MT2 analysis
    
    triggers = {}
    
    if year == 2018:
        triggers['MuonEG'] = [\
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu27_Ele37_CaloIdL_MW",
            "Mu37_Ele27_CaloIdL_MW",
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
        
    if dataset.lower().count('muon') or dataset.lower().count('electron'):
        return mask_or(ev, "HLT", triggers[dataset])
    else:
        return mask_or(ev, "HLT", triggers['MuonEG']) # should be OR of all dilepton triggers

