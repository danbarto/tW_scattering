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
        except AttributeError:
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
        except AttributeError:
            continue
    return decision


def getFilters(ev, year=2018, dataset='None', UL=True):
    # filters, recommendations in https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
    if year == 2018:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",  # NOTE: this is still optional
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ]
        
        filters_data = filters_MC + []
        
    elif year == 2017:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",  # NOTE: this is still optional
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ]
        
        filters_data = filters_MC + []

    elif year == 2016:
        filters_MC = [\
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            #"hfNoisyHitsFilter",  # FIXME: still being studied
            "eeBadScFilter",
        ]
        
        filters_data = filters_MC + []
        
    if dataset.lower().count('muon') or dataset.lower().count('electron'):
        return mask_and(ev, "Flag", filters_data)
    else:
        return mask_and(ev, "Flag", filters_MC)
        
def getTriggers(ev, leading_pdg=[], subleading_pdg=[], year=2018, dataset='None', era=None):
    # these are the MET triggers from the MT2 analysis
    
    triggers = {}
    
    #same_flavor = (abs(leading_pdg) == abs(subleading_pdg))
    #leading_ele = (abs(leading_pdg) == 11)
    #leading_mu  = (abs(leading_pdg) == 13)


    # lepton triggers from here: https://indico.cern.ch/event/718554/contributions/3027981/attachments/1667626/2674497/leptontriggerreview.pdf
    # TOP triggers are here: https://indico.cern.ch/event/995560/contributions/4189577/attachments/2174069/3671077/Dilepton_TriggerSF_TOPPAG.pdf
    #FIXME how to include single lepton triggers / PDs without introducing overlap?
    if year == 2018:
        triggers['MuonEG'] = [\
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
            "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
            #"Mu27_Ele37_CaloIdL_MW",  #FIXME this is not in the TOP list
            #"Mu37_Ele27_CaloIdL_MW",  #FIXME this is not in the TOP list
        ]

        triggers['DoubleMuon'] = [\
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "TripleMu_12_10_5",
            #"Mu37_TkMu27",  #FIXME this is not in the TOP list
        ]

        triggers['DoubleEG'] = [\
            "Ele32_WPTight_Gsf",
            "Ele35_WPTight_Gsf",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            #"DoubleEle25_CaloIdL_MW",
        ]

        triggers['SingleMuon'] = [\
            "IsoMu24",
            "IsoMu27",
        ]

        triggers['SingleElectron'] = [\
        ]

        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMET120_PFMHT120_IDTight_PFHT60",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        ]
        
    elif year == 2017:
        triggers['MuonEG'] = [\
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "Mu8_DiEle12_CaloIdL_TrackIdL_DZ", # Note: Listed in Andrew's thesis, but not TOP-19-001 AN
            "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
        ]

        triggers['DoubleMuon'] = [\
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "TripleMu_12_10_5",
            #"Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
        ]

        triggers['DoubleEG'] = [\
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            #"DoubleEle33_CaloIdL_MW",
        ]

        triggers['SingleMuon'] = [\
            "IsoMu24",
            "IsoMu27",
        ]

        triggers['SingleElectron'] = [\
            "Ele32_WPTight_Gsf",
            "Ele35_WPTight_Gsf",
        ]

        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
        ]
        
    elif year == 2016:
        # no era dependence anymore.
        if not era=='APV':
            triggers['MuonEG'] = [\
            #    "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            #    "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            ]
        else:
            triggers['MuonEG'] = []

        triggers['MuonEG'] += [\
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "DiMu9_Ele9_CaloIdL_TrackIdL",
        ]

        triggers['DoubleMuon'] = [\
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
            "TripleMu_12_10_5",
        ]

        triggers['DoubleEG'] = [\
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
            #"DoubleEle33_CaloIdL_MW",
            #"DoubleEle33_CaloIdL_GsfTrkIdVL",
        ]

        triggers['SingleMuon'] = [\
            "IsoMu24",
            "IsoTkMu24",
            "IsoMu22_eta2p1",
            "IsoTkMu22_eta2p1",
            "IsoMu22",
            "IsoTkMu22",
            "IsoMu27",
        ]

        triggers['SingleElectron'] = [\
            "Ele27_WPTight_Gsf",
            "Ele25_eta2p1_WPTight_Gsf",
            "Ele27_eta2p1_WPLoose_Gsf",
        ]

        triggers['MET'] = [\
            "PFMET120_PFMHT120_IDTight",
            "PFMETNoMu120_PFMHTNoMu120_IDTight",
        ]
        
    if re.search(re.compile("DoubleMuon"), dataset):
        trigger = mask_or(ev, "HLT", triggers["DoubleMuon"])
        return trigger

    elif re.search(re.compile("SingleMuon"), dataset):
        trigger = (mask_or(ev, "HLT", triggers["SingleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleMuon"]))
        return trigger

    elif re.search(re.compile("DoubleEG|EGamma"), dataset):  # 
        trigger = (mask_or(ev, "HLT", triggers["DoubleEG"]) & \
                  ~mask_or(ev, "HLT", triggers["SingleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleMuon"]))
        return trigger

    elif re.search(re.compile("SingleElectron"), dataset):
        trigger = (mask_or(ev, "HLT", triggers["SingleElectron"]) & \
                  ~mask_or(ev, "HLT", triggers["SingleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleEG"]))
        return trigger

    elif re.search(re.compile("MuonEG"), dataset):
        trigger = (mask_or(ev, "HLT", triggers["MuonEG"]) & \
                  ~mask_or(ev, "HLT", triggers["SingleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleMuon"]) & \
                  ~mask_or(ev, "HLT", triggers["DoubleEG"]) & \
                  ~mask_or(ev, "HLT", triggers["SingleElectron"]))
        return trigger



    else:

        mm = (mask_or(ev, "HLT", triggers['DoubleMuon']) )
        ee = (mask_or(ev, "HLT", triggers['DoubleEG']) )
        em = (mask_or(ev, "HLT", triggers['MuonEG']) )
        m = (mask_or(ev, "HLT", triggers['SingleMuon']) )
        e = (mask_or(ev, "HLT", triggers['SingleElectron']) )
        trig = (mm | ee | em | m | e)
        return trig

