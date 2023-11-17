#!/usr/bin/env python3

# JES uncertainties from here https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources
variations_jet_all_list = [
     # all kept in v0.8.0: https://github.com/danbarto/nanoAOD-tools/blob/topW_v0.8.0/scripts/run_processor.py#L96
     'jesAbsoluteStat',
     'jesAbsoluteScale',
     'jesAbsoluteMPFBias',
     'jesFragmentation',
     'jesSinglePionECAL',
     'jesSinglePionHCAL',
     'jesFlavorQCD',  # default, others currently not kept in our samples
     'jesTimePtEta',
     'jesRelativeJEREC1',
     'jesRelativeJEREC2',
     'jesRelativeJERHF',
     'jesRelativePtBB',
     'jesRelativePtEC1',
     'jesRelativePtEC2',
     'jesRelativePtHF',
     'jesRelativeBal',
     'jesRelativeSample',
     'jesRelativeFSR',
     'jesRelativeStatFSR',
     'jesRelativeStatEC',
     'jesRelativeStatHF',
     'jesPileUpDataMC',
     'jesPileUpPtRef',
     'jesPileUpPtBB',
     'jesPileUpPtEC1',
     'jesPileUpPtEC2',
     'jesPileUpPtHF',
]
