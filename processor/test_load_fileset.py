#!/usr/bin/env python3


import os
import re
import datetime
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist, util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np
import pandas as pd
import pickle

from Tools.objects import Collections, getNonPromptFromFlavour, getChargeFlips, prompt, nonprompt, choose, cross, delta_r, delta_r2, match, prompt_no_conv, nonprompt_no_conv, external_conversion, fast_match
from Tools.basic_objects import getJets, getTaus, getIsoTracks, getBTagsDeepFlavB, getFwdJet, getMET
from Tools.cutflow import Cutflow
from Tools.helpers import pad_and_flatten, mt, fill_multiple, zip_run_lumi_event, get_four_vec_fromPtEtaPhiM, get_samples
from Tools.config_helpers import loadConfig, make_small, data_pattern, get_latest_output, load_yaml, data_path
from Tools.triggers import getFilters, getTriggers
from Tools.btag_scalefactors import btag_scalefactor
from Tools.trigger_scalefactors import triggerSF
from Tools.ttH_lepton_scalefactors import LeptonSF
from Tools.selections import Selection, get_pt
from Tools.nonprompt_weight import NonpromptWeight
from Tools.chargeFlip import charge_flip
from Tools.pileup import pileup

import warnings
warnings.filterwarnings("ignore")

from ML.multiclassifier_tools import load_onnx_model, predict_onnx, load_transformer
from BoostedInformationTreeP3 import BoostedInformationTree
from analysis.boosted_information_tree import get_bit_score

from sklearn.preprocessing import QuantileTransformer


if __name__ == '__main__':

    from processor.default_accumulators import *

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=None, help="Rerun or try using existing results??")
    argParser.add_argument('--minimal', action='store_true', default=None, help="Only run minimal set of histograms")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on a DASK cluster?")
    argParser.add_argument('--central', action='store_true', default=None, help="Only run the central value (no systematics)")
    argParser.add_argument('--profile', action='store_true', default=None, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=None, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=None, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    argParser.add_argument('--sample', action='store', default='all', )
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = args.rerun
    small       = args.small

    year        = int(args.year[0:4])
    ul          = "UL%s"%(args.year[2:])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    if profile:
        from pympler import muppy, summary

    # load the config
    cfg = loadConfig()

    samples = get_samples("samples_%s.yaml"%ul)
    mapping = load_yaml(data_path+"nano_mapping.yaml")


    if args.sample == 'MCall':
        sample_list = ['DY', 'topW_lep', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
    elif args.sample == 'data':
        if year == 2018:
            sample_list = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        else:
            sample_list = ['DoubleMuon', 'MuonEG', 'DoubleEG', 'SingleMuon', 'SingleElectron']
    else:
        sample_list = [args.sample]

    cutflow_output = {}

    for sample in sample_list:
        # NOTE we could also rescale processes here?
        #
        print (f"Working on samples: {sample}")

        # NOTE we could also rescale processes here?
        reweight = {}
        renorm   = {}
        for dataset in mapping[ul][sample]:
            if samples[dataset]['reweight'] == 1:
                reweight[dataset] = 1
                renorm[dataset] = 1
            else:
                # Currently only supporting a single reweight.
                weight, index = samples[dataset]['reweight'].split(',')
                index = int(index)
                renorm[dataset] = samples[dataset]['sumWeight']/samples[dataset][weight][index]  # NOTE: needs to be divided out
                reweight[dataset] = (weight, index)

        from Tools.nano_mapping import make_fileset
        fileset = make_fileset([sample], samples, year=ul, skim=True, small=small, n_max=1, buaf=True)
