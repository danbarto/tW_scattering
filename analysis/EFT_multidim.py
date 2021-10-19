'''
This should run the final analyis:
- pick up cached histograms
- rebin distributions
- create inputs for data card
- run fits
'''

import os
import re

import numpy as np
import pandas as pd

from coffea import hist

from klepto.archives import dir_archive

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot
from Tools.config_helpers import get_cache
from Tools.limits import get_unc, get_pdf_unc, get_scale_unc, makeCardFromHist
from Tools.yahist_to_root import yahist_to_root
from Tools.dataCard import dataCard

from Tools.HyperPoly import HyperPoly 

'''
Taken from the NanoAOD-tools module
'''

def get_points(points):
    points = points.replace('LHEWeight_','').replace('_nlo','')
    ops = points.split('_')[::2]
    vals = [ float(x.replace('p','.')) for x in points.split('_')[1::2] ]
    return dict(zip(ops, vals))

def get_coordinates(points):
    points = points.replace('LHEWeight_','').replace('_nlo','')
    vals = [ float(x.replace('p','.')) for x in points.split('_')[1::2] ]
    return tuple(vals)


if __name__ == '__main__':

    output_EFT = get_cache('EFT_ctW_scan_2016APV')
    eft_mapping = {k[0]:k[0] for k in output_EFT['LT_SR_pp'].values().keys() if 'topW_full_EFT_ctZ' in k[0]}  # not strictly necessary
    weights = [ k[0].replace('topW_full_EFT_','').replace('_nlo','') for k in output_EFT['LT_SR_pp'].values().keys() if 'topW_full_EFT_ctZ' in k[0] ]
    
    ref_point = 'ctZ_2p_cpt_4p_cpQM_4p_cpQ3_4p_ctW_2p_ctp_2p'
    ref_values = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]

    hp = HyperPoly(order=2)
    hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

    
