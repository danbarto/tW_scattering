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
from Tools.limits import regroup_and_rebin, get_systematics
from Tools.EFT_tools import make_scan

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

def histo_values(histo, weight):
    return histo[weight].sum('dataset').values(overflow='all')[()]

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    ht_bins     = hist.Bin('ht', r'$H_{T}\ (GeV)$', [100,200,300,400,500,600,700,800])

    hist_name = 'LT_SR_mm'

    output_EFT = get_cache('EFT_ctW_scan_2017')
    eft_mapping = {k[0]:k[0] for k in output_EFT[hist_name].values().keys() if 'topW_full_EFT_ctZ' in k[0]}  # not strictly necessary
    weights = [ k[0].replace('topW_full_EFT_','').replace('_nlo','') for k in output_EFT[hist_name].values().keys() if 'topW_full_EFT_ctZ' in k[0] ]

    output_EFT[hist_name] = regroup_and_rebin(output_EFT[hist_name], ht_bins, eft_mapping)
    
    ref_point = 'ctZ_2p_cpt_4p_cpQM_4p_cpQ3_4p_ctW_2p_ctp_2p'
    ref_values = [ float(x.replace('p','.')) for x in ref_point.split('_')[1::2] ]

    hp = HyperPoly(order=2)
    hp.initialize( [get_coordinates(weight) for weight in weights], ref_values )

    coeff = hp.get_parametrization( [histo_values(output_EFT[hist_name], 'topW_full_EFT_%s_nlo'%w) for w in weights] )


    # just an example.
    points = make_scan(operator='cpQM', C_min=0, C_max=10, step=1)

    for i in range(0,11):
        print (i, hp.eval(coeff, points[i]['point']))

    pred_matrix = np.array([ np.array(hp.eval(coeff,points[i]['point'])) for i in range(11) ])

    # plot the increase in yield 
    
    fig, ax = plt.subplots()
    
    hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60.0+41.5+35.9,
        loc=0,
        ax=ax,
    )
    
    plt.plot(range(11), np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[0,:]), label=r'inclusive', c='green')
    plt.plot(range(11), np.sum(pred_matrix[:,7:], axis=1)/np.sum(pred_matrix[0,7:]), label=r'$L_{T} \geq 700\ GeV$', c='blue')
    
    plt.xlabel(r'$C$')
    plt.ylabel(r'$\sigma/\sigma_{SM}$')
    plt.legend()
    
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/cpQM_scaling.pdf')
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/Esquared/cpQM_scaling.png')
    
