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

# load a default cache
output = get_cache('SS_analysis_2016')

# check out that the histogram we want is actually there
all_processes = [ x[0] for x in output['N_ele'].values().keys() ]
data_all = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
data    = data_all
order   = ['topW_v3', 'np_est_mc', 'conv_mc', 'cf_est_mc', 'TTW', 'TTH', 'TTZ','rare', 'diboson']
signals = []
omit    = [ x for x in all_processes if (x not in signals and x not in order and x not in data) ]

no_data_or_signal  = re.compile('(?!(%s))'%('|'.join(omit)))

score_bins = hist.Bin("score",          r"N", 8, 0, 1)

regions = [
    ('SR_1', 'node0_score_transform_pp'),
    ('SR_2', 'node0_score_transform_mm'),
    #('CR', 'node1_score'),
    #('CR_ext', 'node'),
]

mapping = {
    'rare': ['rare', 'diboson'],
    'TTW': ['TTW'],
    'TTZ': ['TTZ'],
    'TTH': ['TTH'],
    'ttbar': ['ttbar'],
    'nonprompt': ['np_est_mc'],
    'chargeflip': ['cf_est_mc'],
    'conversion': ['conv_mc'],
    'signal': ['topW_v3'],
}

## FIXME this should also become a function?
new_hists = {}
for short_name, long_name in regions:
    new_hists[short_name] = output[long_name][no_data_or_signal]
    if short_name.count('CR_ext')<1:
        new_hists[short_name] = new_hists[short_name].rebin('score', score_bins)
    else:
        new_hists[short_name] = new_hists[short_name].rebin('multiplicity', N_bins)
    new_hists[short_name] = new_hists[short_name].group("dataset", hist.Cat("dataset", "new grouped dataset"), mapping)


card = dataCard(releaseLocation=os.path.expandvars('/home/users/$USER/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

print ("- JES uncertainties for TTW:")
get_unc(output, 'node0_score_transform_pp', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
get_unc(output, 'node0_score_transform_mm', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
print ("- JES uncertainties for signal:")
get_unc(output, 'node0_score_transform_pp', 'topW_v3', '_pt_jesTotal', score_bins, quiet=False)
get_unc(output, 'node0_score_transform_mm', 'topW_v3', '_pt_jesTotal', score_bins, quiet=False)

## FIXME 

systematics = [
    ('jes', get_unc(output, 'node0_score_transform_pp', 'topW_v3', '_pt_jesTotal', score_bins), 'signal'),
    ('b', get_unc(output, 'node0_score_transform_pp', 'topW_v3', '_b', score_bins), 'signal'),
    ('light', get_unc(output, 'node0_score_transform_pp', 'topW_v3', '_l', score_bins), 'signal'),
    ('PU', get_unc(output, 'node0_score_transform_pp', 'topW_v3', '_PU', score_bins), 'signal'), 
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_pp', 'TTW', score_bins), 'TTW'),
    ('scale_TTW', get_scale_unc(output, 'node0_score_transform_pp', 'TTW', score_bins), 'TTW'),
    ('jes', get_unc(output, 'node0_score_transform_pp', 'TTW', '_pt_jesTotal', score_bins), 'TTW'),
    ('b', get_unc(output, 'node0_score_transform_pp', 'TTW', '_b', score_bins), 'TTW'),
    ('light', get_unc(output, 'node0_score_transform_pp', 'TTW', '_l', score_bins), 'TTW'),
    ('PU', get_unc(output, 'node0_score_transform_pp', 'TTW', '_PU', score_bins), 'TTW'),
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_pp', 'TTH', score_bins), 'TTH'),
    ('scale_TTH', get_scale_unc(output, 'node0_score_transform_pp', 'TTH', score_bins), 'TTH'),
    ('jes', get_unc(output, 'node0_score_transform_pp', 'TTH', '_pt_jesTotal', score_bins), 'TTH'),
    ('b', get_unc(output, 'node0_score_transform_pp', 'TTH', '_b', score_bins), 'TTH'),
    ('light', get_unc(output, 'node0_score_transform_pp', 'TTH', '_l', score_bins), 'TTH'),
    ('PU', get_unc(output, 'node0_score_transform_pp', 'TTH', '_PU', score_bins), 'TTH'),
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_pp', 'TTZ', score_bins), 'TTZ'),
    ('scale_TTZ', get_scale_unc(output, 'node0_score_transform_pp', 'TTZ', score_bins), 'TTZ'),
    ('jes', get_unc(output, 'node0_score_transform_pp', 'TTZ', '_pt_jesTotal', score_bins), 'TTZ'),
    ('b', get_unc(output, 'node0_score_transform_pp', 'TTZ', '_b', score_bins), 'TTZ'),
    ('light', get_unc(output, 'node0_score_transform_pp', 'TTZ', '_l', score_bins), 'TTZ'),
    ('PU', get_unc(output, 'node0_score_transform_pp', 'TTZ', '_PU', score_bins), 'TTZ'),
    
    #('ttz_norm', 1.10, 'TTZ'),
    #('tth_norm', 1.20, 'TTH'),
    ('rare_norm', 1.20, 'rare'),
    ('nonprompt_norm', 1.30, 'nonprompt'),
    ('chargeflip_norm', 1.20, 'chargeflip'),
    ('conversion_norm', 1.20, 'conversion')
]

sm_card_sr1 = makeCardFromHist(
    new_hists,
    'SR_1',
    overflow='all',
    ext='',
    systematics = systematics,
)


systematics = [
    ('jes', get_unc(output, 'node0_score_transform_mm', 'topW_v3', '_pt_jesTotal', score_bins), 'signal'),
    ('b', get_unc(output, 'node0_score_transform_mm', 'topW_v3', '_b', score_bins), 'signal'),
    ('light', get_unc(output, 'node0_score_transform_mm', 'topW_v3', '_l', score_bins), 'signal'),
    ('PU', get_unc(output, 'node0_score_transform_mm', 'topW_v3', '_PU', score_bins), 'signal'),    
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_mm', 'TTW', score_bins), 'TTW'),
    ('scale_TTW', get_scale_unc(output, 'node0_score_transform_mm', 'TTW', score_bins), 'TTW'),
    ('jes', get_unc(output, 'node0_score_transform_mm', 'TTW', '_pt_jesTotal', score_bins), 'TTW'),
    ('b', get_unc(output, 'node0_score_transform_mm', 'TTW', '_b', score_bins), 'TTW'),
    ('light', get_unc(output, 'node0_score_transform_mm', 'TTW', '_l', score_bins), 'TTW'),
    ('PU', get_unc(output, 'node0_score_transform_mm', 'TTW', '_PU', score_bins), 'TTW'),
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_mm', 'TTH', score_bins), 'TTH'),
    ('scale_TTH', get_scale_unc(output, 'node0_score_transform_mm', 'TTH', score_bins), 'TTH'),
    ('jes', get_unc(output, 'node0_score_transform_mm', 'TTH', '_pt_jesTotal', score_bins), 'TTH'),
    ('b', get_unc(output, 'node0_score_transform_mm', 'TTH', '_b', score_bins), 'TTH'),
    ('light', get_unc(output, 'node0_score_transform_mm', 'TTH', '_l', score_bins), 'TTH'),
    ('PU', get_unc(output, 'node0_score_transform_mm', 'TTH', '_PU', score_bins), 'TTH'),
    
    ('pdf', get_pdf_unc(output, 'node0_score_transform_mm', 'TTZ', score_bins), 'TTZ'),
    ('scale_TTZ', get_scale_unc(output, 'node0_score_transform_mm', 'TTZ', score_bins), 'TTZ'),
    ('jes', get_unc(output, 'node0_score_transform_mm', 'TTZ', '_pt_jesTotal', score_bins), 'TTZ'),
    ('b', get_unc(output, 'node0_score_transform_mm', 'TTZ', '_b', score_bins), 'TTZ'),
    ('light', get_unc(output, 'node0_score_transform_mm', 'TTZ', '_l', score_bins), 'TTZ'),
    ('PU', get_unc(output, 'node0_score_transform_mm', 'TTZ', '_PU', score_bins), 'TTZ'),
    
    #('ttz_norm', 1.10, 'TTZ'),
    #('tth_norm', 1.20, 'TTH'),
    ('rare_norm', 1.20, 'rare'),
    ('nonprompt_norm', 1.30, 'nonprompt'),
    ('chargeflip_norm', 1.20, 'chargeflip'),
    ('conversion_norm', 1.20, 'conversion')
]

sm_card_sr2 = makeCardFromHist(
    new_hists,
    'SR_2',
    overflow='all',
    ext='',
    systematics = systematics,
)

data_cards = {'2016_SR1': sm_card_sr1, '2016_SR2': sm_card_sr2}

combined = card.combineCards(data_cards)

result_combined = card.nllScan(combined, rmin=0, rmax=3, npoints=61, options=' -v -1')

print ("Significance: %.2f sigma"%np.sqrt(result_combined['deltaNLL'][1]*2))

run_impacts = True
if run_impacts:
    card.run_impacts(combined, plot_dir='/home/users/dspitzba/public_html/tW_scattering/')

card.cleanUp()
