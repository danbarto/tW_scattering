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
N_bins     = hist.Bin("multiplicity",   r"N", 3, 1.5, 4.5)

regions = [
    ('SR_1', 'node0_score_transform_pp'),
    ('SR_2', 'node0_score_transform_mm'),
    ('CR', 'node1_score'),
    ('CR_ext', 'node'),
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

# FIXME I should just rebin / map everything at the same time.
from Tools.limits import regroup_and_rebin, get_systematics
# First, rebin & map
for k in output.keys():
    if 'node0_score' in k:
        output[k] = regroup_and_rebin(output[k], score_bins, mapping)
    elif 'node1_score' in k:
        output[k] = regroup_and_rebin(output[k], score_bins, mapping)
    elif k.startswith('node') and not k.count('score'):
        output[k] = regroup_and_rebin(output[k], N_bins, mapping)

# then make copies for SR and CR
new_hists = {}
for short_name, long_name in regions:
    new_hists[short_name] = output[long_name][no_data_or_signal]

card = dataCard(releaseLocation=os.path.expandvars('/home/users/$USER/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'))

print ("- JES uncertainties for TTW:")
get_unc(output, 'node0_score_transform_pp', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
get_unc(output, 'node0_score_transform_mm', 'TTW', '_pt_jesTotal', score_bins, quiet=False)
print ("- JES uncertainties for signal:")
get_unc(output, 'node0_score_transform_pp', 'signal', '_pt_jesTotal', score_bins, quiet=False)
get_unc(output, 'node0_score_transform_mm', 'signal', '_pt_jesTotal', score_bins, quiet=False)

sm_card_sr1 = makeCardFromHist(
    new_hists,
    'SR_1',
    overflow='all',
    ext='',
    systematics = get_systematics(output, 'node0_score_transform_pp'),
)

sm_card_sr2 = makeCardFromHist(
    new_hists,
    'SR_2',
    overflow='all',
    ext='',
    systematics = get_systematics(output, 'node0_score_transform_mm'),
)

sm_card_cr = makeCardFromHist(
    new_hists,
    'CR',
    overflow='all',
    ext='',
    systematics = get_systematics(output, 'node1_score'),
)

sm_card_cr_ext = makeCardFromHist(
    new_hists,
    'CR_ext',
    overflow='all',
    ext='',
    systematics = get_systematics(output, 'node'),
)

data_cards = {
    '2016_SR1': sm_card_sr1,
    '2016_SR2': sm_card_sr2,
    '2016_CR': sm_card_cr,
    '2016_CR_ext': sm_card_cr_ext,
}

combined = card.combineCards(data_cards)

result_combined = card.nllScan(combined, rmin=0, rmax=3, npoints=61, options=' -v -1')

run_impacts = True
if run_impacts:
    card.run_impacts(combined, plot_dir='/home/users/dspitzba/public_html/tW_scattering/')


print ("Significance: %.2f sigma"%np.sqrt(result_combined['deltaNLL'][1]*2))

card.cleanUp()
