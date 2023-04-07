#!/usr/bin/env python3
import numpy as np
from coffea import processor, hist, util
from coffea.processor import accumulate

from analysis.Tools.config_helpers import loadConfig, make_small, lumi, load_yaml, get_merged_output
from analysis.Tools.samples import Samples

if __name__ == '__main__':

    # define the scan
    xr = np.arange(-7,8,1)
    yr = np.arange(-7,8,1)
    X, Y = np.meshgrid(xr, yr)
    scan = zip(X.flatten(), Y.flatten())

    year = '2018'

    outputs = []
    if year == '2019':
        lumi = sum([lumi[y] for y in years])
    else:
        lumi = lumi[year]

    samples = Samples.from_yaml(f'analysis/Tools/data/samples_v0_8_0_SS.yaml')
    mapping = load_yaml('analysis/Tools/data/nano_mapping.yaml')

    for x, y in scan:
        #print(x,y)
        postfix = f'_cpt_{x}_cpqm_{y}_fixed'
        outputs.append(get_merged_output(
            'SS_analysis',
            year,
            './outputs/',
            samples, mapping,
            lumi=lumi,
            postfix=postfix,
            variations = ['central', 'fake', 'base', 'jes'],
            select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'],
            ))

    output = accumulate(outputs)
    util.save(output, f'./outputs/{year}_fixed_merged.coffea')  # this will just always be the latest one
