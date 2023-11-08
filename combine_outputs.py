#!/usr/bin/env python3
import numpy as np
from coffea import processor, hist, util
from coffea.processor import accumulate

from analysis.Tools.config_helpers import loadConfig, make_small, lumi, load_yaml, get_merged_output
from analysis.Tools.samples import Samples

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--signal', action='store_true', help="Merge signal histograms")
    argParser.add_argument('--parametrized', action='store_true', default=None, help="Run fully parametrized, otherwise use a fixed template (hard coded to cpt=5, cpqm=-5)")
    argParser.add_argument('--trilep', action='store_true', default=None, help="Trilep outputs")
    args = argParser.parse_args()

    years = ['2016', '2016APV', '2017', '2018']
    year = args.year

    outputs = []
    if year == '2019':
        lumi = sum([lumi[y] for y in years])
    else:
        lumi = lumi[year]

    samples = Samples.from_yaml(f'analysis/Tools/data/samples_v0_8_0_SS.yaml')
    mapping = load_yaml('analysis/Tools/data/nano_mapping.yaml')

    if args.signal:
        if args.trilep:
            outputs.append(get_merged_output(
                'trilep_analysis',
                year,
                './outputs/',
                samples, mapping,
                lumi=lumi,
                postfix='',
                select_datasets = ['topW_lep'],
                variations = ['central', 'base', 'jes', 'jer'],
                select_histograms = ['dilepton_mass_ttZ', 'signal_region_topW', 'LT_WZ', 'LT_XG'],
                ))
            output = accumulate(outputs)
            util.save(output, f'./outputs/{year}_signal_trilep_merged.coffea')  # this will just always be the latest one
        else:
            outputs.append(get_merged_output(
                'SS_analysis',
                year,
                './outputs/',
                samples, mapping,
                lumi=lumi,
                postfix='_fixed',
                select_datasets = ['topW_lep'],
                variations = ['central', 'base', 'jes', 'jer'],
                select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'],
                ))
            output = accumulate(outputs)
            util.save(output, f'./outputs/{year}_signal_merged.coffea')  # this will just always be the latest one
        #
    if args.trilep and not args.signal:
        outputs.append(get_merged_output(
            'trilep_analysis',
            year,
            './outputs/',
            samples, mapping,
            lumi=lumi,
            postfix='_cpt_0_cpqm_0',
            select_datasets = ['data', 'MCall'],
            variations = ['central', 'fake', 'base', 'jes', 'jer'],
            select_histograms = ["dilepton_mass_ttZ", "signal_region_topW", "LT_WZ", "LT_XG"],
            ))

        output = accumulate(outputs)
        util.save(output, f'./outputs/{year}_trilep_merged.coffea')  # this will just always be the latest one

    elif not args.signal:

        if args.parametrized:
            # define the scan
            xr = np.arange(-7,8,1)
            yr = np.arange(-7,8,1)
            X, Y = np.meshgrid(xr, yr)
            scan = zip(X.flatten(), Y.flatten())

            for x, y in scan:
                #print(x,y)
                postfix = f'_cpt_{x}_cpqm_{y}_fixed'  # NOTE not sure if fixed is what should be here??
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
        else:
            outputs.append(get_merged_output(
                'SS_analysis',
                year,
                './outputs/',
                samples, mapping,
                lumi=lumi,
                postfix='_cpt_0_cpqm_0_fixed',
                select_datasets = ['data', 'MCall'],
                variations = ['central', 'fake', 'base', 'jes', 'jer'],
                select_histograms = ['bit_score_incl', 'bit_score_pp', 'bit_score_mm'],
                ))

        output = accumulate(outputs)
        util.save(output, f'./outputs/{year}_fixed_merged.coffea')  # this will just always be the latest one
