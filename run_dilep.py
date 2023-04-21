#!/usr/bin/env python3
import os
import re
import datetime
import time
import awkward as ak
import glob
import pandas as pd

from coffea import processor, hist, util
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import accumulate

from analysis.Tools.samples import Samples
from analysis.SS_analysis import SS_analysis, histograms, variations, base_variations, variations_jet_all, nonprompt_variations, central_variation
from analysis.default_accumulators import add_processes_to_output, desired_output
from analysis.Tools.config_helpers import loadConfig, data_pattern, get_latest_output, load_yaml
from analysis.Tools.helpers import scale_and_merge, getCutFlowTable

if __name__ == '__main__':


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--rerun', action='store_true', default=False, help="Rerun or try using existing results??")
    argParser.add_argument('--minimal', action='store_true', default=False, help="Only run minimal set of histograms")
    argParser.add_argument('--dask', action='store_true', default=False, help="Run on a DASK cluster?")
#    argParser.add_argument('--central', action='store_true', default=False, help="Only run the central value (no systematics)")
    argParser.add_argument('--profile', action='store_true', default=False, help="Memory profiling?")
    argParser.add_argument('--iterative', action='store_true', default=False, help="Run iterative?")
    argParser.add_argument('--small', action='store_true', default=False, help="Run on a small subset?")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--evaluate', action='store_true', default=None, help="Evaluate the NN?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--workers', action='store', default=10, help="How many threads for local running?")
    argParser.add_argument('--dump', action='store_true', default=None, help="Dump a DF for NN training?")
    argParser.add_argument('--check_double_counting', action='store_true', default=None, help="Check for double counting in data?")
    argParser.add_argument('--sample', action='store', default='all', )
    argParser.add_argument('--cpt', action='store', default=0, help="Select the cpt point")
    argParser.add_argument('--cpqm', action='store', default=0, help="Select the cpqm point")
    argParser.add_argument('--buaf', action='store', default="false", help="Run on BU AF")
    argParser.add_argument('--skim_file', action='store', default="samples_v0_8_0_SS.yaml", help="Define the skim to run on")
    argParser.add_argument('--select_systematic', action='store', default="central", help="Define the skim to run on")
    argParser.add_argument('--scan', action='store_true', default=None, help="Run the entire cpt/cpqm scan")
    argParser.add_argument('--skip_bit', action='store_true', default=None, help="Skip running BIT evaluation")
    argParser.add_argument('--parametrized', action='store_true', default=None, help="Run fully parametrized, otherwise use a fixed template (hard coded to cpt=5, cpqm=-5)")
    args = argParser.parse_args()

    profile     = args.profile
    iterative   = args.iterative
    overwrite   = args.rerun
    small       = args.small
    fixed_template = not args.parametrized

    year        = int(args.year[0:4])
    ul          = "UL%s"%(args.year[2:])
    era         = args.year[4:7]
    local       = not args.dask
    save        = True

    variations =  central_variation + base_variations + variations_jet_all + nonprompt_variations
    if args.select_systematic == 'all':
        variations = variations  # lol
    elif args.select_systematic == 'central':
        variations = central_variation
    elif args.select_systematic == 'base':
        variations = base_variations
    else:
        variations = [v for v in variations if v['name'].count(args.select_systematic)]

    print("Running variations:")
    pd.set_option('display.max_rows', None)
    print(pd.DataFrame(variations))

    coordinates = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (6.0, 0.0), (3.0, 3.0), (0.0, 6.0)]
    ref_coordinates = [0,0]

    from analysis.Tools.awkwardHyperPoly import *
    hp = HyperPoly(2)
    hp.initialize(coordinates,ref_coordinates)

    # inclusive EFT weights
    eft_weights = [\
        'cpt_0p_cpqm_0p_nlo',
        'cpt_0p_cpqm_3p_nlo',
        'cpt_0p_cpqm_6p_nlo',
        'cpt_3p_cpqm_0p_nlo',
        'cpt_6p_cpqm_0p_nlo',
        'cpt_3p_cpqm_3p_nlo',
    ]

    # NOTE new way of defining points.
    if args.scan:
        x = np.arange(-7,8,1)
        y = np.arange(-7,8,1)
    else:
        x = np.array([int(args.cpt)])
        y = np.array([int(args.cpqm)])

    CPT, CPQM = np.meshgrid(x, y)

    points = []
    for cpt, cpqm in zip(CPT.flatten(), CPQM.flatten()):
        points.append({
            'name': f'eft_cpt_{cpt}_cpqm_{cpqm}',
            'point': [cpt, cpqm],
        })

    cfg = load_yaml('analysis/Tools/data/config.yaml')
    try:
        lumi = cfg['lumi'][int(str(year)+era)]
    except:
        lumi = cfg['lumi'][str(year)+era]
    print (f"Data taking era {year}{era}")
    print (f"Will be using lumi {lumi} for scaled outputs.")

    samples = Samples.from_yaml(f'analysis/Tools/data/{args.skim_file}')  # NOTE this could be era etc dependent
    mapping = load_yaml('analysis/Tools/data/nano_mapping.yaml')
    reweight = samples.get_reweight()  # this gets the reweighting weight name and index for the processor
    weights = samples.get_sample_weight(lumi=lumi)  # this gets renorm

    if args.sample == 'MCall':
        sample_list = ['DY', 'topW_lep', 'top', 'TTW', 'TTZ', 'TTH', 'XG', 'rare', 'diboson']
    elif args.sample == 'data':
        if year == 2018:
            sample_list = ['DoubleMuon', 'MuonEG', 'EGamma', 'SingleMuon']
        else:
            sample_list = ['DoubleMuon', 'MuonEG', 'DoubleEG', 'SingleMuon', 'SingleElectron']
    else:
        sample_list = [args.sample]
    #sample_list = ['top']

    if local:# and not profile:
        print("Make sure you have BIT in your local path (unfortunately it's not properly packaged)")
        print("This is automatically done on the workers")
        print("export PYTHONPATH=${PYTHONPATH}:$PWD/analysis/BIT/")
        exe = processor.FuturesExecutor(workers=int(args.workers))

    elif iterative:
        exe = processor.IterativeExecutor()

    else:
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            #cores = 1,
            #disk = '2000MB',
            #death_timeout = '60',
            #lcg = True,
            #nanny = False,
            #log_directory = '/eos/user/a/anpotreb/condor/log',
            memory = '4000MB',
            shared_temp_directory='/tmp',
            transfer_input_files=["analysis", "plots"],
            worker_extra_args=['--worker-port 10000:10070', '--nanny-port 10070:10100', '--no-dashboard'],
            job_script_prologue=[
                # https://jobqueue.dask.org/en/latest/advanced-tips-and-tricks.html#run-setup-commands-before-starting-the-worker-with-job-script-prologue
                "export PYTHONPATH=${PYTHONPATH}:$PWD/analysis/BIT/:$PWD/analysis/",
            ],
        )
        #cluster = LPCCondorCluster()
        # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=1000)
        client = Client(cluster)


        exe = processor.DaskExecutor(
            client=client,
            #savemetrics=True,
            status=True,
            #align_clusters=True,
            retries=3,
        )


    fileset = samples.get_fileset(year=ul, groups=sample_list)

    # define the cache name
    cache_name = f'./SS_analysis_{args.sample}_{args.select_systematic}_{year}{era}'
    cache_dir = './outputs/'
    if not args.scan:
        cache_name += f'_cpt_{args.cpt}_cpqm_{args.cpqm}'
    if fixed_template:
        cache_name += '_fixed'
    output = get_latest_output(cache_name, cache_dir)
    # find an old existing output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_name += f'_{timestamp}.coffea'
    if small: cache_name += '_small'

    desired_output.update(histograms)
    add_processes_to_output(fileset, desired_output)

    if output is None or args.rerun:

        if not local:
            print("Waiting for at least one worker...")
            client.wait_for_workers(1)

        print ("I'm running now")
        tic = time.time()

        runner = processor.Runner(
            exe,
            #retries=3,
            savemetrics=True,
            schema=NanoAODSchema,
            chunksize=20000,
            maxchunks=None,
        )

        output, metrics = runner(
            fileset,
            treename="Events",
            processor_instance=SS_analysis(
                year=year,
                variations=variations,
                accumulator=desired_output,
                evaluate=args.evaluate,
                bit=not args.skip_bit,
                training=args.training,
                dump=args.dump,
                era=era,
                weights=eft_weights,
                reweight=reweight,
                points=points,
                hyperpoly=hp,
                minimal=args.minimal,
                fixed_template=fixed_template,
            ),
        )
        util.save(output, cache_dir+cache_name)

        elapsed = time.time() - tic
        #print(f"Metrics: {metrics}")
        print(f"Finished in {elapsed:.1f}s")
        print(f"Total events {round(metrics['entries']/1e6,2)}M, in {metrics['chunks']} chunks")
        print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
        print(f"Output saved as {cache_dir}{cache_name}")

        #outputs.append(output)

    # basic merging / scaling
    #output = accumulate(outputs)
    output_scaled = {}
    for k in output:
        if isinstance(output[k], hist.Hist):
            try:
                output_scaled[k] = scale_and_merge(output[k], weights, mapping[ul])
            except KeyError:
                print (f"Failed to scale/merge histogram {k}")

    if 'central' in [v['name'] for v in variations ]:

        # Cutflow business below
        output_cutflow = {}
        print ("{:170}{}".format("Dataset", "Normalization"))
        for group in mapping[ul]:
            output_cutflow[group] = {}
            first = True
            for dataset in mapping[ul][group]:
                if dataset in output:
            #elif k in samples.db:
                    print ("{:170}{:.2f}".format(dataset, float(samples.db[dataset].xsec) * lumi * 1000))
                    if first:
                        for key in output[dataset]:
                            output_cutflow[group][key] = output[dataset][key]*weights[dataset]
                    else:
                        for key in output[dataset]:
                            output_cutflow[group][key] += output[dataset][key]*weights[dataset]

                    first = False

        lines= [
            'filter',
            'dilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS',
            'N_jet>3',
            'N_central>2',
            'N_btag>0',
            'N_light>0',
            'MET>30',
            'N_fwd>0',
            'min_mll'
        ]

        print (getCutFlowTable(output_cutflow,
                            processes=sample_list,
                            lines=lines,
                            significantFigures=3,
                            absolute=True,
                            #signal='topW_v3',
                            total=False,
                            ))
