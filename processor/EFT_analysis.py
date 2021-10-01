import yahist
from Tools.config_helpers import *
from Tools.EFT_tools import make_scan

if __name__ == '__main__':

    from klepto.archives import dir_archive
    from Tools.samples import get_babies
    from processor.default_accumulators import *
    from processor.SS_analysis import *
    from Tools.samples import get_babies

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--keep', action='store_true', default=None, help="Keep/use existing results??")
    argParser.add_argument('--year', action='store', default='2018', help="Which year to run on?")
    argParser.add_argument('--training', action='store', default='v21', help="Which training to use?")
    argParser.add_argument('--scan', action='store', default='ctW', choices=['ctW','ctp',], help="Which training to use?")
    args = argParser.parse_args()

    overwrite   = not args.keep

    year        = int(args.year[0:4])
    era         = args.year[4:7]
    save        = True

    # load the config and the cache
    cfg = loadConfig()

    cacheName = 'EFT_%s_scan_%s%s'%(args.scan, year, era)
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)

    if args.scan == 'ctW':
        points = make_scan(operator='ctW', C_min=-2.5, C_max=2.5, step=0.25)
    elif args.scan == 'ctp':
        points = get_scan('ctp', C_min=-30, C_max=30, step=5)

    in_path = '/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.5.2_dilep/'

    fileset_all = get_babies(in_path, year='UL%s%s'%(year,era))

    fileset = {
        'topW_NLO': fileset_all['topW_NLO'],
        'topW_full_EFT': fileset_all['topW_EFT']
    }
    
    add_processes_to_output(fileset, desired_output)
    
    desired_output.update({
            "ST": hist.Hist("Counts", dataset_axis, ht_axis),
            "HT": hist.Hist("Counts", dataset_axis, ht_axis),
            "LT": hist.Hist("Counts", dataset_axis, ht_axis),
            "lead_lep_SR_pp": hist.Hist("Counts", dataset_axis, pt_axis),
            "lead_lep_SR_mm": hist.Hist("Counts", dataset_axis, pt_axis),
            "LT_SR_pp": hist.Hist("Counts", dataset_axis, ht_axis),
            "LT_SR_mm": hist.Hist("Counts", dataset_axis, ht_axis),
            "node": hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "node0_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
            "node2_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
            "node3_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
            "node4_score_incl": hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score": hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score": hist.Hist("Counts", dataset_axis, score_axis),
            "node2_score": hist.Hist("Counts", dataset_axis, score_axis),
            "node3_score": hist.Hist("Counts", dataset_axis, score_axis),
            "node4_score": hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score_pp": hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score_mm": hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score_transform_pp": hist.Hist("Counts", dataset_axis, score_axis),
            "node0_score_transform_mm": hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_pp": hist.Hist("Counts", dataset_axis, score_axis),
            "node1_score_mm": hist.Hist("Counts", dataset_axis, score_axis),
            "norm": hist.Hist("Counts", dataset_axis, one_axis),
    })
    
    histograms = sorted(list(desired_output.keys()))
    
    exe_args = {
        'workers': 16,
        'function_args': {'flatten': False},
        "schema": NanoAODSchema,
    }
    exe = processor.futures_executor
    
    ## EFT reweighting
    f_in = fileset['topW_full_EFT'][0]
    
    from Tools.reweighting import get_coordinates_and_ref, get_coordinates
    coordinates, ref_coordinates = get_coordinates_and_ref(f_in)
    
    from Tools.awkwardHyperPoly import *
    hp = HyperPoly(2)
    hp.initialize( coordinates, ref_coordinates )
    
    if not overwrite:
        cache.load()
    
    variations = [
        {'name': 'central',     'ext': '',                  'weight': None,   'pt_var': 'pt_nom'},
    ]
        
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and cache.get('simple_output'):
        output = cache.get('simple_output')
    
    else:
        print ("I'm running now")
        
        output = processor.run_uproot_job(
            fileset,
            "Events",
            SS_analysis(year=year,
                        variations=variations,
                        accumulator=desired_output,
                        evaluate=True,
                        training=args.training,
                        dump=False,
                        era='',
                        hyperpoly=hp,
                        points=points),
            exe,
            exe_args,
            chunksize=250000,
        )
        
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()
