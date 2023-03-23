
#import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#from yaml import Loader, Dumper

import os
import shutil
import math
import copy
import re

import glob

redirector_ucsd = 'root://xcache-redirector.t2.ucsd.edu:2042/' # this one is exclusively for NanoAOD. 165 TB cap.
redirector_ucsd_mini = 'root://xcache-redirector.t2.ucsd.edu:2040/' # this cache can keep anything, also Nano.
redirector_fnal = 'root://cmsxrootd.fnal.gov/'

data_pattern = re.compile('MuonEG|DoubleMuon|DoubleEG|EGamma|SingleMuon|SingleElectron')

data_path = os.path.expandvars('data/')

lumi = {
    '2016': 16.8,
    '2016APV': 19.5,
    '2017': 41.48,
    '2018': 59.83,
}

def get_samples(f_in='samples.yaml'):
    with open(data_path+f_in) as f:
        return load(f, Loader=Loader)

def load_yaml(f_in=data_path+'nano_mapping.yaml'):
    with open(f_in) as f:
        res = load(f, Loader=Loader)
    return res

def dump_yaml(data, f_out):
    with open(f_out, 'w') as f:
        dump(data, f, Dumper=Dumper, default_flow_style=False)
    return True

def loadConfig():
    return load_yaml(data_path+'config.yaml')

def get_cache(cache_name):
    from klepto.archives import dir_archive
    cfg = loadConfig()
    cache_dir = os.path.expandvars(cfg['caches']['base'])

    cache = dir_archive(cache_dir+cache_name, serialized=True)
    cache.load()
    return cache.get('simple_output')

def dumpConfig(cfg):
    with open(data_path+'config.yaml', 'w') as f:
        dump(cfg, f, Dumper=Dumper, default_flow_style=False)
    return True

def getName( DAS ):
    split = DAS.split('/')
    if split[-1].count('AOD'):
        return '_'.join(DAS.split('/')[1:3])
    else:
        return '_'.join(DAS.split('/')[-3:-1])
        #return'dummy'

def finalizePlotDir( path ):
    path = os.path.expandvars(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copy( os.path.expandvars( 'Tools/php/index.php' ), path )

def make_small(fileset, small, n_max=1):
    if small:
        for proc in fileset:
            fileset[proc] = fileset[proc][:n_max]
    return fileset

def load_wrapper(f_in, select_histograms):
    from coffea.processor.accumulator import dict_accumulator
    from coffea import util
    tmp = util.load(f_in)
    if select_histograms:
        res = dict_accumulator()
        for hist in select_histograms:
            res[hist] = tmp[hist]
    else:
        res = tmp
    return res


def get_latest_output(cache_name, cache_dir, date=None, max_time='999999', select_histograms=False):
    import concurrent.futures
    from functools import partial
    import re

    all_caches = glob.glob(cache_dir+'/*.coffea')
    #filtered = [f for f in all_caches if f.count(cache_name)]
    filtered = [f for f in all_caches if re.search('(.*)'+cache_name+'_(\d{8})_(\d{6}).coffea', f)]
    if date:
        filtered = [f for f in filtered if f.count(date)]
    if not cache_name.count('APV'):
        # manually filter out everything with APV if it's not APV
        filtered = [f for f in filtered if not f.count('APV')]
    # FIXME need to ensure that only <cache_name>_<datetime> is allowed
    filtered = [f for f in filtered if int(f.replace('.coffea','').split('_')[-1])<int(max_time)]
    filtered.sort(reverse=True)
    print (filtered)
    try:
        latest = filtered[0]
    except:
        print ("Couldn't find a suitable cache!")
        return None
    print ("Found the following cache: %s"%latest)

    func = partial(load_wrapper, select_histograms=select_histograms)
    # this is a bad hack
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for f_in, hist in zip(filtered[:1], executor.map(func, filtered[:1])):
            res = hist
    return res
#    return filtered[0]

def get_merged_output(
        name,
        year,
        cache_dir,
        samples,
        mapping,
        lumi=1,
        postfix=None,
        quiet=False,
        select_datasets=None,
        date=None,
        max_time='999999',
        select_histograms=False,
        variations = ['central'],
):
    '''
    name: e.g. SS_analysis
    year: string like 2016APV
    postfix: string, e.g. _DY
    '''
    from coffea.processor import accumulate
    from coffea import hist
    from analysis.Tools.helpers import scale_and_merge
    ul = "UL"+year[2:] if year != '2022' else "Run3_%s"%(year[2:])

    reweight = samples.get_reweight()  # this gets the reweighting weight name and index for the processor
    weights = samples.get_sample_weight(lumi=lumi)  # this gets renorm
    renorm   = {}

    #datasets = data + order
    datasets = list(mapping[ul].keys())

    if isinstance(select_datasets, list):
        datasets = select_datasets

    outputs = []


    #for sample in datasets:
    for variation in variations:
        for sample in ['MCall', 'data']:
            if not quiet: print ("Loading output for sample:", sample)
            if variation != '':
                cache_name = '_'.join([name, sample, variation, year])
            else:
                cache_name = '_'.join([name, sample, year])
            #f'{name}_{sample}_{year}'
            if postfix:
                cache_name += postfix
            print (cache_name)
            outputs.append(get_latest_output(cache_name, cache_dir, date=date, max_time=max_time, select_histograms=select_histograms))

    output = accumulate(outputs)

    #res = scale_and_merge(output['N_jet'], renorm, mapping[ul])

    output_scaled = {}
    for key in output.keys():
        if isinstance(output[key], hist.Hist):
            try:
                print ("Merging histogram", key)
                output_scaled[key] = scale_and_merge(output[key], weights, mapping[ul])
            except:
                print ("Scale and merge failed for:",key)
                print ("At least I tried.")

    del outputs, output
    return output_scaled
