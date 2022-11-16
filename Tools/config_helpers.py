
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

data_path = os.path.expandvars('$TWHOME/data/')

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
    shutil.copy( os.path.expandvars( '$TWHOME/Tools/php/index.php' ), path )

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


def get_latest_output(cache_name, cfg, date=None, max_time='999999', select_histograms=False):
    import concurrent.futures
    from functools import partial
    import re

    cache_dir = os.path.expandvars(cfg['caches']['base'])
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

def get_merged_output(name, year, samples=None, postfix=None, quiet=False, select_datasets=None, date=None, max_time='999999', select_histograms=False):
    '''
    name: e.g. SS_analysis
    year: string like 2016APV
    postfix: string, e.g. _DY
    '''
    from coffea.processor import accumulate
    from coffea import hist
    from plots.helpers import scale_and_merge
    ul = "UL"+year[2:] if year != '2022' else "Run3_%s"%(year[2:])
    if samples is None:
        samples = get_samples("samples_%s.yaml"%ul)
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    renorm   = {}

    if year in ['2018', '2022']:
        data = ['SingleMuon', 'DoubleMuon', 'EGamma', 'MuonEG']
    else:
        data = ['SingleMuon', 'DoubleMuon', 'DoubleEG', 'MuonEG', 'SingleElectron']
    order = ['topW_lep', 'diboson', 'TTW', 'TTH', 'TTZ', 'DY', 'top', 'XG', 'rare']

    datasets = data + order

    if isinstance(select_datasets, list):
        datasets = select_datasets

    cfg = loadConfig()

    outputs = []

    lumi_year = int(year[:4])
    lumi = cfg['lumi'][lumi_year]

    for sample in datasets:
        if not quiet: print ("Loading output for sample:", sample)
        cache_name = '_'.join([name, sample, year])
        #f'{name}_{sample}_{year}'
        if postfix:
            cache_name += postfix
        print (cache_name)
        outputs.append(get_latest_output(cache_name, cfg, date=date, max_time=max_time, select_histograms=select_histograms))

        for dataset in mapping[ul][sample]:
            if samples[dataset]['reweight'] == 1:
                renorm[dataset] = 1
            else:
                # Currently only supporting a single reweight.
                weight, index = samples[dataset]['reweight'].split(',')
                index = int(index)
                renorm[dataset] = samples[dataset]['sumWeight']/samples[dataset][weight][index]  # NOTE: needs to be divided out
            try:
                renorm[dataset] = (samples[dataset]['xsec']*1000*cfg['lumi'][lumi_year]/samples[dataset]['sumWeight'])*renorm[dataset]
            except:
                print ("Failed to renorm sample:", dataset)
                renorm[dataset] = 1
            #print (dataset, renorm[dataset])

    output = accumulate(outputs)

    #res = scale_and_merge(output['N_jet'], renorm, mapping[ul])

    output_scaled = {}
    for key in output.keys():
        if isinstance(output[key], hist.Hist):
            try:
                print ("Merging histogram", key)
                output_scaled[key] = scale_and_merge(output[key], renorm, mapping[ul])
            except:
                print ("Scale and merge failed for:",key)
                print ("At least I tried.")

    del outputs, output
    return output_scaled
