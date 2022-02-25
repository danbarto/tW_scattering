
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

def get_latest_output(cache_name, cfg):
    from coffea import util
    cache_dir = os.path.expandvars(cfg['caches']['base'])
    all_caches = glob.glob(cache_dir+'/*.coffea')
    filtered = [f for f in all_caches if f.count(cache_name)]
    filtered.sort(reverse=True)
    try:
        latest = filtered[0]
    except:
        print ("Couldn't find a suitable cache! Rerunning.")
        return None
    print ("Found the following cache: %s"%latest)
    return util.load(filtered[0])
#    return filtered[0]
