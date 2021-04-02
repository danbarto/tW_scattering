
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

import glob

redirector_ucsd = 'root://xcache-redirector.t2.ucsd.edu:2040/'
redirector_fnal = 'root://cmsxrootd.fnal.gov/'

data_path = os.path.expandvars('$TWHOME/data/')

def load_yaml(f_in=data_path+'nano_mapping.yaml'):
    with open(f_in) as f:
        res = load(f, Loader=Loader)
    return res

def loadConfig():
    with open(data_path+'config.yaml') as f:
        config = load(f, Loader=Loader)
    return config

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

