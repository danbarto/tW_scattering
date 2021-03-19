
import time

import numpy as np
import re

from metis.Sample import DirectorySample, DBSSample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser
from metis.Utils import do_cmd

from Tools.helpers import data_path, get_samples
from Tools.config_helpers import *

# load samples
import yaml
from yaml import Loader, Dumper

import os
from github import Github

def getYearFromDAS(DASname):
    isData = True if DASname.count('Run20') else False
    isFastSim = False if not DASname.count('Fast') else True
    era = DASname[DASname.find("Run")+len('Run2000'):DASname.find("Run")+len('Run2000A')]
    if DASname.count('Autumn18') or DASname.count('Run2018'):
        return 2018, era, isData, isFastSim
    elif DASname.count('Fall17') or DASname.count('Run2017'):
        return 2017, era, isData, isFastSim
    elif DASname.count('Summer16') or DASname.count('Run2016'):
        return 2016, era, isData, isFastSim
    else:
        ### our private samples right now are all Autumn18 but have no identifier.
        return 2018, 'X', False, False

samples = get_samples()  # loads the nanoAOD samples

# load config
cfg = loadConfig()

print ("Loaded version %s from config."%cfg['meta']['version'])

import argparse

argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--tag', action='store', default=None, help="Tag on github for baby production")
argParser.add_argument('--user', action='store', help="Your github user name")
argParser.add_argument('--skim', action='store', default="leptons", choices=["leptons","MET", "dijet"], help="Which skim to use")
argParser.add_argument('--dryRun', action='store_true', default=None, help="Don't submit?")
argParser.add_argument('--small', action='store_true', default=None, help="Only submit first two samples?")
argParser.add_argument('--only', action='store', default='', help="Just select one sample")
args = argParser.parse_args()

tag = str(args.tag)

### Read github credentials
with open('github_credentials.txt', 'r') as f:
    lines = f.readlines()
    cred = lines[0].replace('\n','')

print ("Found github credentials: %s"%cred)

### We test that the tag is actually there
repo_name = '%s/NanoAOD-tools'%args.user

g = Github(cred)
repo = g.get_repo(repo_name)
tags = [ x.name for x in repo.get_tags() ]
if not tag in tags:
    print ("The specified tag %s was not found in the repository: %s"%(tag, repo_name))
    print ("Exiting. Nothing was submitted.")
    exit()
else:
    print ("Yay, located tag %s in repository %s. Will start creating tasks now."%(tag, repo_name) )

# example
sample = DirectorySample(dataset='TTWJetsToLNu_Autumn18v4', location='/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/')

outDir = os.path.join(os.path.expandvars(cfg['meta']['localSkim']), tag)

print ("Output will be here: %s"%outDir)

maker_tasks = []
merge_tasks = []

sample_list = samples.keys() if not args.small else samples.keys()[:2]

sample_list = [ x for x in samples.keys() if args.only in x ] #

print ("Will run over the following samples:")
print (sample_list)
print ()



for s in sample_list:
    if samples[s]['path'] is not None:
        sample = DirectorySample(dataset = samples[s]['name'], location = samples[s]['path'])
    else:
        sample = DBSSample(dataset = s) # should we make use of the files??

    year, era, isData, isFastSim = getYearFromDAS(s)

    print ("Now working on sample: %s"%s)
    print ("- has %s files"%len(sample.get_files()))
    print ("- is %s, corresponding to year %s. %s simulation is used."%('Data' if isData else 'MC', year, 'Fast' if isFastSim else 'Full'  ) )
    if isData:
        print ("The era is: %s"%era)
    # merge three files into one for all MC samples except ones where we expect a high efficiency of the skim
    signal_string = re.compile("TTW.*EWK")
    mergeFactor = min(4, samples[s]['split']) if not (samples[s]['name'].count('tW_scattering') or re.search(signal_string, samples[s]['name']) ) else samples[s]['split'] # not running over more than 4 files because we prefetch...
    print ("- using merge factor: %s"%mergeFactor)

    #lumiWeightString = 1000*samples[s]['xsec']/samples[s]['sumWeight'] if not isData else 1
    lumiWeightString = 1 if (isData or samples[s]['name'].count('TChiWH')) else 1000*samples[s]['xsec']/samples[s]['sumWeight']
    print ("- found sumWeight %s and x-sec %s"%(samples[s]['sumWeight'], samples[s]['xsec']) )

    maker_task = CondorTask(
        sample = sample,
        executable = "executable.sh",
        arguments = " ".join([ str(x) for x in [tag, lumiWeightString, 1 if isData else 0, year, era, 1 if isFastSim else 0, args.skim, args.user ]] ),
        files_per_output = int(mergeFactor),
        output_dir = os.path.join(outDir, samples[s]['name']),
        output_name = "nanoSkim.root",
        output_is_tree = True,
        tag = tag,
        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        cmssw_version = "CMSSW_10_2_9",
        scram_arch = "slc6_amd64_gcc700",
        min_completion_fraction = 0.99,
    )
    
    maker_tasks.append(maker_task)

if not args.dryRun:
    for i in range(100):
        total_summary = {}
    
        #for maker_task, merge_task in zip(maker_tasks,merge_tasks):
        for maker_task in maker_tasks:
            maker_task.process()
    
            frac = maker_task.complete(return_fraction=True)
            total_summary[maker_task.get_sample().get_datasetname()] = maker_task.get_task_summary()
 
        print (frac)
   
        # parse the total summary and write out the dashboard
        StatsParser(data=total_summary, webdir="~/public_html/dump/metis_tW_scattering/").do()
    
        # 15 min power nap
        time.sleep(15.*60)



