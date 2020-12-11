
import time

from metis.Sample import DirectorySample, DBSSample
from metis.CondorTask import CondorTask
from metis.StatsParser import StatsParser
from metis.Utils import do_cmd

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

data_path = os.path.expandvars('$TWHOME/data/')
with open(data_path+'samples.yaml') as f:
    samples = yaml.load(f, Loader=Loader)

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


#metisSamples = []
#for sample in samples.keys():   

#outDir = os.path.join(version, tag)
outDir = os.path.join(os.path.expandvars(cfg['meta']['localSkim']), tag)

print ("Output will be here: %s"%outDir)

maker_tasks = []
merge_tasks = []

#raise NotImplementedError

#samples = {'/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/': samples['/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/']}

#if True:

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

    print ("Sample: %s"%s)
    print ("The sample is %s, corresponding to year %s. %s simulation is used."%('Data' if isData else 'MC', year, 'Fast' if isFastSim else 'Full'  ) )
    if isData:
        print ("The era is: %s"%era)

    #lumiWeightString = 1000*samples[s]['xsec']/samples[s]['sumWeight'] if not isData else 1
    lumiWeightString = 1 if (isData or samples[s]['name'].count('TChiWH')) else 1000*samples[s]['xsec']/samples[s]['sumWeight']

    #tag = str(cfg['meta']['version']).replace('.','p')
    
    maker_task = CondorTask(
        sample = sample,
            #'/hadoop/cms/store/user/dspitzba/nanoAOD/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/',
        # open_dataset = True, flush = True,
        executable = "executable.sh",
        arguments = " ".join([ str(x) for x in [tag, lumiWeightString, 1 if isData else 0, year, era, 1 if isFastSim else 0, args.skim, args.user ]] ),
        #tarfile = "merge_scripts.tar.gz",
        files_per_output = 1 if (isData or samples[s]['name'].count('TChiWH') or samples[s]['name'].count('ZJets') or samples[s]['name'].count('DYJets')) else 3, # should also not use 3 files when using dilepton skim alltogether
        output_dir = os.path.join(outDir, samples[s]['name']),
        output_name = "nanoSkim.root",
        output_is_tree = True,
        # check_expectedevents = True,
        tag = tag,
        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        cmssw_version = "CMSSW_10_2_9",
        scram_arch = "slc6_amd64_gcc700",
        # recopy_inputs = True,
        # no_load_from_backup = True,
        min_completion_fraction = 0.99,
    )
    
    maker_tasks.append(maker_task)



##if False:
#    merge_task = CondorTask(
#        sample = DirectorySample(
#            dataset="merge_"+samples[s]['name'],
#            location=maker_task.get_outputdir(),
#        ),
#        # open_dataset = True, flush = True,
#        executable = "merge_executable.sh",
#        arguments = "%s %s"%(tag, lumiWeightString),
#        #tarfile = "merge_scripts.tar.gz",
#        files_per_output = 10,
#        output_dir = maker_task.get_outputdir() + "/merged",
#        output_name = "nanoSkim.root",
#        output_is_tree = True,
#        # check_expectedevents = True,
#        tag = tag,
#        # condor_submit_params = {"sites":"T2_US_UCSD"},
#        # cmssw_version = "CMSSW_9_2_8",
#        # scram_arch = "slc6_amd64_gcc530",
#        condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
#        cmssw_version = "CMSSW_10_2_9",
#        scram_arch = "slc6_amd64_gcc700",
#        # recopy_inputs = True,
#        # no_load_from_backup = True,
#        min_completion_fraction = 0.99,
#    )
#
#    merge_tasks.append(merge_task)

if not args.dryRun:
    for i in range(100):
        total_summary = {}
    
        #for maker_task, merge_task in zip(maker_tasks,merge_tasks):
        for maker_task in maker_tasks:
            maker_task.process()
    
            frac = maker_task.complete(return_fraction=True)
            #if frac >= maker_task.min_completion_fraction:
            ## if maker_task.complete():
            #    do_cmd("mkdir -p {}/merged".format(maker_task.get_outputdir()))
            #    do_cmd("mkdir -p {}/skimmed".format(maker_task.get_outputdir()))
            #    merge_task.reset_io_mapping()
            #    merge_task.update_mapping()
            #    merge_task.process()
    
            total_summary[maker_task.get_sample().get_datasetname()] = maker_task.get_task_summary()
            #total_summary[merge_task.get_sample().get_datasetname()] = merge_task.get_task_summary()
 
        print (frac)
   
        # parse the total summary and write out the dashboard
        StatsParser(data=total_summary, webdir="~/public_html/dump/metis_tW_scattering/").do()
    
        # 15 min power nap
        time.sleep(15.*60)



