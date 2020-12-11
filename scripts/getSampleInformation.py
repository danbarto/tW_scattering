'''
takes DAS name, checks for local availability, reads norm, x-sec

e.g.
/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/NANOAODSIM
-->
TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1

'''

import yaml
from yaml import Loader, Dumper

import concurrent.futures

import os

import uproot
import glob

from Tools.config_helpers import *

data_path = os.path.expandvars('$TWHOME/data/')

def readSampleNames( sampleFile ):
    with open( sampleFile ) as f:
        samples = [ tuple(line.split()) for line in f.readlines() ]
    return samples
    
def getYearFromDAS(DASname):
    isData = True if DASname.count('Run20') else False
    isFastSim = False if not DASname.count('Fast') else True
    era = DASname[DASname.find("Run"):DASname.find("Run")+len('Run2000A')]
    if DASname.count('Autumn18') or DASname.count('Run2018'):
        return 2018, era, isData, isFastSim
    elif DASname.count('Fall17') or DASname.count('Run2017'):
        return 2017, era, isData, isFastSim
    elif DASname.count('Summer16') or DASname.count('Run2016'):
        return 2016, era, isData, isFastSim
    else:
        return -1, era, isData, isFastSim

def getMeta(file, local=True):
    '''
    for some reason, xrootd doesn't work in my environment with uproot. need to use pyroot for now...
    let's update to miniconda soon.
    '''
    import ROOT
    c = ROOT.TChain("Runs")
    c.Add(file)
    c.GetEntry(0)
    try:
        if local:
            res = c.genEventCount, c.genEventSumw, c.genEventSumw2
        else:
            try:
                res = c.genEventCount_, c.genEventSumw_, c.genEventSumw2_
            except:
                res = c.genEventCount, c.genEventSumw, c.genEventSumw2
        del c
        return res
    except:
        return 0,0,0

def getMetaUproot(file, local=True):
    try:
        f = uproot.open(file)
        r = f['Runs']
    except:
        return 0,0,0

    if local:
        res = r.array('genEventCount')[0], r.array('genEventSumw')[0], r.array('genEventSumw2')[0]
    else:
        try:
            res = r.array('genEventCount_')[0], r.array('genEventSumw_')[0], r.array('genEventSumw2_')[0]
        except:
            res = r.array('genEventCount')[0], r.array('genEventSumw')[0], r.array('genEventSumw2')[0]
    return res
    

def dasWrapper(DASname, query='file'):
    sampleName = DASname.rstrip('/')

    dbs='dasgoclient -query="%s dataset=%s"'%(query, sampleName)
    dbsOut = os.popen(dbs).readlines()
    dbsOut = [ l.replace('\n','') for l in dbsOut ]
    return dbsOut

def getSampleNorm(files, local=True, redirector='root://xrootd.t2.ucsd.edu:2040/'):
    files = [ redirector+f for f in files ] if not local else files
    nEvents, sumw, sumw2 = 0,0,0
    for f in files:
        res = getMetaUproot(f, local=local)
        nEvents += res[0]
        sumw += res[1]
        sumw2 += res[2]
    return nEvents, sumw, sumw2

def getDict(sample):
        sample_dict = {}

        print ("Will get info now.")

        # First, get the name
        name = getName(sample[0])
        print (name)

        year, era, isData, isFastSim = getYearFromDAS(sample[0])

        # local/private sample?
        local = (sample[0].count('hadoop') + sample[0].count('home'))
        print ("Is local?", local)
        print (sample[0])

        if local:
            sample_dict['path'] = sample[0]
            allFiles = glob.glob(sample[0] + '/*.root')
        else:
            sample_dict['path'] = None
            allFiles = dasWrapper(sample[0], query='file')
        # 
        print (allFiles)
        sample_dict['files'] = allFiles

        if not isData and not name.count('TChiWH'):
            nEvents, sumw, sumw2 = getSampleNorm(allFiles, local=local, redirector='root://cmsxrootd.fnal.gov/')
        else:
            nEvents, sumw, sumw2 = 0,0,0

        print (nEvents, sumw, sumw2)
        sample_dict.update({'sumWeight': float(sumw), 'nEvents': int(nEvents), 'xsec': float(sample[1]), 'name':name})
        
        return sample_dict


def main():

    config = loadConfig()

    # get list of samples
    sampleList = readSampleNames( data_path+'samples.txt' )

    if os.path.isfile(data_path+'samples.yaml'):
        with open(data_path+'samples.yaml') as f:
            samples = yaml.load(f, Loader=Loader)
    else:
        samples = {}

    sampleList_missing = []
    # check which samples are already there
    for sample in sampleList:
        print ("Checking if sample info for sample: %s is here already"%sample[0])
        if sample[0] in samples.keys(): continue
        sampleList_missing.append(sample)
    

    workers = 12
    # then, run over the missing ones
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for sample, result in zip(sampleList_missing, executor.map(getDict, sampleList_missing)):
            samples.update({str(sample[0]): result})


    with open(data_path+'samples.yaml', 'w') as f:
        yaml.dump(samples, f, Dumper=Dumper)

    return samples


if __name__ == '__main__':
    samples = main()



