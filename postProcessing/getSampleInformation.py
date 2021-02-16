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
import awkward as ak
import numpy as np
import glob

from Tools.config_helpers import *

from metis.Sample import DirectorySample, DBSSample

data_path = os.path.expandvars('$TWHOME/data/')

def getSplitFactor(sample, target=1e6):

    if sample.get_nevents() == 0:
        average_events = 10e3
        fin = sample.get_files()[0].name
    else:
        average_events = sample.get_nevents()/len(sample.get_files())
        fin = redirector_fnal + sample.get_files()[0].name

    print (fin)
    tree = uproot.open(fin)["Events"]
    print (len(tree))
    met = tree['MET_pt'].array()
    muon_pt = tree['Muon_pt'].array()
    nMuon = ak.num(muon_pt[( (muon_pt>10) & (np.abs(tree['Muon_eta'].array())<2.4) )])
    electron_pt = tree['Electron_pt'].array()
    nElectron = ak.num(electron_pt[( (electron_pt>10) & (np.abs(tree['Electron_eta'].array())<2.4) )])
    lepton_filter = (nMuon+nElectron)>1
    jet_pt = tree['Jet_pt'].array()
    jet_filter = ( ak.num(jet_pt[ ( (jet_pt>25) & (np.abs(tree['Jet_eta'].array())<2.4) ) ]) > 1 )
    nEvents_all  = len(met)
    nEvents_pass = len(met[lepton_filter & jet_filter])

    filter_eff = nEvents_pass/nEvents_all
    print ("Average number of events in file:", average_events)
    print ("Filter efficiency:", filter_eff)

    return min(max(1, int(round(target/(average_events*filter_eff),0))), len(sample.get_files()))

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

def getMetaUproot(file, local=True):
    try:
        f = uproot.open(file)
        r = f['Runs']
    except:
        return 0,0,0

    if local:
        res = r['genEventCount'].array()[0], r['genEventSumw'].array()[0], r['genEventSumw2'].array()[0]
    else:
        try:
            res = r['genEventCount_'].array()[0], r['genEventSumw_'].array()[0], r['genEventSumw2_'].array()[0]
        except:
            res = r['genEventCount'].array()[0], r['genEventSumw'].array()[0], r['genEventSumw2'].array()[0]
    return res
    

def dasWrapper(DASname, query='file'):
    sampleName = DASname.rstrip('/')

    dbs='dasgoclient -query="%s dataset=%s"'%(query, sampleName)
    dbsOut = os.popen(dbs).readlines()
    dbsOut = [ l.replace('\n','') for l in dbsOut ]
    return dbsOut

def getSampleNorm(files, local=True, redirector=redirector_ucsd):
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

        #print ("Will get info now.")

        # First, get the name
        name = getName(sample[0])
        print ("Started with: %s"%name)

        year, era, isData, isFastSim = getYearFromDAS(sample[0])

        # local/private sample?
        local = (sample[0].count('hadoop') + sample[0].count('home'))
        #print ("Is local?", local)
        #print (sample[0])

        if local:
            sample_dict['path'] = sample[0]
            metis_sample = DirectorySample(dataset=name, location = sample[0])
            
        else:
            sample_dict['path'] = None
            metis_sample = DBSSample(dataset = sample[0] )

        allFiles = [ f.name for f in metis_sample.get_files() ]

        split_factor = getSplitFactor(metis_sample, target=1e6)
        # 
        #print (allFiles)
        sample_dict['files'] = len(allFiles)

        if not isData:
            nEvents, sumw, sumw2 = getSampleNorm(allFiles, local=local, redirector=redirector_fnal)
        else:
            nEvents, sumw, sumw2 = metis_sample.get_nevents(),0,0

        #print (nEvents, sumw, sumw2)
        sample_dict.update({'sumWeight': float(sumw), 'nEvents': int(nEvents), 'xsec': float(sample[1]), 'name':name, 'split':split_factor})

        print ("Done with: %s"%name)
        
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
    print ("Will have to work in %s samples."%len(sampleList_missing))

    counter = 0
    sample_tmp = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for sample, result in zip(sampleList_missing, executor.map(getDict, sampleList_missing)):
            try:
                samples.update({str(sample[0]): result})
            except:
                print ("Failed, will try again next time...")
            #sample_tmp += [{str(sample[0]): result}]
            #counter += 1
            #print (sample[0])
            #print (result)
            #print ("Done with %s samples."%counter)

    print ("Done with the heavy lifting. Dumping results to yaml file now.")

    with open(data_path+'samples.yaml', 'w') as f:
        yaml.dump(samples, f, Dumper=Dumper)

    print ("Done.")

    return samples


if __name__ == '__main__':
    samples = main()



