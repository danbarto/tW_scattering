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

from Tools.config_helpers import loadConfig, redirector_fnal, redirector_ucsd, getName

from metis.Sample import DirectorySample, DBSSample

data_path = os.path.expandvars('$TWHOME/data/')

def getSplitFactor(sample, target=1e6):

    if sample.get_nevents() == 0:
        average_events = 10e3
        fin = sample.get_files()[0].name
    else:
        average_events = sample.get_nevents()/len(sample.get_files())
        fin = sample.get_files()[0].name

    print (fin)
    if fin.count('ceph'): redirectors = ['/']
    else: redirectors = [redirector_ucsd, redirector_fnal]
    for red in redirectors:
        try:
            with uproot.open(f"{red}/{fin}") as f:
                tree = f["Events"]
                print (len(tree))
                if len(tree['event'].array())<1:
                    print ("Empty file")
                    return 1
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
        except OSError:
            print( "Something failed with redirector", red )

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
        print ("Couldn't open file: %s"%file)
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
    good_files = []
    for f in files:
        res = getMetaUproot(f, local=local)
        if res[0]>0:
            nEvents += res[0]
            sumw += res[1]
            sumw2 += res[2]
            good_files.append(f)
        
    return nEvents, sumw, sumw2, good_files

def getDict(sample):
        sample_dict = {}

        #print ("Will get info now.")

        # First, get the name
        name = getName(sample[0])
        print ("Started with: %s"%name)
        print (sample[0])

        year, era, isData, isFastSim = getYearFromDAS(sample[0])

        local = (sample[0].count('ceph') + sample[0].count('home') + sample[0].count('hadoop'))

        if local:
            print ("local sample")
            sample_dict['path'] = sample[0]
            metis_sample = DirectorySample(dataset=name, location = sample[0])
            
        else:
            sample_dict['path'] = None
            metis_sample = DBSSample(dataset = sample[0] )

        allFiles = [ f.name for f in metis_sample.get_files() ]

        split_factor = getSplitFactor(metis_sample, target=1e6)

        sample_dict['files'] = len(allFiles)

        nEvents, sumw, sumw2, good_files = metis_sample.get_nevents(),0,0, []  # [redirector_ucsd+f.get_name() for f in metis_sample.get_files()]

        sample_dict.update({'sumWeight': float(sumw), 'nEvents': int(nEvents), 'xsec': float(sample[1]), 'name':name, 'split':split_factor, 'files': good_files})
        try:
            sample_dict['reweight'] = int(sample[2])
        except ValueError:
            sample_dict['reweight'] = sample[2]

        print ("Done with: %s"%name)
        
        return sample_dict


def main():

    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--name',  action='store', default='samples', help='Name of the samples txt file in data/')
    argParser.add_argument('--version',  action='store', default=None, help='Skim version')
    argParser.add_argument('--dump',  action='store_true', help='Dump a latex table?')
    argParser.add_argument('--overwrite',  action='store_true', help='Overwrite')
    args = argParser.parse_args()

    config = loadConfig()

    name = args.name

    if args.version is not None:
        skim_path = '{}/{}'.format(config['meta']['localSkim'], args.version)

    # get list of samples
    sampleList = readSampleNames( data_path+'%s.txt'%name )

    if os.path.isfile(data_path+'%s.yaml'%name):
        with open(data_path+'%s.yaml'%name) as f:
            samples = yaml.load(f, Loader=Loader)
    else:
        samples = {}

    sampleList_missing = []
    # check which samples are already there
    for sample in sampleList:
        if args.overwrite:
            sampleList_missing.append(sample)
        else:
            print ("Checking if sample info for sample: %s is here already"%sample[0])
            if sample[0] in samples.keys(): continue
            sampleList_missing.append(sample)

    workers = 1
    # then, run over the missing ones
    print ("Will have to work in %s samples."%len(sampleList_missing))

    counter = 0
    sample_tmp = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for sample, result in zip(sampleList_missing, executor.map(getDict, sampleList_missing)):
            print ("Working on now", sample[0])
            try:
                samples.update({str(sample[0]): result})
                print ("Success.")
            except:
                print ("Failed, will try again next time...")
            #sample_tmp += [{str(sample[0]): result}]
            #counter += 1
            #print (sample[0])
            #print (result)
            #print ("Done with %s samples."%counter)

            print ("Done with the heavy lifting. Dumping results to yaml file now.")

            with open(data_path+'%s.yaml'%name, 'w') as f:
                print ("Dumping info into yaml file.")
                yaml.dump(samples, f, Dumper=Dumper)

    for sample in samples.keys():
        sample_name = samples[sample]['name']
        print (sample_name)
        if args.version is not None:
            skim_path_total = f"{skim_path}/{sample_name}/merged/"
            print (skim_path_total)
            samples[sample]['files'] = glob.glob(skim_path_total+"*.root")
            if samples[sample]['xsec'] > 0:  # NOTE: identifier for data / MC
                samples[sample]['sumWeight'] = 0
                first = True
                for f_in in samples[sample]['files']:
                    with uproot.open(f_in) as f:
                        samples[sample]['sumWeight'] += float(f['genEventSumw'].counts()[0])
                        if first:
                            first = False
                            samples[sample]['LHEPdfWeight'] = f['LHEPdfSumw'].counts()
                            samples[sample]['LHEScaleWeight'] = f['LHEScaleSumw'].counts()
                            if 'LHEReweightingSumw' in f:
                                samples[sample]['LHEReweightingWeight'] = f['LHEReweightingSumw'].counts()
                        else:
                            samples[sample]['LHEPdfWeight'] += f['LHEPdfSumw'].counts()
                            samples[sample]['LHEScaleWeight'] += f['LHEScaleSumw'].counts()
                            if 'LHEReweightingSumw' in f:
                                samples[sample]['LHEReweightingWeight'] += f['LHEReweightingSumw'].counts()


        with open(data_path+'%s.yaml'%name, 'w') as f:
            yaml.dump(samples, f, Dumper=Dumper)

    print ("Done.")

    if args.dump:
        import pandas as pd
        df = pd.DataFrame(samples)
        with pd.option_context("max_colwidth", 1000):
            print(df.transpose()[['xsec']].to_latex())

    return samples


if __name__ == '__main__':
    samples = main()



