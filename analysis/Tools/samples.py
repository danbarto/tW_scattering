#!/usr/bin/env python3

'''
my 10th attempt to get all the boilerplate sample management under control
- no external dependencies except uproot
- transparent to site this runs on

'''
import os
import uproot
import subprocess
import glob
import yaml
import pandas as pd
from yaml import Loader, Dumper
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

dasgoclient = '/cvmfs/cms.cern.ch/common/dasgoclient'
redirectors = {
    'ucsd': 'root://redirector.t2.ucsd.edu:1095/',
    'ucsd_xcache': 'root://xcache-redirector.t2.ucsd.edu:2042/',
    'fnal': 'root://cmsxrootd.fnal.gov/',
    'fnal_eos': 'root://cmseos.fnal.gov/',
    'global': 'root://cms-xrd-global.cern.ch/',
}

def das_wrapper(DASname, query='file'):
    sampleName = DASname.rstrip('/')
    dbsOut = subprocess.check_output(
        [dasgoclient, f"-query={query} dataset={sampleName}"],
        stderr=subprocess.PIPE,
        text=True,
    )
    dbsOut = dbsOut.split('\n')
    if len(dbsOut[-1]) < 1: dbsOut = dbsOut[:-1]
    return dbsOut

def xrdfsls(path, redirector):
    cmd = f"xrdfs {redirector} ls {path}"
    out = os.popen(cmd).readlines()
    out = [ l.replace('\n','') for l in out ]
    return out


class Sample:
    # samples_UL16APV.yaml
    def __init__(self, das_name=None, xsec=0, reweight=None, redirector=None):
        self.name           = das_name
        self.simple_name    = self.get_simple_name()
        self.redirector     = redirector if redirector != None else redirectors['fnal']
        #self.reweight       = reweight
        self.xsec           = float(xsec)
        self.is_data        = self.simple_name.count('Run20') > 0
        if reweight == '1':
            self.reweight = 1
        elif reweight == None or reweight == 1 or isinstance(reweight, tuple):
            pass
        else:
            tmp = reweight.split(',')
            self.reweight = (tmp[0], int(tmp[1]))

    def get_simple_name(self):
        split = self.name.split('/')
        if split[-1].count('AOD'):
            return '_'.join(self.name.split('/')[1:3])
        else:
            return '_'.join(self.name.split('/')[-3:-1])

    def get_files(self):
        # FIXME needs workaround for private samples not in DBS
        # NOTE for the future: run crab and publish private MC?
        if not hasattr(self, 'files'):
            if self.name.count('ceph'):
                path = self.name.split('/cms')[-1]
                self.files = [ f for f in xrdfsls(path, redirectors['ucsd']) if f.endswith('.root') ]
                self.redirector = redirectors['ucsd']
            else:
                self.files = das_wrapper(self.name, query='file')
        return self.files

    def get_absolute_files(self):
        if not hasattr(self, 'abs_files'):
            self.abs_files = [self.redirector + f for f in self.get_files()]
        return self.abs_files

    def add_skim(self, path, local=False, redirector=None):
        # could make the assumption that the subdir is always using simple name?
        self.skim_path = path
        if local:
            self.skim_files = glob.glob(f"{path}/{self.simple_name}/*.root")
        else:
            redirector = redirector if redirector != None else self.redirector
            path = path.split('/cms')[-1]  # strip any local fs and make path relative
            path = f"{path}/{self.simple_name}/"  # NOTE think about this?
            res = xrdfsls(path, redirector)
            merged_exists = False
            for f in res:
                if f.count('/merged'):
                    new_path = f
                    merged_exists = True
                    break
            if merged_exists:
                self.skim_files = [f"{redirector}/{f}" for f in xrdfsls(new_path, redirector)]
            else:
                self.skim_files = [f"{redirector}/{f}" for f in xrdfsls(path, redirector)]

    def get_meta(self):
        # NOTE this is fairly slow, at least for a large number of files
        self.nano_nEvents = 0
        self.nano_sumWeight = 0
        for f_in in self.get_absolute_files():
            with uproot.open(f_in) as f:
                if not self.is_data:
                    self.nano_nEvents += sum(f['Runs']['genEventCount'].array())
                    self.nano_sumWeight += sum(f['Runs']['genEventSumw'].array())
                else:
                    self.nano_nEvents += len(f['Events']['event'].array())
                    self.nano_sumWeight += len(f['Events']['event'].array())

                # NOTE we don't care about all the other information right now.
                # Can be added at a later stage

    def get_meta_skim(self, verbose=False):
        first = True
        self.sumWeight = 0
        self.nEvents = 0

        for f_in in self.skim_files:
            try:
                with uproot.open(f_in) as f:
                    if not self.is_data:
                        # NOTE: these numbers are pre-skim
                        if verbose: print (f_in, float(f['genEventSumw'].counts()[0]), float(f['genEventCount'].counts()[0]))
                        self.sumWeight += float(f['genEventSumw'].counts()[0])
                        self.nEvents += float(f['genEventCount'].counts()[0])
                        if first:
                            first = False
                            self.LHEPdfWeight = f['LHEPdfSumw'].counts()
                            self.LHEScaleWeight = f['LHEScaleSumw'].counts()
                            if 'LHEReweightingSumw' in f:
                                self.LHEReweightingWeight = f['LHEReweightingSumw'].counts()
                        else:
                            self.LHEPdfWeight += f['LHEPdfSumw'].counts()
                            self.LHEScaleWeight += f['LHEScaleSumw'].counts()
                            if 'LHEReweightingSumw' in f:
                                self.LHEReweightingWeight += f['LHEReweightingSumw'].counts()
                    else:
                        # NOTE: these numbers are post-skim because we don't keep
                        # pre-skim histograms for data (irrelevant)
                        self.nEvents += len(f['Events']['event'].array())
                        self.sumWeight += len(f['Events']['event'].array())
            except:
                print ("Skipping faulty file:", f_in)
                self.skim_files.remove(f_in)  # Should it still through an error?
                #raise

        # convert into list so that we can store meta data in a human readable yaml file
        if hasattr(self, 'LHEPdfWeight'):
            self.LHEPdfWeight = [float(x) for x in self.LHEPdfWeight]
        if hasattr(self, 'LHEScaleWeight'):
            self.LHEScaleWeight = [float(x) for x in self.LHEScaleWeight]
        if hasattr(self, 'LHEReweightingWeight'):
            self.LHEReweightingWeight = [float(x) for x in self.LHEReweightingWeight]
        else:
            self.LHEReweightingWeight = [1,1]


    def skim_completion(self):
        if hasattr(self, "nEvents") and hasattr(self, "nEvents_nano"):
            return self.nEvents / self.nEvents_nano
        else:
            # NOTE could also be smarter
            return -1

    def skim_efficiency(self, skim='(1)'):
        # FIXME this is the one feature that's still missing
        pass

    def run_xsec_analyzer(self, verbose=False):
        # FIXME this will need some more work

        cfgFile     = 'xsecCfg.py'
        identifier  = "After filter: final cross section ="
        cfg = """
import FWCore.ParameterSet.Config as cms
process = cms.Process("GenXSec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring('{FILEPATH}') )
process.dummy = cms.EDAnalyzer("GenXSecAnalyzer", genFilterInfoTag = cms.InputTag("genFilterEfficiencyProducer") )
process.p = cms.Path(process.dummy)"""

        mini = das_wrapper(self.name, query='parent')[0]
        if verbose: print(f"- Parent: {mini}")
        mini_file = das_wrapper(mini, query='file')[0]
        if verbose: print(f"- File: {mini_file}")

        replaceString = {'FILEPATH': mini_file}
        cmsCfgString = cfg.format( **replaceString )
        with open(cfgFile, 'w') as f:
            f.write(cmsCfgString)

        cmd = "/cvmfs/cms.cern.ch/cmsset_default.sh; cd /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_10_6_19/src/; scramv1 runtime -sh; cd -; cmsRun xsecCfg.py"
        if verbose: print("Running command:", cmd)
        ret = subprocess.run(cmd, capture_output=True, shell=True, text=True)

        print(ret.stderr)
        output = ret.stderr.split('\n')
        for line in output:
            #print line
            if line.startswith(identifier): result = line

        xsec, unc = float(result.split(identifier)[1].split('+-')[0]), float(result.split(identifier)[1].split('+-')[1].replace('pb',''))
        return xsec, unc

    def get_nano_tree(self, file_index=0, name="Events"):
        assert file_index<len(self.get_files())-1, "Index out of range"
        return uproot.open(self.get_absolute_files()[file_index])[name]

    def get_skim_file(self, file_index=0):
        assert file_index<len(self.skim_files)-1, "Index out of range"
        return uproot.open(self.skim_files[file_index])

    def get_skim_tree(self, file_index=0, name="Events"):
        return self.get_skim_file(file_index)[name]

    def get_NanoEvents_from_nano(self, file_index=0, n_max=5000):
        return NanoEventsFactory.from_root(
            self.get_absolute_files()[file_index],
            schemaclass = NanoAODSchema,
        ).events()

    def get_NanoEvents_from_skim(self, file_index=0, n_max=5000):
        return NanoEventsFactory.from_root(
            self.skim_files[file_index],
            schemaclass = NanoAODSchema,
        ).events()

    def copy_to(self):
        pass

class Samples:
    def __init__(self, f_in, db={}):
        self.db = db
        self.f_in = f_in

        self.make_df(db)

    @classmethod
    def from_txt(cls, f_in, skim=None, redirector=redirectors['ucsd']):
        # classmethod?
        db = {}
        with open( f_in ) as f:
            samples = [ tuple(line.split()) for line in f.readlines() ]
        for s in samples:
            print(f"Working on sample: {s[0]}")
            db[s[0]] = Sample(s[0], xsec=s[1], reweight=s[2])
            if skim:
                db[s[0]].add_skim(
                    skim,
                    redirector = redirector
                )
                db[s[0]].get_meta_skim()

        return cls(f_in, db)

    @classmethod
    def from_yaml(cls, f_in):
        with open(f_in, 'r') as f:
            db_dict = yaml.load(f, Loader=Loader)

        db = {}
        for k in db_dict:
            db[k] = Sample(db_dict[k]['name'], db_dict[k]['xsec'], db_dict[k]['reweight'])
            #db[k].__dict__.update(vars(db[k]))
            db[k].__dict__.update(db_dict[k])

        return cls(f_in, db)

    def to_yaml(self, f_out):
        db_dict = {}
        for k in self.db:
            db_dict[k] = vars(self.db[k])
        with open(f_out, 'w') as f:
            yaml.dump(db_dict, f, Dumper=Dumper)

    def make_df(self, db):
        tmp = []
        for k in db:
            tmp.append(vars(db[k]))
        self.df = pd.DataFrame(tmp)

    def mapping(self, mapping):
        new_dict = {}
        for year in mapping:
            for group in mapping[year]:
                for sample in mapping[year][group]:
                    new_dict[sample] = {'year': year, 'group': group}
        for s in self.db:
            if s in new_dict:
                self.db[s].group = new_dict[s]['group']
                self.db[s].year = new_dict[s]['year']
        self.make_df(self.db)

    def mapping_from_file(self, f_in):
        with open(f_in, 'r') as f:
            mapping = yaml.load(f, Loader=Loader)
        self.mapping(mapping)

    def get_sample_list(self, year='UL16', group='topW'):
        return self.df[((self.df.year == year) & (self.df.group==group))]['name'].to_list()

    def get_fileset(self, year='UL16', group='topW'):
        tmp = self.df[((self.df.year == year)&(self.df.group == group))]
        return dict(zip(tmp.name, tmp.skim_files))

    def get_reweight(self, year='UL16', group='topW'):
        # NOTE could this be done at another stage?
        for k in self.db:
            if isinstance(self.db[k].reweight, tuple):
                try:
                    self.db[k].reweight_weight = getattr(self.db[k], self.db[k].reweight[0])[self.db[k].reweight[1]]/self.db[k].sumWeight
                except:
                    pass
                    #print (k)
            else:
                self.db[k].reweight_weight = self.db[k].reweight
        self.make_df(self.db)
        return dict(zip(self.df.name, self.df.reweight))

    def get_sample_weight(self, lumi=1):
        # could maybe use a filter?
        if not hasattr(self.db, 'reweight_weight'):
            self.get_reweight()

        factor = (self.df.reweight_weight*lumi*self.df.xsec*1000/self.df.sumWeight) * (self.df.xsec>0) + 1*(self.df.xsec<0)
        return dict(zip(self.df.name, factor))


if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--tests', action='store_true', default=False, help="Run simple and fast tests")
    argParser.add_argument('--run', action='store_true', default=False, help="Run the whole damn thing")
    argParser.add_argument('--load', action='store_true', default=False, help="Run the whole damn thing")
    argParser.add_argument('--input', action='store', default='data/samples.txt', help="Input file")
    argParser.add_argument('--output', action='store', default='data/samples.yaml', help="Output yaml file")
    argParser.add_argument('--skim', action='store', default=None, help="Skim path")
    args = argParser.parse_args()

    if args.run:
        samples = Samples.from_txt(args.input, skim=args.skim)
        samples.mapping_from_file('data/nano_mapping.yaml')
        samples.to_yaml(args.output)

    if args.load:
        samples = Samples.from_yaml(args.input)


    if args.tests:

        samples = Samples.from_txt('data/samples_test.txt', skim='/ceph/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.7.1_SS/')
        samples.mapping_from_file('data/nano_mapping.yaml')
        samples.to_yaml('data/test_samples.yaml')

        samples_loaded = samples.from_yaml('data/test_samples.yaml')

        example = '/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM'
        print(f"Working on central sample {example}")
        s1 = Sample(
            example,
            redirector = redirectors['fnal'],
        )
        print("- Getting files")
        s1.get_files()
        print("- Adding skim")
        s1.add_skim(
            '/ceph/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.7.1_SS/',
            redirector=redirectors['ucsd'],
        )
        print("- Getting meta data")
        s1.get_meta_skim(verbose=False)
        print("- Done!\n")


        example = '/ceph/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL18_NanoAODv9_NANO_v12/'
        print(f"Working on private sample {example}")
        s2 = Sample(
            example,
            redirector = redirectors['fnal'],
        )
        print("- Getting files")
        s2.get_files()
        print("- Adding skim")
        s2.add_skim(
            '/ceph/cms/store/user/dspitzba/nanoAOD/ttw_samples/topW_v0.7.1_SS/',
            redirector=redirectors['ucsd'],
        )
        print("- Getting meta data")
        s2.get_meta_skim(verbose=False)
        print("- Done!")
