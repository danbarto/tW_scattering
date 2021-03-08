"""
Helpes for the use of raw NanoAOD
"""
#from yaml import load, dump
#try:
#    from yaml import CLoader as Loader, CDumper as Dumper
#except ImportError:
#    from yaml import Loader, Dumper
from metis.Sample import DBSSample
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd

nano_mapping = {
    'top':[
        '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
        ],
    'DY': [
        '/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
        ],
    'TTW': [
        '/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
        '/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM',
        ],
    'TTZ': [
        '/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM',
        '/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM',
        ],
}


def make_fileset(datasets, samples, redirector=redirector_ucsd, small=False):
    fileset = {}
    for dataset in datasets:
        for nano_sample in nano_mapping[dataset]:
            files = [ redirector+x.name for x in DBSSample(dataset=nano_sample).get_files() ]
            if not small:
                fileset.update({nano_sample: files})
            else:
                fileset.update({nano_sample: files[:1]})

    return fileset


if __name__ == '__main__':

    samples = get_samples()
    fileset = make_fileset(['TTW', 'TTZ'], samples)
