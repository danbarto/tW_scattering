import os
import re
import glob
from Tools.config_helpers import loadConfig
cfg = loadConfig()

version = cfg['meta']['version']
#tag = version.replace('.','p')
tag = 'topW_v0.2.3' #!FIXME update this!

data_path = os.path.join(cfg['meta']['localSkim'], tag)

# All samples. Careful: There are some overlaps.
groups_2018 = {
    'tW_scattering': ['/tW_scattering[-_]'],
    'topW_v2':       ['/ProjectMetis_TTWJetsToLNuEWK'],
    'topW_v3':       ['/ProjectMetis_TTWplusJetsToLNuEWK', '/ProjectMetis_TTWminusJetsToLNuEWK'],
    'topW_EFT_cp8':  ['/ProjectMetis_TTWJetsToLNuEWK_5f_EFT_myNLO_full'],
    'topW_EFT_mix':  ['/ProjectMetis_TTWJetsToLNuEWK_5f_EFT_mix_myNLO_full'],
    # careful - TTX is a sum of all TTX but TTW
    'TTXnoW':        ['/TTZToLLNuNu[-_]', '/ST_tWll[-_]', '/ST_tWnunu[-_]', '/TH[W,Q][-_]', '/TT[T,W,Z][T,W,Z][-_]', '/tZq[-_]', '/ttHToNonbb[-_]'],
    'TTW':           ['/TTWJets'],
    'TTH':           ['/TH[W,Q][-_]', '/ttHToNonbb[-_]'],
    'TTZ':           ['/TTZToLLNuNu[-_]', '/ST_tWll[-_]', '/ST_tWnunu[-_]', '/tZq[-_]', '/TT[W,Z][W,Z][-_]'],
    'TTTT':          ['/TTTT[-_]'],
    'ttbar':         ['/TTTo2L2Nu', '/TTToSemiLeptonic', '/ST_[s,t]-channel', '/ST_tW[-_]'],
    'top':           ['/TTTo2L2Nu', '/TTToSemiLeptonic', '/ST_[s,t]-channel', '/ST_tW[-_]'],
    'top1l':         ['/TTToSemiLeptonic', '/ST_[s,t]-channel', '/ST_tW[-_]'],
    'ttbar1l':       ['/TTToSemiLeptonic'],
    #'ttbar2l':       ['/TTTo2L2Nu', '/ST_[s,t]-channel', '/ST_tW[-_]'],
    'top2l':         ['/TTTo2L2Nu', '/ST_t-channel', '/ST_tW[-_]'],
    'ttbar2l':       ['/TTTo2L2Nu'],
    'ttbar1l_MG':    ['/TTJets_SingleLept'],
    'TTW':           ['/TTWJets'],
    'wjets':         ['/W[1-4]JetsToLNu[-_]'],
    'diboson':       ['/WZTo.*amcatnloFXFX', '/WWTo', '/ZZTo', '/[W,Z][W,Z][W,Z][-_]'], # there's also a powheg sample
    'triboson':      ['/[W,Z][W,Z][W,Z][-_]'],
    'WW':            ['/WWTo'], 
    'WZ':            ['/WZTo.*amcatnloFXFX'], # there's also a powheg sample
    'DY':            ['/DYJetsToLL'],

    'TTWZ':             ['/TTWZ[-_]'],
    'TTZToLLNuNu_M-10': ['/TTZToLLNuNu_M-10[-_]'],
    'TTZToLL_M-1to10':  ['/TTZToLL_M-1to10[-_]'],
    'TTZToQQ':          ['/TTZToQQ[-_]'],
    'ST_tWll':          ['/ST_tWll[-_]'],
    'tZq':              ['/tZq[-_]'],

    'ttHToNonbb':       ['/ttHToNonbb[-_]'],
    'THW':              ['/THW[-_]'],
    'THQ':              ['/THQ[-_]'],

    'TTWJetsToLNu':     ['/TTWJetsToLNu'],
    'TTWJetsToQQ':      ['/TTWJetsToQQ'],

    'TTToSemiLeptonic': ['/TTToSemiLeptonic'],
    'ST_t-channel':     ['/ST_t-channel'],
    'ST_s-channel':     ['/ST_s-channel'],
    'ST_tW':            ['/ST_tW[-_]'],

    'MuonEG':        ['/MuonEG_Run2018[ABCD]'],
}

#samples_2016 = glob.glob(data_path_2016 + '/*')
#fileset_2016 = { group: [] for group in groups_2016.keys() }
#
#samples_2017 = glob.glob(data_path_2017 + '/*')
#fileset_2017 = { group: [] for group in groups_2017.keys() }

samples_2018 = glob.glob(data_path + '/*')
fileset_2018 = { group: [] for group in groups_2018.keys() }

#for sample in samples_2016:
#    for group in groups_2016.keys():
#        for process in groups_2016[group]:
#            if re.search(process, sample):
#                fileset_2016[group] += glob.glob(sample+'/*.root')
#
#fileset_2016_small = { sample: fileset_2016[sample][:2] for sample in fileset_2016.keys() }
#
#for sample in samples_2017:
#    for group in groups_2017.keys():
#        for process in groups_2017[group]:
#            if re.search(process, sample):
#                fileset_2017[group] += glob.glob(sample+'/*.root')
#
#fileset_2017_small = { sample: fileset_2017[sample][:2] for sample in fileset_2017.keys() }

for sample in samples_2018:
    for group in groups_2018.keys():
        for process in groups_2018[group]:
            if re.search(process, sample):
                fileset_2018[group] += glob.glob(sample+'/*.root')

fileset_2018_small = { sample: fileset_2018[sample][:2] for sample in fileset_2018.keys() }

