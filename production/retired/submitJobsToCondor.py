### Script to submit Condor jobs for LHE event generation and showering at UCSD

### Authors:
### Ana Ovcharova
### Dustin Anderson

import os
import sys
import argparse
import shutil

from submitCondor import submitCondorJob

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proc', help="Names of physics model")
    parser.add_argument('--fragment', '-f', help='Path to gen fragment', required=True)
    parser.add_argument('--nevents', '-n', help="Number of events per job", type=int, default=25000)
    parser.add_argument('--njobs', '-j', help="Number of condor jobs", type=int, default=1)
    parser.add_argument('--proxy', dest="proxy", help="Path to proxy", default=os.environ["X509_USER_PROXY"])
    parser.add_argument('--rseed-start', dest='rseedStart', help='Initial value for random seed', type=int, default=500)
    parser.add_argument('--no-sub', dest='noSub', action='store_true', help='Do not submit jobs')
    parser.add_argument('--executable', help='Path to executable that should be run', default = script_dir+'/runLHEPythiaJob.sh')
    args = parser.parse_args()

    proc = args.proc
    fragment = args.fragment
    nevents = args.nevents
    njobs = args.njobs
    rseedStart = args.rseedStart
    executable = args.executable

    label = 'tW_scattering'

    script_dir = os.path.dirname(os.path.realpath(__file__))

    out_dir='/hadoop/cms/store/user/'+os.environ['USER']+'/tW_scattering'

    #need to transfer gen fragment
    fragfile = os.path.basename(fragment)

    logDir = os.path.join("logs",proc)
    if not os.path.isdir(logDir):
        os.makedirs(logDir)
    #else:
    #    shutil.rmtree(logDir)
    #    os.makedirs(logDir)

    outdir = out_dir+'/'+label


    for j in range(0,njobs):
        rseed = str(rseedStart+j)
        print "Random seed",rseed
        options = [str(nevents), str(rseed), outdir]
        print "Options:",(' '.join(options))
        submitCondorJob(proc, executable, options, fragment, label=label+'_'+rseed, submit=(not args.noSub), proxy=args.proxy)

