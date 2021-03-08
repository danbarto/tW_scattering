'''
Produce a nanoGEN sample on condor, using metis.
Inspired by https://github.com/aminnj/scouting/blob/master/generation/submit_jobs.py

'''

from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample,DummySample
from metis.Path import Path
from metis.StatsParser import StatsParser
import time

def submit():

    requests = {
        #'TTWJetsToLNuEWK_5f_NLO':       '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_NLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball_retired.tar.xz', # that's the SM point, but using the SMEFT model. No lepton filtering, so name is actually confusing
        #'TTWJetsToLNuEWK_5f_NLO_v2':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_NLO_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz', # that's the actual SM
        #'TTWplusJetsToLNuEWK_5f_NLO_v2':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWplusJetsToLNuEWK_5f_NLO_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz', # that's the actual SM
        #'TTWminusJetsToLNuEWK_5f_NLO_v2':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWminusJetsToLNuEWK_5f_NLO_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz', # that's the actual SM
        #'TTWJetsToLNuEWK_5f_EFT_myNLO_full':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_cpt8_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz', # one of the BSM points
        'TTWJetsToLNuEWK_5f_EFT_mix_myNLO_full':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz',  # EFT mix
    }

    total_summary = {}

    extra_requirements = "true"

    tag = "v4"
    events_per_point = 500000 # produced 500k events before
    events_per_job = 1000 # up to 2000 works
    #events_per_point = 500
    #events_per_job = 100
    njobs = int(events_per_point)//events_per_job

    for reqname in requests:
        gridpack = requests[reqname]

        #reqname = "TTWJetsToLNuEWK_5f_EFT_myNLO"
        #gridpack = '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz'

        task = CondorTask(
                sample = DummySample(dataset="/%s/RunIIAutumn18/NANO"%reqname,N=njobs,nevents=int(events_per_point)),
                output_name = "nanoAOD.root",
                executable = "executables/condor_executable_Autumn18.sh",
                tarfile = "package.tar.gz",
                #scram_arch = "slc7_amd64_gcc630",
                open_dataset = False,
                files_per_output = 1,
                arguments = gridpack,
                condor_submit_params = {
                    "sites":"T2_US_UCSD", # 
                    "classads": [
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        #["SingularityImage", "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'  # && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                min_completion_fraction = 0.95,
                )

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/tW_gen/").do()

if __name__ == "__main__":

    for i in range(500):
        submit()
        nap_time = 1
        time.sleep(60*60*nap_time)  # take a super-long power nap

