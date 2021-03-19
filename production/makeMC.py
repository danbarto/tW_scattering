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
        'TTWJetsToLNuEWK_5f_EFT_myNLO': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz', # that's EFT
        'TTWJetsToLNuEWK_5f_EFT_myNLO_cpt8': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_cpt8_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz', # that's EFT
        'TTWJetsToLNuEWK_5f_NLO':       '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_NLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz', # that's the SM
    }

    total_summary = {}

    extra_requirements = "true"

    tag = "v1"
    events_per_point = 500000
    events_per_job = 10000
    #events_per_point = 500
    #events_per_job = 100
    njobs = int(events_per_point)//events_per_job

    for reqname in requests:
        gridpack = requests[reqname]

        #reqname = "TTWJetsToLNuEWK_5f_EFT_myNLO"
        #gridpack = '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz'

        task = CondorTask(
                sample = DummySample(dataset="/%s/RunIIAutumn18/NANOGEN"%reqname,N=njobs,nevents=int(events_per_point)),
                output_name = "output.root",
                executable = "executables/condor_executable.sh",
                tarfile = "package.tar.gz",
                open_dataset = False,
                files_per_output = 1,
                arguments = gridpack,
                condor_submit_params = {
                    "sites":"T2_US_UCSD", # 
                    "classads": [
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        ],
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                )

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/tW_gen/").do()

if __name__ == "__main__":

    for i in range(500):
        submit()
        time.sleep(60*60)

