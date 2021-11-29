'''
Produce a NanoAOD sample on condor, using metis.
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
        #'TTWJetsToLNuEWK_5f_EFT_mix_myNLO_full':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz',  # EFT mix
        #'TTWJetsToLNuEWK_5f_EFT_cpq3_4_myNLO_full':    '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks//TTWJetsToLNuEWK_5f_EFT_myNLO_cpq3_4_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz',  # C_pq3 = 4
        #'TTWJetsToLNuEWK_5f_NLO': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_NLO_slc7_amd64_gcc730_CMSSW_9_3_16_tarball.tar.xz',
        #'TTWJetsToLNuEWK_5f_SMEFTatNLO_weight': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
        'TTW_5f_EFT_NLO': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTW_5f_EFT_NLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
    }

    total_summary = {}

    extra_requirements = "true"

    # v6+ is UL

    #campaign = 'UL16'
    campaign = 'UL16_preVFP'
    tag = "v9"
    #tag = "v8_pre"
    #events_per_point = 250000
    #events_per_job = 250
    #events_per_point = 2000000
    events_per_point = 1500000
    events_per_job = 5000  ## 2000 -> 4h runtime, 4000 -> 8h runtime
    #events_per_point = 200
    #events_per_job = 40
    njobs = int(events_per_point)//events_per_job

    for reqname in requests:
        gridpack = requests[reqname]

        task = CondorTask(
                sample = DummySample(dataset="/%s/RunIISummer20%s/NANO"%(reqname, campaign),N=njobs,nevents=int(events_per_point)),
                output_name = "nanoAOD.root",
                executable = "executables/condor_executable_%s.sh"%campaign,
                tarfile = "package.tar.gz",
                additional_input_files = [gridpack],
                open_dataset = False,
                files_per_output = 1,
                arguments = gridpack.split('/')[-1],
                condor_submit_params = {
                    "sites":"T2_US_UCSD", #
                    #"memory": 1950,
                    #"cpus": 1,
                    "memory": 15600,
                    "cpus": 8,
                    "classads": [
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        #["IS_CLOUD_JOB", "yes"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'
                    },
                tag = tag,
                min_completion_fraction = 0.90,
                )

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/tW_gen/").do()

if __name__ == "__main__":

    print ("Running")

    for i in range(500):
        submit()
        nap_time = 3
        time.sleep(60*60*nap_time)  # take a super-long power nap

