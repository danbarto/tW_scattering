'''
Produce a NanoAOD sample on condor, using metis.
Inspired by https://github.com/aminnj/scouting/blob/master/generation/submit_jobs.py

'''

from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample,DummySample,DBSSample
from metis.Path import Path
from metis.StatsParser import StatsParser
import time

def submit():

    samples = [
        '/DoubleMuon/Run2022A-PromptReco-v1/MINIAOD',
        '/DoubleMuon/Run2022B-PromptReco-v1/MINIAOD',
        '/DoubleMuon/Run2022C-PromptReco-v1/MINIAOD',
        '/SingleMuon/Run2022A-PromptReco-v1/MINIAOD',
        '/SingleMuon/Run2022B-PromptReco-v1/MINIAOD',
        '/SingleMuon/Run2022C-PromptReco-v1/MINIAOD',
        '/EGamma/Run2022A-PromptReco-v1/MINIAOD',
        '/EGamma/Run2022B-PromptReco-v1/MINIAOD',
        '/EGamma/Run2022C-PromptReco-v1/MINIAOD',
        '/EGamma/Run2022D-PromptReco-v1/MINIAOD',
        '/MuonEG/Run2022A-PromptReco-v1/MINIAOD',
        '/MuonEG/Run2022B-PromptReco-v1/MINIAOD',
        '/MuonEG/Run2022C-PromptReco-v1/MINIAOD',
        '/MuonEG/Run2022D-PromptReco-v1/MINIAOD',

        "/DYJetsToLL_M-10to50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9_ext2-v2/MINIAODSIM",
        "/TTTo2L2Nu_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/TWminus_DR_AtLeastOneLepton_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/TbarWplus_DR_AtLeastOneLepton_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/WW_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/WZ_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        "/ZZ_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",

    ]

    total_summary = {}

    extra_requirements = "true"

    tag = "v1"
    files_per_output = 1

    for sample_name in samples:

        if sample_name.count("Run2022B") or sample_name.count("Run2022A"):
            cfg_file = "nano_runB_v10_cfg.py"
        elif sample_name.count("Run2022"):
            cfg_file = "nano_runC_v10_cfg.py"
        else:
            cfg_file = "nano_mc_v10_cfg.py"

        print ("Sample:", sample_name)
        print ("Config", cfg_file)
        
        task = CondorTask(
                sample = DBSSample(dataset=sample_name),
                output_name = "nanoAOD.root",
                executable = "executables/condor_executable_run3_nano.sh",
                tarfile = "run3_cmssw.tar.gz",
                additional_input_files = ["psets/2022/"+cfg_file],
                open_dataset = False,
                files_per_output = 1,
                arguments = cfg_file,
                condor_submit_params = {
                    "sites":"T2_US_UCSD", #
                    "memory": 1950,
                    "cpus": 1,
                    #"memory": 15600,
                    #"cpus": 8,
                    "classads": [
                        ["metis_extraargs",""],
                        #["JobBatchName",reqname],
                        #["IS_CLOUD_JOB", "yes"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'
                    },
                tag = tag,
                min_completion_fraction = 1.00,
                )

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/Run3_prod/").do()

if __name__ == "__main__":

    print ("Running")

    for i in range(500):
        submit()
        nap_time = 3
        time.sleep(60*60*nap_time)  # take a super-long power nap

