'''
Produce a NanoAOD sample on condor, using metis.
Inspired by https://github.com/aminnj/scouting/blob/master/generation/submit_jobs.py

'''
import os

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

        #"/DYJetsToLL_M-10to50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9_ext2-v2/MINIAODSIM",
        #"/TTTo2L2Nu_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/TWminus_DR_AtLeastOneLepton_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/TbarWplus_DR_AtLeastOneLepton_CP5_13p6TeV_powheg-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        ##"/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/WW_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/WZ_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",
        #"/ZZ_TuneCP5_13p6TeV-pythia8/Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM",

    ]

    total_summary = {}

    extra_requirements = "true"

    tag = "v18"
    files_per_output = 1

    make_tasks = []
    merge_tasks = []

    for sample_name in samples:

        if sample_name.count("Run2022B") or sample_name.count("Run2022A"):
            cfg_file = "nano_runB_v10_cfg.py"
            min_completion = 0.98
        elif sample_name.count("Run2022"):
            cfg_file = "nano_runC_v10_cfg.py"
            min_completion = 0.98
        else:
            cfg_file = "nano_mc_v10_cfg.py"
            min_completion = 0.90

        print ("Sample:", sample_name)
        print ("Config", cfg_file)

        sample = DBSSample(dataset=sample_name, remove_empty_files = True)
        #print (sample.get_files())

        task = CondorTask(
                sample = sample,
                output_name = "nanoAOD.root",
                executable = "executables/condor_executable_run3_nano.sh",
                tarfile = "run3_cmssw_recompile.tar.gz",
                additional_input_files = ["psets/2022/"+cfg_file],
                open_dataset = False,
                files_per_output = 1,
                arguments = cfg_file,
                max_jobs=0,
                cmssw_version="CMSSW_12_4_7",
                scram_arch="slc7_amd64_gcc10",
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
                min_completion_fraction = min_completion,
                )

        make_tasks.append(task)


        merge_task = CondorTask(
            sample = DirectorySample(
                dataset="merge_"+sample.get_datasetname(),
                location=task.get_outputdir(),
            ),
            executable = "executables/condor_executable_merge.sh",
            #arguments = "WHv1p2 %s %s %s"%(year, isData, isFast),
            files_per_output = 10,
            output_dir = task.get_outputdir() + "/merged",
            output_name = "nanoAOD.root",
            output_is_tree = True,
            # check_expectedevents = True,
            tag = tag,
            # condor_submit_params = {"sites":"T2_US_UCSD"},
            # cmssw_version = "CMSSW_9_2_8",
            # scram_arch = "slc6_amd64_gcc530",
            condor_submit_params = {"sites":"T2_US_UCSD"},
            cmssw_version="CMSSW_12_4_7",
            scram_arch="slc7_amd64_gcc10",
            min_completion_fraction = 1.00,
        )

        merge_tasks.append(merge_task)

    return make_tasks, merge_tasks


        #task.process()
        #total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        #StatsParser(data=total_summary, webdir="~/public_html/dump/Run3_prod/").do()

if __name__ == "__main__":

    force_merge = False  # FIXME not sure if I want to implement something like this
    print ("Running")

    for i in range(500):
        make_tasks, merge_tasks = submit()

        total_summary = {}

        for make_task, merge_task in zip(make_tasks,merge_tasks):
            if not os.path.isdir(make_task.get_outputdir()):
                os.makedirs(make_task.get_outputdir())
            if not os.path.isdir(make_task.get_outputdir()+'/merged'):
                os.makedirs(make_task.get_outputdir()+'/merged')

            if not force_merge:
                make_task.process()

            frac = make_task.complete(return_fraction=True)
            if frac >= make_task.min_completion_fraction or force_merge:
            # if maker_task.complete():
            #    do_cmd("mkdir -p {}/merged".format(maker_task.get_outputdir()))
            #    do_cmd("mkdir -p {}/skimmed".format(maker_task.get_outputdir()))
                merge_task.reset_io_mapping()
                merge_task.update_mapping()
                merge_task.process()

            total_summary[make_task.get_sample().get_datasetname()] = make_task.get_task_summary()
            total_summary[merge_task.get_sample().get_datasetname()] = merge_task.get_task_summary()

            print (frac)

        # parse the total summary and write out the dashboard
        StatsParser(data=total_summary, webdir="~/public_html/dump/Run3_prod/").do()

        print ("Nap time.")
        nap_time = 3
        time.sleep(60*60*nap_time)  # take a super-long power nap

