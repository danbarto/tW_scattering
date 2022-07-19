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
        #'TTZ_EFT_NLO_fixed': '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTZ_5f_NLO_fixed_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
        #'TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO': '/ceph/cms/store/user/dspitzba/tW_scattering/gridpacks/TTW_5f_EFT_NLO_test_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz', # this corresponds to TTWToLNu_TtoAll_aTtoLep_5f_EFT_NLO
        #'TTWToLNu_TtoLep_aTtoHad_5f_EFT_NLO': '/ceph/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWToLNu_TtoLep_aTtoHad_5f_EFT_NLO_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
        'TTZ_5f_LO_SMEFT': '',
        'TTZ_5f_NLO_SMEFT': '',
    }

    total_summary = {}

    extra_requirements = "true"

    # v6+ is UL

    tag = "v13"
    events_per_point = int(2.5e6)
    #events_per_point = 200
    events_per_job = 10000
    njobs = int(events_per_point)//events_per_job

    for reqname in requests:
        gridpack = requests[reqname]

        nlo = reqname.count("NLO") > 0  # NOTE: is this good enough?
        exe = "executables/condor_executable_nanogen_nlo.sh" if nlo else "executables/condor_executable_nanogen.sh"

        print (f"Working on request: {reqname}")
        print (f"NLO mode:", nlo)
        print (f"Executable used: {exe}")
        print (f"Gridpack: {gridpack}")

        task = CondorTask(
                sample = DummySample(dataset="/%s/RunIISummer20_NanoGEN/NANO"%(reqname),N=njobs,nevents=int(events_per_point)),
                output_name = "output.root",
                executable = exe,
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

        # FIXME this would be useful
        #merge_task = CondorTask(
        #    sample = DirectorySample(
        #        dataset="merge_"+sample.get_datasetname(),
        #        location=maker_task.get_outputdir(),
        #    ),
        #    executable = "merge_executable.sh",
        #    arguments = " ".join([ str(x) for x in [tag, lumiWeightString, 1 if isData else 0, year, era, 1 if isFastSim else 0, args.skim, args.user ]] ),  # just use the same arguments for simplicity
        #    files_per_output = 20,
        #    output_dir = os.path.join(maker_task.get_outputdir(), 'merged'),
        #    #output_name = "output.root",
        #    output_is_tree = True,
        #    tag = tag_skim,
        #    condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
        #    cmssw_version = "CMSSW_10_2_9",
        #    scram_arch = "slc6_amd64_gcc700",
        #)

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/NanoGEN/").do()

if __name__ == "__main__":

    print ("Running")

    for i in range(500):
        submit()
        nap_time = 3
        time.sleep(60*60*nap_time)  # take a super-long power nap

