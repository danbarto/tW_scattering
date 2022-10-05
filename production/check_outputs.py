#!/usr/bin/env python3

import uproot
import glob
import os

# 2016 is v10 because of the PU mixing files, otherwise should be the same.
#
# /hadoop/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL16_preVFP_NANO_v10/  # NOTE: 1325436 events
# /hadoop/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL17_NANO_v9/  # NOTE: 2468795 events
# /hadoop/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL18_NANO_v9/  # NOTE: 3415000 events

#file_list = glob.glob("/hadoop/cms/store/user/dspitzba/ProjectMetis/TTW_5f_EFT_NLO_RunIISummer20UL16_preVFP_NANO_v10/*.root")
#file_list = glob.glob("/ceph/cms/store/user/dspitzba/ProjectMetis/DoubleMuon_Run2022B-PromptReco-v1_MINIAOD_v18/merged/*.root")
## non-merged: 954643
## merged: 954643
file_list = glob.glob("/ceph/cms/store/user/dspitzba/ProjectMetis/DoubleMuon_Run2022A-PromptReco-v1_MINIAOD_v18/merged/*.root")
## non-merged: 36540
## merged: 36540
#file_list = glob.glob("/ceph/cms/store/user/dspitzba/ProjectMetis/WW_TuneCP5_13p6TeV-pythia8_Run3Winter22MiniAOD-122X_mcRun3_2021_realistic_v9-v2_MINIAODSIM_v18/merged/*.root")
## non-merged: 15130996
## merged: same
corrupted_list = []
nevents = 0

for f_in in file_list:
    try:
        with uproot.open(f_in) as f:
            try:
                #nevents += len(f["Events"].arrays("LHEWeight_cpt_0p_cpQM_3p_nlo"))
                nevents += len(f["Events"].arrays("MET_pt"))
            #except (KeyError, KeyInFileError):
            except KeyError:
                corrupted_list.append(f_in)
                print (f_in, "seems to be corrupted")
    except (ValueError, OSError) as e:
        corrupted_list.append(f_in)
        print (f_in, "seems to be corrupted")

print (f"Found a total of {nevents}")


if False:
    for f in corrupted_list:
        os.remove(f)
