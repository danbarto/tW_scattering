#!/bin/bash

# This is nanoAOD based sample making condor executable for CondorTask of ProjectMetis. Passed in arguments are:
# arguments = [outdir, outname_noext, inputs_commasep, index, cmssw_ver, scramarch, self.arguments]

OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSW_VERSION=$5
SCRAM_ARCH=$6

VERSION=$7
SUMWEIGHT=$8
ISDATA=$9
YEAR=${10}
ERA=${11}
ISFASTSIM=${12}
SKIM=${13}
GITHUBUSER=${14}

OUTPUTNAME=$(echo $OUTPUTNAME | sed 's/\.root//')

## from https://github.com/aminnj/ProjectMetis/blob/master/metis/executables/condor_cmssw_exe.sh#L76
function stageout {
    COPY_SRC=$1
    COPY_DEST=$2
    retries=0
    COPY_STATUS=1
    until [ $retries -ge 5 ]
    do
        echo "Stageout attempt $((retries+1)): env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}
        COPY_STATUS=$?
        if [ $COPY_STATUS -ne 0 ]; then
            echo "Failed stageout attempt $((retries+1))"
        else
            echo "Successful stageout with $retries retries"
            break
        fi
        retries=$[$retries+1]
        echo "Sleeping for 15m"
        sleep 15m
    done
    if [ $COPY_STATUS -ne 0 ]; then
        echo "Removing output file because gfal-copy crashed with code $COPY_STATUS"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-rm --verbose ${COPY_DEST}
        REMOVE_STATUS=$?
        if [ $REMOVE_STATUS -ne 0 ]; then
            echo "Uhh, gfal-copy crashed and then the gfal-rm also crashed with code $REMOVE_STATUS"
            echo "You probably have a corrupt file sitting on hadoop now."
            exit 1
        fi
    fi
}

echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "OUTPUTNAME: $OUTPUTNAME"
echo "INPUTFILENAMES: $INPUTFILENAMES"
echo "IFILE: $IFILE"
echo "CMSSW_VERSION: $CMSSW_VERSION"
echo "SCRAM_ARCH: $SCRAM_ARCH"

echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"

echo -e "\n--- end header output ---\n" #                       <----- section division
ls -ltrha
echo ----------------------------------------------

# Setup Enviroment
export SCRAM_ARCH=$SCRAM_ARCH
source /cvmfs/cms.cern.ch/cmsset_default.sh
#pushd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src/ > /dev/null
#eval `scramv1 runtime -sh`
#popd > /dev/null
scramv1 project CMSSW $CMSSW_VERSION
cd $CMSSW_VERSION/src
eval `scramv1 runtime -sh`

# The output name is the sample name for stop baby
SAMPLE_NAME=$OUTPUTNAME
NEVENTS=-1

echo $VERSION

# checkout the package
git clone --branch $VERSION --depth 1  https://github.com/$GITHUBUSER/nanoAOD-tools.git PhysicsTools/NanoAODTools

scram b


echo "Input:"
echo $INPUTFILENAMES

OUTFILE=$(python -c "print('$INPUTFILENAMES'.split('/')[-1].split('.root')[0]+'_Skim.root')")

echo $OUTFILE

echo "Running python PhysicsTools/NanoAODTools/scripts/run_processor.py $INPUTFILENAMES $SUMWEIGHT $ISDATA $YEAR $ERA $ISFASTSIM"

python PhysicsTools/NanoAODTools/scripts/run_processor.py $INPUTFILENAMES $SUMWEIGHT $ISDATA $YEAR $ERA $ISFASTSIM $SKIM

mv tree.root ${OUTPUTNAME}_${IFILE}.root

# Rigorous sweeproot which checks ALL branches for ALL events.
# If GetEntry() returns -1, then there was an I/O problem, so we will delete it
python << EOL
import ROOT as r
import os
import traceback
foundBad = False
try:
    f1 = r.TFile("${OUTPUTNAME}_${IFILE}.root")
    t = f1.Get("Events")
    nevts = t.GetEntries()
    print "[SweepRoot] ntuple has %i events." % t.GetEntries()
    if int(t.GetEntries()) <= 0:
        foundBad = True
    for i in range(0,t.GetEntries(),1):
        if t.GetEntry(i) < 0:
            foundBad = True
            print "[RSR] found bad event %i" % i
            break
except Exception as ex:
    msg = traceback.format_exc()
    print "Encounter error during SweepRoot:"
    print msg
    foundBad = True
if foundBad:
    print "[RSR] removing output file because it does not deserve to live"
    os.system("rm ${OUTPUTNAME}_${IFILE}.root")
else:
    print "[RSR] passed the rigorous sweeproot"
EOL

echo -e "\n--- end running ---\n" #                             <----- section division

# Copy back the output file

COPY_SRC="file://`pwd`/${OUTPUTNAME}_${IFILE}.root"
COPY_DEST="gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST


echo -e "\n--- cleaning up ---\n" #                             <----- section division
cd ../../
rm -r $CMSSW_VERSION/


