#!/bin/bash

OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSWVERSION=$5
SCRAMARCH=$6
GRIDPACK=$7

# # FIXME FIXME
# OUTPUTDIR="/hadoop/cms/store/user/namin/metis_test/dimuon/"
# OUTPUTNAME="output"
# INPUTFILENAMES="dummy"
# IFILE="15"
# CMSSWVERSION="dummy"
# SCRAMARCH="slc6_amd64_gcc700"
# NEVENTS="2"

# Make sure OUTPUTNAME doesn't have .root since we add it manually
OUTPUTNAME=$(echo $OUTPUTNAME | sed 's/\.root//')

export SCRAM_ARCH=${SCRAMARCH}

function getjobad {
    grep -i "^$1" "$_CONDOR_JOB_AD" | cut -d= -f2- | xargs echo
}

function setup_chirp {
    if [ -e ./condor_chirp ]; then
    # Note, in the home directory
        mkdir chirpdir
        mv condor_chirp chirpdir/
        export PATH="$PATH:$(pwd)/chirpdir"
        echo "[chirp] Found and put condor_chirp into $(pwd)/chirpdir"
    elif [ -e /usr/libexec/condor/condor_chirp ]; then
        export PATH="$PATH:/usr/libexec/condor"
        echo "[chirp] Found condor_chirp in /usr/libexec/condor"
    else
        echo "[chirp] No condor_chirp :("
    fi
}

function chirp {
    # Note, $1 (the classad name) must start with Chirp
    condor_chirp set_job_attr_delayed $1 $2
    ret=$?
    echo "[chirp] Chirped $1 => $2 with exit code $ret"
}

function stageout {
    COPY_SRC=$1
    COPY_DEST=$2
    retries=0
    COPY_STATUS=1
    until [ $retries -ge 3 ]
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
        echo "Sleeping for 30m"
        sleep 30m
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

function setup_environment {
    if [ -r "$OSGVO_CMSSW_Path"/cmsset_default.sh ]; then
        echo "sourcing environment: source $OSGVO_CMSSW_Path/cmsset_default.sh"
        source "$OSGVO_CMSSW_Path"/cmsset_default.sh
    elif [ -r "$OSG_APP"/cmssoft/cms/cmsset_default.sh ]; then
        echo "sourcing environment: source $OSG_APP/cmssoft/cms/cmsset_default.sh"
        source "$OSG_APP"/cmssoft/cms/cmsset_default.sh
    elif [ -r /cvmfs/cms.cern.ch/cmsset_default.sh ]; then
        echo "sourcing environment: source /cvmfs/cms.cern.ch/cmsset_default.sh"
        source /cvmfs/cms.cern.ch/cmsset_default.sh
    else
        echo "ERROR! Couldn't find $OSGVO_CMSSW_Path/cmsset_default.sh or /cvmfs/cms.cern.ch/cmsset_default.sh or $OSG_APP/cmssoft/cms/cmsset_default.sh"
        exit 1
    fi
}

function setup_cmssw {
  CMSSW=$1
  export SCRAM_ARCH=$2
  scram p CMSSW $CMSSW
  cd $CMSSW
  eval $(scramv1 runtime -sh)
  cd -
}


function edit_psets {
    gridpack="$1"
    seed=$2
    nevents=$3

    # gensim
    echo "process.RandomNumberGeneratorService.externalLHEProducer.initialSeed = $seed" >> $gencfg
    echo "process.externalLHEProducer.args = [\"$gridpack\"]" >> $gencfg
    echo "process.externalLHEProducer.nEvents = $nevents" >> $gencfg
    echo "process.maxEvents.input = $nevents" >> $gencfg
    echo "process.source.firstLuminosityBlock = cms.untracked.uint32($seed)" >> $gencfg

    # nanogen
    echo "process.maxEvents.input = $nevents" >> $nanocfg
    echo "process.source.fileNames = [\"file:output_gen.root\"]" >> $nanocfg


}


echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "OUTPUTNAME: $OUTPUTNAME"
echo "INPUTFILENAMES: $INPUTFILENAMES"
echo "IFILE: $IFILE"
echo "CMSSWVERSION: $CMSSWVERSION"
echo "SCRAMARCH: $SCRAMARCH"

echo "GLIDEIN_CMSSite: $GLIDEIN_CMSSite"
echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"
echo "tag: $(getjobad tag)"
echo "taskname: $(getjobad taskname)"

NEVENTS=$(getjobad param_nevents)

echo -e "\n--- end header output ---\n" #                       <----- section division


gencfg="psets/2018/gen_cfg.py"
nanocfg="psets/2018/nano_cfg.py"

setup_chirp
setup_environment

# Make temporary directory to keep original dir clean
# Go inside and extract the package tarball
mkdir temp
cd temp
cp ../*.gz .
tar xf *.gz

edit_psets $GRIDPACK $IFILE $NEVENTS

echo "before running: ls -lrth"
ls -lrth

echo -e "\n--- begin running ---\n" #                           <----- section division

chirp ChirpMetisStatus "before_cmsRun"

setup_cmssw CMSSW_10_2_9 slc6_amd64_gcc700 
cmsRun $gencfg
cmsRun $nanocfg
CMSRUN_STATUS=$?

chirp ChirpMetisStatus "after_cmsRun"

echo "after running: ls -lrth"
ls -lrth

if [[ $CMSRUN_STATUS != 0 ]]; then
    echo "Removing output file because cmsRun crashed with exit code $?"
    rm ${OUTPUTNAME}.root
    exit 1
fi

echo -e "\n--- end running ---\n" #                             <----- section division

echo -e "\n--- begin copying output ---\n" #                    <----- section division

echo "Sending output file $OUTPUTNAME.root"

if [ ! -e "$OUTPUTNAME.root" ]; then
    echo "ERROR! Output $OUTPUTNAME.root doesn't exist"
    exit 1
fi

echo "time before copy: $(date +%s)"
chirp ChirpMetisStatus "before_copy"

COPY_SRC="file://`pwd`/${OUTPUTNAME}.root"
COPY_DEST="gsiftp://gftp.t2.ucsd.edu${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST

echo -e "\n--- end copying output ---\n" #                      <----- section division

echo "time at end: $(date +%s)"

chirp ChirpMetisStatus "done"

