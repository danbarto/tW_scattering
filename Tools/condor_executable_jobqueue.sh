#!/usr/bin/env bash

function getjobad {
    grep -i "^$1" "$_CONDOR_JOB_AD" | cut -d= -f2- | xargs echo
}

if [ -r "$OSGVO_CMSSW_Path"/cmsset_default.sh ]; then source "$OSGVO_CMSSW_Path"/cmsset_default.sh
elif [ -r "$OSG_APP"/cmssoft/cms/cmsset_default.sh ]; then source "$OSG_APP"/cmssoft/cms/cmsset_default.sh
elif [ -r /cvmfs/cms.cern.ch/cmsset_default.sh ]; then source /cvmfs/cms.cern.ch/cmsset_default.sh
else
    echo "ERROR! Couldn't find $OSGVO_CMSSW_Path/cmsset_default.sh or /cvmfs/cms.cern.ch/cmsset_default.sh or $OSG_APP/cmssoft/cms/cmsset_default.sh"
    exit 1
fi

hostname

if ! ls /hadoop/cms/store/ ; then
    echo "ERROR! hadoop is not visible, so the worker would be useless later. dying."
    exit 1
fi

echo "Entry point. I should have all the tar balls now!"
ls -lrth

mkdir temp ; cd temp

mv ../{daskworkerenv.tar.*,*.py,analysis.tar.*} .
echo "started extracting at $(date +%s)"
tar xf daskworkerenv.tar.*
tar xf analysis.tar.*
echo "finished extracting at $(date +%s)"

source daskworkerenv/bin/activate

ls -lrth
export PYTHONPATH=`pwd`:$PYTHONPATH
export PATH=`pwd`/daskworkerenv/bin:$PATH

echo "I'm currently here:"
pwd
cd tW_scattering/
echo "In tW_scattering directory"
pwd
ls -ltrh
export TWHOME=`pwd`
export PYTHONPATH=`pwd`:${PYTHONPATH}
cd ../

echo "after checking out tW code"
pwd
ls -lrth
echo $PYTHONPATH
echo "Test run for python"
python -m Tools.cutflow

echo "Downloading some files that will be handy"
wget https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions18/13TeV/ReReco/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt


$@
