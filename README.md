# Measuring tW scattering

## Sample version history

- v0p1p12: data processing (no MC submitted yet)
- v0p1p11: **dilep/trilep skim only!** added variables for ttH lepton ID
- v0p1p10: intermediate version
- v0p1p9: used for initial tW scattering studies presented in SnT

## Setting up the code

Prerequisite: if you haven't, add this line to your `~/.profile`:
```
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

Currently lives within CMSSW_10_2_9. Set up in a fresh directory, recipe as follows:
```
cmsrel CMSSW_10_2_9
cd CMSSW_10_2_9/src
cmsenv
git cms-init

git clone --branch tW_scattering https://github.com/danbarto/nanoAOD-tools.git PhysicsTools/NanoAODTools

cd $CMSSW_BASE/src

git clone --recursive https://github.com/danbarto/tW_scattering.git

scram b -j 8
cmsenv

```

Then you can set up the tools to run coffea. You might get some error messages about packages that couldn't get uninstalled that you can ignore.
```
cd tW_scattering
source setup_environment.sh
deactivate
source activate_environment.sh
```

Every time you want to use coffea you need to activate the environment *this has changed in order to disentangle coffea from CMSSW*
```
source activate_environment.sh
```

To deactivate the coffea environment, just type `deactivate`


Use available nanoAOD tools to quickly process samples.

## Usage of jupyter notebooks

To install jupyter inside the coffeaEnv do the following (now part of the setup script too):
```
python -m ipykernel install --user --name=coffeaEnv
jupyter nbextension install --py widgetsnbextension --user
jupyter nbextension enable widgetsnbextension --user --py
```

To start the server run the following script:
```
source start_jupyter.sh
```
Which should return
```
Starting up jupyter server. Once this is done, run the following command on your computer:
  ssh -N -f -L localhost:8893:localhost:8893 johndoe@uaf-10.t2.ucsd.edu
Enabling notebook extension jupyter-js-widgets/extension...
      - Validating: OK
[I 05:05:26.927 NotebookApp] Serving notebooks from local directory: /home/users/johndoe/TTW/CMSSW_10_2_9/src/tW_scattering
[I 05:05:26.927 NotebookApp] 0 active kernels
[I 05:05:26.927 NotebookApp] The Jupyter Notebook is running at:
[I 05:05:26.928 NotebookApp] http://localhost:8893/?token=abcdefghijkl
[I 05:05:26.928 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 05:05:26.930 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8893/?token=abcdefghijkl
```

On your local machine do the following to connect to uaf
```
ssh -N -f -L localhost:8893:localhost:8893 uaf-10.t2.ucsd.edu
```

Then just paste the jupyter link into your browser and start working.

### Troubleshooting
- If the ssh command does not work, you might need to add a username like `ssh -N -f -L localhost:8893:localhost:8893 YOUR_UAF_USER@uaf-10.t2.ucsd.edu`, where YOUR_UAF_USER is your username on the uaf.
- If you already have a jupyter server running **on the uaf**, another port will be used instead of 8893. In this case, alter the `ssh -N -f ...` command so that it matches the ports. To stop a running jupyter server that is running but you can't find anymore, run `ps aux | grep $USER`. This will return you the list of processes attributed to your user. You should also find sth like
```
dspitzba 3964766  1.3  0.0  87556 44720 pts/17   S+   05:03   0:02 python /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw/CMSSW_10_2_9/external/slc6_amd64_gcc700/bin/jupyter-notebook --no-browser --port=8893
```
To stop this process, just type `kill 3964766`. In this case, 3964766 is the process id (PID) of the jupyter server process.
- If a port is already used on your machine because of a not properly terminated ssh session, run the following command **on your computer** `ps aux | grep ssh`. This returns a similar list as before. There should be a job like
```
daniel           27709   0.0  0.0  4318008    604   ??  Ss    8:11AM   0:00.00 ssh -N -f -L localhost:8893:localhost:8893 uaf-10.t2.ucsd.edu
```
Similarly, you can stop the process by running `kill 27709`.


## Get combine (for later)
Latest recommendations at https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#setting-up-the-environment-and-installation
```
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.0.1
scramv1 b clean; scramv1 b # always make a clean build
```

## for combineTools (for later)
```
cd $CMSSW_BASE/src
wget https://raw.githubusercontent.com/cms-analysis/CombineHarvester/master/CombineTools/scripts/sparse-checkout-https.sh; source sparse-checkout-https.sh
scram b -j 8
```
