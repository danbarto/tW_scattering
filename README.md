# Measuring tW scattering


## Setting up the code

Prerequisite: if you haven't, add this line to your `~/.profile`:
```
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

## Setting up miniconda

Skip this part if you already have conda running on uaf.

From within your home directory on the uaf, follow the below instructions to set up the tools to run coffea.
We do this in a virtual environment, using the miniconda environment management package.
You might get some error messages about packages that couldn't get uninstalled that you (usually) can ignore.

```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b 
```

Add conda to the end of ~/.bashrc, so relogin after executing this line
```
~/miniconda3/bin/conda init
```

Stop conda from activating the base environment on login
```
conda config --set auto_activate_base false
conda config --add channels conda-forge
```

Install package to tarball environments
```
conda install --name base conda-pack -y
```

## Setting up a environment for our work

Create environments with as much stuff from anaconda
```
conda create --name coffeadev uproot dask dask-jobqueue matplotlib pandas jupyter hdfs3 pyarrow fastparquet numba numexpr -y
```
And then install residual packages with pip
```
conda run --name coffeadev pip install jupyter-server-proxy coffea autopep8 jupyter_nbextensions_configurator klepto yahist
```

## Setting up an environment for the DASK workers

```
conda deactivate
conda create --name workerenv uproot dask dask-jobqueue pyarrow fastparquet numba numexpr boost-histogram onnxruntime -y
conda run --name workerenv pip install coffea yahist
conda activate workerenv
```

Pack it
```
conda pack -n workerenv --arcroot workerenv -f --format tar.gz --compress-level 9 -j 8 --exclude "*.pyc" --exclude "*.js.map" --exclude "*.a"
```

### DASK trouble shooting

I had to update my (local) python version, so you might have to run
```
conda install python=3.8.6
```

### Coffea developer mode

If you want to fix bugs, get the very latest version of coffea or are just adventurous you can install coffea direct from the github repository.
This gives instructions for our private fork that fixes the root export for our needs.
Based on coffea 0.7.0.
```
git clone https://github.com/CoffeaTeam/coffea
cd coffea
git remote add upstream git@github.com:danbarto/coffea.git
git fetch upstream
git checkout upstream/root_export
pip install --editable .[dev]
```
Full instructions are given [here](https://coffeateam.github.io/coffea/installation.html#for-developers).

### Setting up CMS software and analysis code

Deactivate the conda environment with 
```
conda deactivate
```
and then follow the below instructions.
Some of the code lives within CMSSW_10_2_9. Ideally set it up in a fresh directory, recipe as follows:
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

cd tW_scattering
```
You should be set up now. The following steps have to be repeated everytime you log in to uaf (from within tW_scattering)
```
source activate_conda.sh
```

To start a jupyter server just do
```
( conda activate coffeadev && jupyter notebook --no-browser )
```
In order to use jupyter you need to establish another ssh connection from your computer:
```
ssh -N -f -L localhost:8893:localhost:8893 johndoe@uaf-10.t2.ucsd.edu
```
Where 8893 represents the port (check the output of the jupyter server, uaf-10 should be replaced with the uaf you're working on, and johndoe by your user name, of course

## Using DASK

When you ran the above setup you should have created a `daskworkerenv.tar.gz` file. Move this into `tW_scattering/Tools/`. If you lost the tarball, just rerun
```
conda pack -n daskworkerenv --arcroot daskworkerenv -f --format tar.gz \
    --compress-level 9 -j 8 --exclude "*.pyc" --exclude "*.js.map" --exclude "*.a"
```

Then, run `packCode.sh`, which is located in `tW_scattering`. This script downloads the latest version of the tW_scattering code and creates a tarball that's shipped to the DASK workers.

```
ipython -i start_cluster.py
```
Starts a cluster with 50 workers. The scheduler address is automatically dumped into a text file so that it can be picked up easily in any notebook using coffea. You can get the status of the cluster by just typing `c` into the ipython prompt.


## Troubleshooting

To deactivate the environment, just type `conda deactivate`

Uninstall the jupyter kernel if you're having problems with it:
```
jupyter kernelspec uninstall coffeadev
```
and then reinstall it again
```
python -m ipykernel install --user --name=coffeadev
jupyter nbextension install --py widgetsnbextension --user
jupyter nbextension enable widgetsnbextension --user --py
```


If you already have a jupyter server running **on the uaf**, a different port than 8893 might be used. In this case, alter the `ssh -N -f ...` command so that it matches the ports. To stop a running jupyter server that is running but you can't find anymore, run `ps aux | grep $USER`. This will return you the list of processes attributed to your user. You should also find sth like
```
dspitzba 3964766  1.3  0.0  87556 44720 pts/17   S+   05:03   0:02 python /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw/CMSSW_10_2_9/external/slc6_amd64_gcc700/bin/jupyter-notebook --no-browser --port=8893
```
To stop this process, just type `kill 3964766`. In this case, 3964766 is the process id (PID) of the jupyter server process.

If a port is already used on your machine because of a not properly terminated ssh session, run the following command **on your computer** `ps aux | grep ssh`. This returns a similar list as before. There should be a job like
```
daniel           27709   0.0  0.0  4318008    604   ??  Ss    8:11AM   0:00.00 ssh -N -f -L localhost:8893:localhost:8893 uaf-10.t2.ucsd.edu
```
Similarly, you can stop the process by running `kill 27709`.


## Get combine (experts only)
Latest recommendations at https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#setting-up-the-environment-and-installation
```
cmsRel CMSSW_10_2_13
cd CMSSW_10_2_13/src
cmsenv
git cms-init
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v8.1.0
scramv1 b clean; scramv1 b # always make a clean build
```

## for combineTools (for later)
```
cd $CMSSW_BASE/src
wget https://raw.githubusercontent.com/cms-analysis/CombineHarvester/master/CombineTools/scripts/sparse-checkout-https.sh; source sparse-checkout-https.sh
scram b -j 8
```

## Sample version history

- v0p1p12: data processing (no MC submitted yet)
- v0p1p11: **dilep/trilep skim only!** added variables for ttH lepton ID
- v0p1p10: intermediate version
- v0p1p9: used for initial tW scattering studies presented in SnT
