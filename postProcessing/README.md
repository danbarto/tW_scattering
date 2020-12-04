
# NanoAOD tools

If you ran the recipe correctly you should have a copy of NanoAOD-tools checked out in `$CMSSW_BASE/src/PhysicsTools/NanoAOD/`.

We use the following fork for producing babies: [danbarto/nanoAOD-tools/tree/tW_scattering](https://github.com/danbarto/nanoAOD-tools/tree/tW_scattering).
Each submission should be done with a unique tag for reproducibility.
Before tasks are created and submitted to condor the code checks if the repository and tag actually exist.
This requires a package that can be installed with `pip install PyGithub`.
In order to establish a connection to github, you need to [create credentials on github](https://github.com/settings/tokens).
Grant permissions to read repositories (repo).
Copy the credentials to a file `github_credentials.txt` in this directory. Never commit the file to github!
Once this is done, you're ready to go.

You can run the submitter like this:
```
python submitter.py --dryRun --tag 0p1p20 --user ksalyer --skim MET
```

Important files to look at are in python/postprocessing/modules/tW_scattering/:
- **GenAnalyzer** loops over the GenParticle collection and writes out the most important generated particles
- **ObjectSelection** has a (basic) object selection, and calculates some event based variables from them
- **lumiWeightProducer** calculates the luminosity weight
- **keep_and_drop** defines what branches to keep

Run the code locally:
- **scripts/run_processor.py** is your place to go within NanoAOD-tools

For local tests just run
```==
cd $CMSSW_BASE/src/
python PhysicsTools/NanoAODTools/scripts/run_processor.py INPUTFILENAMES LUMIWEIGHT ISDATA YEAR ERA ISFASTSIM
```
where INPUTFILENAMES is any NanoAOD file (list), and LUMIWEIGHT can be any float number and doesn't really matter for tests (it will only mess up your *weight* branch.
For tests you can use e.g.:
```
/hadoop/cms/store/user/dspitzba/tW_scattering/tW_scattering/nanoAOD/tW_scattering_nanoAOD_177.root
```
or
```
/store/data/Run2018C/DoubleMuon/NANOAOD/02Apr2020-v1/40000/1B7C6AAE-3CD0-8D4E-9176-3D646D4C04D3.root
```
An example would be the following:
```
python PhysicsTools/NanoAODTools/scripts/run_processor.py /store/data/Run2018C/DoubleMuon/NANOAOD/02Apr2020-v1/40000/1B7C6AAE-3CD0-8D4E-9176-3D646D4C04D3.root 1 1 2018 C 0
```
This processes one DoubleMuon 2018 data file from era C, which of course is data and is not FastSim. The lumiweight is set to 1, but doesn't matter.

:exclamation: NanoAOD-tools is python2, while the coffea environment is python3. Make sure to deactivate the environment with `deactivate` and do `cmsenv` again to ensure NanoAOD-tools to run properly.

## ToDo
- [ ] Implement year branch into babies
- [ ] Keep deepJet branch


## Condor submission

```
export PYTHONPATH=ProjectMetis
```

Start submission in screen session:
```
screen -S sbm
```
Then do
```
source setup.sh
```

Disconnect screen:
ctrl+A, ctrl+D
