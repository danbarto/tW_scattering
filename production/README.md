
# Instructions for usage on UCSD T2

The following scripts allow to produce events of tW scattering.
Both miniAOD and nanoAOD files are produced, and are stored under
```
/ceph/cms/store/user/YOUR_USER/ProjectMetis
```
As an example, see
```
/ceph/cms/store/user/dspitzba/ProjectMetis/
```


In order to start, get your proxy and export it
```
voms-proxy-init -voms cms -valid 100:00 -out /tmp/x509up_YOUR_USER; export X509_USER_PROXY=/tmp/x509up_YOUR_USER
```

## Run 2 era names - this is confusing

- 2016APV = preVFP = HIPM, B-F
- 2016 = postVFP, F-H
- 2017
- 2018

## Local runs

Setup your CMSSW area, e.g. CMSSW_10_6_19 for NanoGEN

``` shell
cmsrel CMSSW_10_6_19
```

Then

``` shell
cd CMSSW_10_6_19/src
cmsenv
wget https://raw.githubusercontent.com/syjeon162/tW_scattering/master/production/psets/nanogen/nanogen_nlo_cfg.py
curl -s -k https://raw.githubusercontent.com/syjeon162/tW_scattering/master/production/psets/tW_scattering.py --retry 3 --create-dirs -o Configuration/GenProduction/python/tW_scattering.py
```

Fix the path to the gridpack, then

``` shell
cmsRun nanogen_nlo_cfg.py
```


## Production using metis

Fripdacks are here:
```
/ceph/cms/store/user/dspitzba/gridpacks/
```
where one is produced with SMEFTatNLO without any NP interactions: FIXME
```
TTWJetsToLNuEWK_5f_NLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz
```
and the other one with some coefficients set to 1:
```
TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz
```

The submitter script is `make_on_azure.py`.
We are currently submitting to UCSD only in order to have access to ceph (gridpacks are not part of the package tarball).


## Old instructions

For a sample with around 100k events do
```
python submitJobsToCondor.py tW_scattering --fragment tW_scattering.py --nevents 1000 --njobs 100 --executable makeSample.sh --rseed-start 500
```
