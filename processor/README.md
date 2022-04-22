# Running the analysis

Histograms and caches can be filled like this.

```
ipython -i forward_jet.py -- --year 2017 --sample topW --central
```
where `--central` will only fill the central value (no systematic variations).
Running without this option will take longer, and potentially not work on a dask cluster because of memory issues.
In order to run on a dask cluster, add the `--dask` option.

## Actual analysis

FIXME: The commands below need some debugging after coffea updates.

The following command evaluates the DNN
```
ipython -i SS_analysis.py -- --year 2017 --sample topW --evaluate --training v1
```

Running the SM fits (in the analysis directory) like
```
ipython -i analysis.py -- --years 2017
```

Full Run 2:

```
ipython -i analysis.py -- --years 2016,2016APV,2017,2018
```
