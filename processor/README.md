# Running the analysis

Histograms and caches will be filled with

```
ipython -i trilep_analysis.py -- --year 2017 --evaluate --training trilep_v3
```

```
ipython -i SS_analysis.py -- --year 2017 --evaluate --training v21
```
A quick test can be run by adding the `--verysmall` flag.


Running the SM fits (in the analysis directory) like
```
ipython -i analysis.py -- --years 2017
```

Full Run 2:

```
ipython -i analysis.py -- --years 2016,2016APV,2017,2018
```
