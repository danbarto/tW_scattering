#!/usr/bin/env bash

### those are not minimal so we should have full systematics for all distributions??

#python run_trilep.py --sample data --year 2016APV --select_systematic central --dask --rerun;
#python run_trilep.py --sample data --year 2016APV --select_systematic fake --dask --rerun;
#
#python run_trilep.py --sample MCall --year 2016APV --select_systematic central --dask --rerun;
python run_trilep.py --sample MCall --year 2016APV --select_systematic base --dask --rerun;
python run_trilep.py --sample MCall --year 2016APV --select_systematic jes --dask --rerun;
#
#python run_trilep.py --sample topW_lep --year 2016APV --select_systematic central --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2016APV --select_systematic base --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2016APV --select_systematic jes --dask --rerun --scan;


#python run_trilep.py --sample data --year 2016 --select_systematic central --dask --rerun;
#python run_trilep.py --sample data --year 2016 --select_systematic fake --dask --rerun;

#python run_trilep.py --sample MCall --year 2016 --select_systematic central --dask --rerun;
#python run_trilep.py --sample MCall --year 2016 --select_systematic base --dask --rerun;
#python run_trilep.py --sample MCall --year 2016 --select_systematic jes --dask --rerun;

#python run_trilep.py --sample topW_lep --year 2016 --select_systematic central --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2016 --select_systematic base --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2016 --select_systematic jes --dask --rerun --scan;


#python run_trilep.py --sample data --year 2017 --select_systematic central --dask --rerun;
#python run_trilep.py --sample data --year 2017 --select_systematic fake --dask --rerun;

#python run_trilep.py --sample MCall --year 2017 --select_systematic central --dask --rerun;
#python run_trilep.py --sample MCall --year 2017 --select_systematic base --dask --rerun;
#python run_trilep.py --sample MCall --year 2017 --select_systematic jes --dask --rerun;

#python run_trilep.py --sample topW_lep --year 2017 --select_systematic central --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2017 --select_systematic base --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2017 --select_systematic jes --dask --rerun --scan;


#python run_trilep.py --sample data --year 2018 --select_systematic central --dask --rerun;
#python run_trilep.py --sample data --year 2018 --select_systematic fake --dask --rerun;

#python run_trilep.py --sample MCall --year 2018 --select_systematic central --dask --rerun;
#python run_trilep.py --sample MCall --year 2018 --select_systematic base --dask --rerun;
#python run_trilep.py --sample MCall --year 2018 --select_systematic jes --dask --rerun;

#python run_trilep.py --sample topW_lep --year 2018 --select_systematic central --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2018 --select_systematic base --dask --rerun --scan;
#python run_trilep.py --sample topW_lep --year 2018 --select_systematic jes --dask --rerun --scan;
