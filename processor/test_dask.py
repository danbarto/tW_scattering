try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np

from start_cluster import *
from Tools.condor_utils import blacklisted_machines

if __name__ == '__main__':


    overwrite = True
    year = 2018
    local = False
    small = True

    from Tools.helpers import get_scheduler_address
    from dask.distributed import Client, progress

    scheduler_address = get_scheduler_address()
    c = Client(scheduler_address)


    # just to test 
    def test(x):
        import yahist
        import coffea
        import awkward
        #import sklearn
        import onnxruntime
        from Tools.cutflow import Cutflow
        return x**2

    def get_coffea_version(x):
        import yahist
        import coffea
        import awkward
        #import sklearn
        import onnxruntime
        from Tools.cutflow import Cutflow
        return coffea.__version__

    print ("Local")
    for res in map(test, range(5)):
        print (res)

    import time
    import random
    futures = c.map(test, range(50))
    results = c.gather(futures)

    print ("DASK")
    print (results)

    futures = c.map(get_coffea_version, range(1))
    results = c.gather(futures)
    print (results)
