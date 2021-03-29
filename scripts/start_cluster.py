'''
Starts up a DASK cluster, dumps the scheduler address in a txt file to be picked up later.
run with ipython -i start_cluster.py, and then keep it in the background

'''

import os

from dask.distributed import Client, progress
import distributed

from Tools.condor_utils import make_htcondor_cluster

cluster = make_htcondor_cluster(local=False, dashboard_address=13349, disk = "5GB", memory = "5GB",)

cluster.scale(50)

print ("Scheduler address:", cluster.scheduler_address)

c = Client(cluster)

with open("../scheduler_address.txt", "w") as f:
     f.write(str(cluster.scheduler_address))

# cluster.close()

