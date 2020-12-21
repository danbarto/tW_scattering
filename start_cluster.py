import os

from dask.distributed import Client
import distributed

from Tools.condor_utils import make_htcondor_cluster

from dask.distributed import Client, progress

def getWorkers( client ):
    logs = client.get_worker_logs()
    return list(logs.keys())

def getAllWarnings( client ):
    logs = client.get_worker_logs()
    workers = getWorkers( client )
    for worker in workers:
        for log in logs[worker]:
            if log[0] == 'WARNING' or log[0] == 'ERROR':
                print ()
                print (" ### Found warning for worker:", worker)
                print (log[1])

def getFilesNotFound( client ):
    allFiles = []
    logs = client.get_worker_logs()
    workers = getWorkers( client )
    for worker in workers:
        for log in logs[worker]:
            if log[0] == 'WARNING':
                print (worker)
                files = [ x for x in log[1].split() if x.count('xrootd') ]
                print ( files )
                allFiles += files

    return allFiles

cluster = make_htcondor_cluster(local=False, dashboard_address=13349, disk = "4GB", memory = "3GB",)

print ("Scaling cluster at address %s now."%cluster.scheduler_address)

cluster.scale(50)

with open('scheduler_address.txt', 'w') as f:
    f.write(str(cluster.scheduler_address))

c = Client(cluster)
