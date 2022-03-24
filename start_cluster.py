'''


c.scheduler_info()['workers']['tcp://169.228.131.90:45413']

c.retire_workers(workers=['tcp://169.228.131.90:45413'])

'''


import os
import socket

from Tools.condor_utils import make_htcondor_cluster

from dask.distributed import Client, progress

def get_workers( client ):
    logs = client.get_worker_logs()
    return list(logs.keys())

def get_hosts( workers, hostname=True ):
    index = 0 if hostname else 2
    hosts = list(set([ socket.gethostbyaddr(w.replace('tcp://','').split(':')[0])[index] for w in workers ]))
    return hosts

def get_all_warnings( client ):
    logs = client.get_worker_logs()
    workers = get_workers( client )
    for worker in workers:
        for log in logs[worker]:
            if log[0] == 'WARNING' or log[0] == 'ERROR':
                print ()
                print (" ### Found warning for worker:", worker)
                print (log[1])

def get_files_not_found( client ):
    allFiles = []
    logs = client.get_worker_logs()
    workers = get_workers( client )
    for worker in workers:
        for log in logs[worker]:
            if log[0] == 'WARNING':
                print (worker)
                files = [ x for x in log[1].split() if x.count('xrootd') ]
                print ( files )
                allFiles += files

    return allFiles

if __name__ == "__main__":
    
    import time
    import argparse
    from tqdm import tqdm

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--local', action='store_true', default=None, help="Overwrite existing results??")
    argParser.add_argument('--scale', action='store', default=5, help="How many workers?")
    argParser.add_argument('--memory', action='store', default=4, help="How much memory?")
    args = argParser.parse_args()
    
    scale  = int(args.scale)
    memory = int(args.memory)
    
    #c = Client(memory_limit='4GB', n_workers=4, threads_per_worker=1)
    
    if args.local:
    
        c = Client(
            memory_limit='%sGB'%memory,
            n_workers=scale,
            threads_per_worker=1,
        )
    
        #c = Client("tcp://127.0.0.1:18124")
        with open('scheduler_address.txt', 'w') as f:
            f.write(str(c.cluster.scheduler_address))
    
    else:
        
        cluster = make_htcondor_cluster(
            local=False,
            dashboard_address=13349,
            disk = "8GB",
            memory = "%sGB"%memory,
        )
        
        print ("Scaling cluster at address %s now."%cluster.scheduler_address)
        
        cluster.scale(scale)
        
        with open('scheduler_address.txt', 'w') as f:
            f.write(str(cluster.scheduler_address))
        
        c = Client(cluster)

    notified = False
    running = []
    retired = []

    with tqdm(total=int(args.scale)) as pbar:
        while True:
            try:
                workers_tmp = get_workers(c)
            except OSError:
                print ("Error getting the workers")
            if len(workers_tmp) > 0.9*int(args.scale) and not notified:
                print ("Cluster ready.")
                notified = True

            delta = len(workers_tmp)-len(running)
            diff  = list(set(running) - set(workers_tmp))
            if delta<0:
                print ("Retired %s workers since I last checked.")
                for d in diff:
                    print (d)
                retired += diff

            running = workers_tmp

            pbar.update(delta)

            time.sleep(10)
            #break


