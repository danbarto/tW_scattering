#!/usr/bin/env python3

import os
import re
import datetime
import subprocess

from Tools.helpers import get_samples

import concurrent.futures

redirector = 'root://redirector.t2.ucsd.edu:1095//'

local_dir = "/data/"

skim = 'topW_v0.7.1_trilep'

def xrdcp(source, target):
    cmd = ['xrdcp', '-f', source, target]
    print ("Running cmd: %s"%(" ".join(cmd)))
    #cmd = ['xrdcp', source, target]
    subprocess.call(cmd)
    return os.path.isfile(target)

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def make_copy(in_n_out):
    file_in, target = in_n_out
    print (file_in, " ---> ", target)
    success =  xrdcp(file_in, target)
    #os.system('xrdcp %s %s'%(f_in, out_dir))
    return success

if __name__ == '__main__':

    ul = "UL16"

    samples = get_samples("samples_%s.yaml"%ul)

    copy_list = []

    workers = 10

    for sample in samples.keys():

        print (sample)

        #all_files = [ f.replace('topW_v0.7.0_dilep', 'topW_v0.7.1_trilep') for f in  samples[sample]['files'] ]  # FIXME something is wrong with the skim names?
        all_files = [ f.replace('topW_v0.7.1_SS', 'topW_v0.7.1_trilep') for f in  samples[sample]['files'] ]

        for f in all_files:

            file_name = f.split('/')[-1]
            # extract the name and create the local path
            subdir = f.split(skim)[-1].split('nanoSkim')[-2]

            #print (subdir)
            out_dir = os.path.join(local_dir, skim, subdir.strip('/'))
            #print (out_dir)
            make_dir(out_dir)

            # replace /ceph/ with redirector
            f_in = f.replace('/ceph/cms/', redirector)

            # copy! NOTE: maybe multithread?
            #print (f_in)
            target = os.path.join(out_dir, file_name)
            if not os.path.isfile(target):
                copy_list.append((f_in, target))
#                os.system('xrdcp %s %s'%(f_in, out_dir))

    #print (copy_list)
    #raise NotImplementedError

    #copy_list = copy_list[:2]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for in_n_out, result in zip(copy_list, executor.map(make_copy, copy_list)):
            print ("Working on", in_n_out[0])

        #if sample.count('ceph'):
        #    sample_name = sample.split('/')[-2]
        #else:
        #    sample_name = sample.split('/')[-3]
        #print (sample_name)
