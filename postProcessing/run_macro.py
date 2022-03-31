#!/usr/bin/env python3
import sys
import ROOT

ROOT.gROOT.ProcessLineSync('.L counter_macro.C+')

f_in = sys.argv[1]
f_out = sys.argv[2]

redirectors = [
    'root://xcache-redirector.t2.ucsd.edu:2042/',
    'root://cmsxrootd.fnal.gov/',
]

if f_in.count('ceph'):
    x = ROOT.counter_macro(f_in, f_out)
else:
    for red in redirectors:
        try:
            x = ROOT.counter_macro(red+f_in, f_out)
            break
        except:
            print ("Redirector failed, trying next one", red)

        #print (x)
