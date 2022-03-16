#!/usr/bin/env python3
import sys
import ROOT

ROOT.gROOT.ProcessLineSync('.L counter_macro.C+')

f_in = sys.argv[1]
f_out = sys.argv[2]

x = ROOT.counter_macro(f_in, f_out)

print (x)
