import os
import sys
import argparse

def submitCondorJob(proc, executable, options, infile, label, outputToTransfer=None, submit=False, proxy=os.environ["X509_USER_PROXY"], longJob=False):
    hostname = os.uname()[1]
    logDir = os.path.join("logs",proc)
    subfile = "condor_"+proc +"_"+label+".cmd"
    f = open(subfile,"w")
    f.write("Universe = vanilla\n")
    if hostname.count('ucsd'):
#      f.write("Grid_Resource = condor cmssubmit-r1.t2.ucsd.edu glidein-collector.t2.ucsd.edu\n")
      f.write("x509userproxy={0}\n".format(proxy))
      f.write("+DESIRED_Sites=\"T2_US_UCSD\"\n")
#    f.write("request_cpus=8\n")
    if longJob:
        f.write("request_memory=4200MB\n")
    if hostname.count('lxplus'):
      if longJob:
        f.write('+JobFlavour = "nextweek"\n')
      else:
        f.write('+JobFlavour = "tomorrow"\n') #longlunch (2h) too short in some cases. tomorrow (1 day), testmatch (3 days), nextweek (1 week) also exist
      f.write('requirements = (OpSysAndVer=?= "SLCern6")\n')
    f.write("Executable = "+executable+"\n")
    f.write("arguments =  "+(' '.join(options))+"\n")
    f.write("Transfer_Executable = True\n")
    f.write("should_transfer_files = YES\n")
    f.write("transfer_input_files = "+infile+"\n")
    if outputToTransfer is not None:
        f.write("transfer_Output_files = "+outputToTransfer+"\n")
        f.write("WhenToTransferOutput  = ON_EXIT\n")
    f.write("Notification = Never\n")
    f.write("Log=%s/gen_%s_%s.log.$(Cluster).$(Process)\n"%(logDir, proc, label))
    f.write("output=%s/gen_%s_%s.out.$(Cluster).$(Process)\n"%(logDir, proc, label))
    f.write("error=%s/gen_%s_%s.err.$(Cluster).$(Process)\n"%(logDir, proc, label))
    f.write("queue 1\n")
    f.close()

    cmd = "condor_submit "+subfile
    print cmd
    if submit:
        os.system(cmd)
