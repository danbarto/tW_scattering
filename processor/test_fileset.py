import uproot
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd, redirector_fnal
from Tools.nano_mapping import make_fileset, nano_mapping


samples = get_samples(f_in='samples_QCD.yaml')

fileset = make_fileset(['QCD'], samples, redirector=redirector_ucsd, small=False)
print(fileset)


good = []
bad = []
#breakpoint()
for n in range(len(list(fileset.keys()))):
    for f_in in fileset[list(fileset.keys())[n]]:
        print (f_in)
        try:
            tree = uproot.open(f_in)["Events"]
            good.append(f_in)
        except OSError:
            print ("XRootD Error")
            bad.append(f_in)
        

good_files = open("files/samples_QCD_good.txt", "w")
for sample in good:
    good_files.write(sample + "\n")
good_files.close()
bad_files = open("files/samples_QCD_bad.txt", "w")
for sample in bad:
    bad_files.write(sample + "\n")
bad_files.close()
