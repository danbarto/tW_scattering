import uproot
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd, redirector_fnal
from Tools.nano_mapping import make_fileset, nano_mapping


samples = get_samples()

fileset = make_fileset(['QCD'], samples, redirector=redirector_ucsd, small=False, year='UL2018')

good = []
bad = []

for sample in list(fileset.keys()):
    for f_in in fileset[sample]:
        print (f_in)
        try:
            tree = uproot.open(f_in)["Events"]
            good.append(f_in)
        except OSError:
            print ("XRootD Error")
            bad.append(f_in)
        
