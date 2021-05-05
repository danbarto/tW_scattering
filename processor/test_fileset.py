import uproot
from Tools.helpers import get_samples
from Tools.config_helpers import redirector_ucsd, redirector_fnal
from Tools.nano_mapping import make_fileset, nano_mapping


samples = get_samples()

fileset = make_fileset(['top'], samples, redirector=redirector_ucsd, small=False)

good = []
bad = []

for f_in in fileset[list(fileset.keys())[0]]:
    print (f_in)
    try:
        tree = uproot.open(f_in)["Events"]
        good.append(f_in)
    except OSError:
        print ("XRootD Error")
        bad.append(f_in)
        
