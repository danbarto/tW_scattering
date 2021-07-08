import os.path

def make_yield_dict(filename):
    file=open(filename, 'r')
    bin_names = []
    rates = []
    processes = []
    for line in file.readlines():
        if "bin" in line:
            if "#" not in line:
                bin_names.append(line.split()[1:])
        elif "rate" in line:
            rates.append(line.split()[1:])
        elif "process" in line:
            processes.append(line.split()[1:])
    file.close()
    bin_names = bin_names[(len(bin_names)-1)]
    rates = [float(rates[0][i]) for i in range(len(rates[0]))]
    processes = processes[0]
    yield_labels = [bin_names[i] + "_" + processes[i] for i in range(len(processes))]
    yield_dict = dict(zip(yield_labels, rates))
    for key in yield_labels:
        if "trilep" in str(key):
            yield_dict.pop(key)
    return yield_dict



def gen_datacard_ratios():
    hut = os.path.expandvars("$TWHOME/postProcessing/datacard_comparison/datacards/datacard_HUT_2018.txt")
    hct = os.path.expandvars("$TWHOME/postProcessing/datacard_comparison/datacards/datacard_HCT_2018.txt")
    kaitlin_hut = os.path.expandvars("$TWHOME/postProcessing/datacard_comparison/datacards/Kaitlin_HUT_2018.txt")
    kaitlin_hct = os.path.expandvars("$TWHOME/postProcessing/datacard_comparison/datacards/Kaitlin_HCT_2018.txt")

    hut_yd = make_yield_dict(hut)
    hct_yd = make_yield_dict(hct)
    k_hut_yd = make_yield_dict(kaitlin_hut)
    k_hct_yd = make_yield_dict(kaitlin_hct)
    
    hut= {"signal":0., "fakes":0., "flips":0., "rares":0.}
    k_hut = {"signal":0., "fakes":0., "flips":0., "rares":0.}
    hct = {"signal":0., "fakes":0., "flips":0., "rares":0.}
    k_hct = {"signal":0., "fakes":0., "flips":0., "rares":0.}
    
    for k in hut_yd.keys():
        if "signal" in str(k):
            hut["signal"] += hut_yd[k]
        elif "fakes" in str(k):
            hut["fakes"] += hut_yd[k]
        elif "flips" in str(k):
            hut["flips"] += hut_yd[k]
        elif "rares" in str(k):
            hut["rares"] += hut_yd[k]
    for k in hct_yd.keys():
        if "signal" in str(k):
            hct["signal"] += hct_yd[k]
        elif "fakes" in str(k):
            hct["fakes"] += hct_yd[k]
        elif "flips" in str(k):
            hct["flips"] += hct_yd[k]
        elif "rares" in str(k):
            hct["rares"] += hct_yd[k]
    for k in k_hut_yd.keys():
        if "signal" in str(k):
            k_hut["signal"] += k_hut_yd[k]
        elif "fakes" in str(k):
            k_hut["fakes"] += k_hut_yd[k]
        elif "flips" in str(k):
            k_hut["flips"] += k_hut_yd[k]
        elif "rares" in str(k):
            k_hut["rares"] += k_hut_yd[k]
    for k in k_hct_yd.keys():
        if "signal" in str(k):
            k_hct["signal"] += k_hct_yd[k]
        elif "fakes" in str(k):
            k_hct["fakes"] += k_hct_yd[k] 
        elif "flips" in str(k):
            k_hct["flips"] += k_hct_yd[k]  
        elif "rares" in str(k):
            k_hct["rares"] += k_hct_yd[k]  
    ratio_dict = {"HCT":{}, "HUT":{}}
    for k in hct.keys():
        #print(k)
#         print("my HCT {}:{}".format(str(k), hct[k]))
#         print("Kaitlin's HCT {}:{}".format(str(k), k_hct[k]))
#         print("my HCT {0} yield / Kaitlin's HCT {0} yield = {1}\n".format(str(k), hct[k] / k_hct[k]))
        ratio_dict["HCT"][k] = (hct[k] / k_hct[k])
    for k in hut.keys():
#         print(k)
#         print("my HUT {}:{}".format(str(k), hut[k]))
#         print("Kaitlin's HUT {}:{}".format(str(k), k_hut[k]))
#         print("my HUT {0} yield / Kaitlin's HUT {0} yield = {1}\n".format(str(k), hut[k] / k_hut[k]))
        ratio_dict["HUT"][k] = (hct[k] / k_hct[k])
    return ratio_dict
