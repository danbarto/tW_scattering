#code to write datacard from tableMaker.py output
#run with python writeCards.py
#Kaitlin Salyer, April 2021

#import some useful packages
import time
import numpy as np
import sys
import pandas as pd


def make_BDT_datacard(outdir, BDT_bins, yield_dict):
    return

#hardcoded variables other users should customize
outdir = "/home/users/cmcmahon/public_html/BDT/datacards/debug"
BDT_bins = np.linspace(0.4, 1.0, 6)
yield_dict = {
    "score_[0.4, 0.52]_signal": 1.,
    "score_[0.52, 0.64]_signal": 1.,
    "score_[0.64, 0.76]_signal": 1.,
    "score_[0.76, 0.88]_signal":1.,
    "score_[0.88, 1.0]_signal":1.,
    "score_[0.4, 0.52]_fakes": 1.,
    "score_[0.52, 0.64]_fakes": 1.,
    "score_[0.64, 0.76]_fakes": 1.,
    "score_[0.76, 0.88]_fakes":1.,
    "score_[0.88, 1.0]_fakes":1.,
    "score_[0.4, 0.52]_flips": 1.,
    "score_[0.52, 0.64]_flips": 1.,
    "score_[0.64, 0.76]_flips": 1.,
    "score_[0.76, 0.88]_flips":1.,
    "score_[0.88, 1.0]_flips":1.,
    "score_[0.4, 0.52]_rares": 1.,
    "score_[0.52, 0.64]_rares": 1.,
    "score_[0.64, 0.76]_rares": 1.,
    "score_[0.76, 0.88]_rares":1.,
    "score_[0.88, 1.0]_rares":1.,
    "score_[0.4, 0.52]_signal_error": 1.,
    "score_[0.52, 0.64]_signal_error": 1.,
    "score_[0.64, 0.76]_signal_error": 1.,
    "score_[0.76, 0.88]_signal_error":1.,
    "score_[0.88, 1.0]_signal_error":1.,
    "score_[0.4, 0.52]_fakes_error": 1.,
    "score_[0.52, 0.64]_fakes_error": 1.,
    "score_[0.64, 0.76]_fakes_error": 1.,
    "score_[0.76, 0.88]_fakes_error":1.,
    "score_[0.88, 1.0]_fakes_error":1.,
    "score_[0.4, 0.52]_flips_error": 1.,
    "score_[0.52, 0.64]_flips_error": 1.,
    "score_[0.64, 0.76]_flips_error": 1.,
    "score_[0.76, 0.88]_flips_error":1.,
    "score_[0.88, 1.0]_flips_error":1.,
    "score_[0.4, 0.52]_rares_error": 1.,
    "score_[0.52, 0.64]_rares_error": 1.,
    "score_[0.64, 0.76]_rares_error": 1.,
    "score_[0.76, 0.88]_rares_error":1.,
    "score_[0.88, 1.0]_rares_error":1.,
    "score_[0.4, 0.52]_Total_Background": 1.,
    "score_[0.52, 0.64]_Total_Background": 1.,
    "score_[0.64, 0.76]_Total_Background": 1.,
    "score_[0.76, 0.88]_Total_Background":1.,
    "score_[0.88, 1.0]_Total_Background":1.
    
}
#for b in range(1, len(BDT_bins)):
BDT_bin_names = ["[{0}, {1}]".format(BDT_bins[b-1], BDT_bins[b]) for b in range(1, len(BDT_bins))]
#first, we load the txt output from the tableMaker.py script into a dataframe
#we will manipulate these data, save it into a different dataframe, and print to an output file
#df = pd.read_csv("/home/users/ksalyer/FranksFCNC/ana/analysis/outputs/tables/tableMaker_"+str(y)+".txt")
print("got yields and stat errors")

#now we have imported the data and manipulated it into the categories we want
#we will do the rest in a loop over signals
signal = "signal_tuh_tch"
year = 2018
outfileName = "datacard_{0}_{1}.txt".format(signal, year)

#signal region bins
# nLeps = [2, 3]
# nJets = [2,3,4]
# nBtags = [0,1,2]
numBins = len(BDT_bin_names)
nProc = ["signal", "rares", "fakes", "flips"]
numBackgrounds = len(nProc)-1

#make some headers for my dataframe columns
dcColumns = []
binNames = []
binNamesObs = []
procNames = []
procIndex = []

for bdt_bin in BDT_bin_names:
    counter = 0
    binName = "score_{0}".format(bdt_bin).ljust(20)
    binNamesObs.append(binName)
    for p in nProc:
        srName = "score_{0}_{1}".format(bdt_bin, p).ljust(20)
        binNames.append(binName)
        dcColumns.append(srName)
        procNames.append(p.ljust(20))
        procIndex.append(str(counter).ljust(20))
        counter+=1

# ok, now I have headers, I can start making the titles for my rows
rowTitles = []
numParameters = 0
for p in nProc:
    for iterator in range(numBins):
        numParameters+=1
        title = "{0}_stat_{1}".format(p, iterator).ljust(17) + "lnN"
        rowTitles.append(title)

for p in nProc:
    title = "{0}_syst".format(p).ljust(17) + "lnN"
    rowTitles.append(title)
    numParameters+=1


#dataframe I will print to datacard file
dcard_df = pd.DataFrame(index = rowTitles, columns = dcColumns)
print("defined output dataframe")
#now I want to know the stat uncertainty as a percentage of the yield
for p in nProc:
    for i in range(len(BDT_bin_names)):
        bdt_bin = BDT_bin_names[i]
        err = yield_dict["score_{0}_{1}_error".format(bdt_bin, p)]
        unc = err / (round(yield_dict["score_{0}_{1}".format(bdt_bin, p)], 3) + 1)
        cTitle = "score_{0}_{1}".format(bdt_bin, p).ljust(20)
        rTitle = "{0}_stat_{1}".format(p, i).ljust(17) + "lnN"
        rTitle = p+"_stat_"+str(i)

        for column in yield_dict.keys():
            if column==cTitle:
                filler = str(unc).ljust(20)
            else:
                filler = "-".ljust(20)
            dcard_df.at[rTitle,column] = filler
            
print("filled stat uncertainties")
print(dcard_df)
outfile = open(outdir+outfileName,"w")
outfile.write(dcard_df.to_csv(sep="\t", index=True, header=False))
outfile.close()
#get MC yields in correct order of bins/processes
rates = []
observation = []
for bdt_bin in BDT_bin_names:
    row = bdt_bin
    obsYld = yield_dict["score_{0}_Total_Background".format(bdt_bin)]
    obsYld = round(obsYld,0)
    obsString = str(obsYld).ljust(20)
    observation.append(obsString)

    for p in nProc:
        if p == "signal":
            p = s
            yld = row[p].values[0]
        else:
            yld = row[p].values[0]

        if yld<0:
            yld = 0.01

        yldString = str(yld)
        while len(yldString)<20:
            yldString+=" "
        rates.append(yldString)



#filling dummy systematic uncertainties
for p in nProc:
    if "signal" in p:
        unc = "0.8/1.2"
    elif p == "flips":
        unc = "0.8/1.2"
    elif p == "other":
        unc = "0.7/1.3"
    elif p == "fakes":
        unc = "0.6/1.4"

    while len(unc)<20:
        unc+=" "
    unc = str(yield_dict[p + "_error"]).ljust(20)
    rTitle = "{0}_syst".format(p).ljust(17) + "lnN"
    for column in dcard_df:
        if p in column:
            filler = unc.ljust(20)
        else:
            filler = "-".ljust(20)
        dcard_df.at[rTitle,column] = filler
print("filled syst uncertainties")


#define output file and write to output file
outfile = open(outdir+outfileName,"w")
binHeadersObs = "bin                 \t"
for b in binNamesObs:
    binHeadersObs+=b
    binHeadersObs+="\t"
binHeadersObs+="\n"
binHeaders = "bin                 \t"
for b in binNames:
    binHeaders+=b
    binHeaders+="\t"
binHeaders+="\n"
procHeaders = "process             \t"
for p in procNames:
    procHeaders+=p
    procHeaders+="\t"
procHeaders+="\n"
pInxHeaders = "process             \t"
for i in procIndex:
    pInxHeaders+=i
    pInxHeaders+="\t"
pInxHeaders+="\n"
rateHeaders = "rate                \t"
for r in rates:
    rateHeaders+=r
    rateHeaders+="\t"
rateHeaders+="\n"
obsHeaders = "observation         \t"
for o in observation:
    obsHeaders+=o
    obsHeaders+="\t"
obsHeaders+="\n"
imaxHeader = "imax "+str(numBins)+" number of channels\n"
jmaxHeader = "jmax "+str(numBackgrounds)+" number of backgrounds\n"
kmaxHeader = "kmax "+str(numParameters)+" number of nuisance parameters\n"
outfile.write(imaxHeader)
outfile.write(jmaxHeader)
outfile.write(kmaxHeader)
outfile.write("shape * * FAKE\n\n")
outfile.write(binHeadersObs)
outfile.write(obsHeaders)
outfile.write("\n")
outfile.write(binHeaders)
outfile.write(procHeaders)
outfile.write(pInxHeaders)
outfile.write(rateHeaders)
outfile.write(dcard_df.to_csv(sep="\t", index=True, header=False))
outfile.close()
