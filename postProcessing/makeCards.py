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
outdir = "/home/users/cmcmahon/public_html/BDT/"
BDT_bins = np.linspace(0.4, 1.0, 6)
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
nProc = ["signal", "rares", "fakes_mc", "flips_mc"]
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
        srName = srName_base+"_"+p
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


#ok, now I want to know the stat uncertainty as a percentage of the yield
for p in nProc:
#     if p == "signal":
#         proc = s
#     else:
#         proc = p
    statUnc = []
    for bdt_bin in BDT_bin_names:
        #calculate signal percentage for statistical unc.
        row = df.loc[(df["nBtags"]==b)]
        yld = row[proc].values[0]
        err = row[proc+" error"].values[0]

        if yld != 0:
            dcPercentage = round(err/yld,3)
        else:
            dcPercentage = 1

        statUnc.append([l,j,b,1+dcPercentage])
    #print(statUnc)
    for i in range(len(statUnc)):
        lep = statUnc[i][0]
        jet = statUnc[i][1]
        btag = statUnc[i][2]
        unc = statUnc[i][3]

        cTitle = str(lep)+"_"+str(jet)+"_"+str(btag)+"_"+p
        rTitle = p+"_stat_"+str(i)
        while len(cTitle) < 20:
            cTitle+=" "
        while len(rTitle) < 17:
            rTitle+=" "
        rTitle+="lnN"

        #print(cTitle)
        #print(rTitle)
        #print(unc)


        for column in dcard_df:
            if column==cTitle:
                filler = str(unc)
                while len(filler)<20:
                    filler+=" "
                dcard_df.at[rTitle,column] = filler
            else:
                filler = "-"
                while len(filler) < 20:
                    filler += " "
                dcard_df.at[rTitle,column] = filler
print("filled stat uncertainties")
#print(dcard_df)

#get MC yields in correct order of bins/processes
rates = []
observation = []
for l in nLeps:
    for j in nJets:
        for b in nBtags:
            row = df.loc[ (df["nLeptons"]==l) & (df["nJets"]==j) & (df["nBtags"]==b) ]
            obsYld = row["Total Background"].values[0]
            obsYld = round(obsYld,0)
            obsString = str(obsYld)
            while len(obsString)<20:
                obsString+=" "
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

    rTitle = p+"_syst"
    while len(rTitle)<17:
        rTitle+=" "
    rTitle+="lnN"

    for column in dcard_df:
        if p in column:
            filler = unc
            while len(filler)<20:
                filler+=" "
            dcard_df.at[rTitle,column] = filler
        else:
            filler = "-"
            while len(filler) < 20:
                filler += " "
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