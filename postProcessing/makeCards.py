#code to write datacard from tableMaker.py output
#run with python writeCards.py
#Kaitlin Salyer, April 2021

#import some useful packages
import time
import numpy as np
import sys
import pandas as pd

#hardcoded variables other users should customize

def make_BDT_datacard(yield_dict, BDT_bins, signal, outdir, label="", year="all", systematics_yields=None, CRstats=None, JES_stats=None):

    BDT_bin_names = [str(b) for b in range(len(BDT_bins)-1)]
    BDT_bin_comments = ["[{0}, {1}]".format(BDT_bins[b-1], BDT_bins[b]) for b in range(1, len(BDT_bins))]
    #first, we load the txt output from the tableMaker.py script into a dataframe
    #we will manipulate these data, save it into a different dataframe, and print to an output file
    #df = pd.read_csv("/home/users/ksalyer/FranksFCNC/ana/analysis/outputs/tables/tableMaker_"+str(y)+".txt")
    #print("got yields and stat errors")

    #now we have imported the data and manipulated it into the categories we want
    #we will do the rest in a loop over signals
    outfileName = "datacard_{0}_{1}".format(signal, year)
    if label != "":
        outfileName += "_{0}".format(label)
    outfileName += ".txt"

    numBins = len(BDT_bin_names)
    nProc = ["signal", "rares", "fakes", "flips"]
    systematicSources = ["LepSF","PU","Trigger","bTag"]#,"jes"]
    numBackgrounds = len(nProc)-1

    #make some headers for my dataframe columns
    dcColumns = []
    binNames = []
    binNamesObs = []
    procNames = []
    procIndex = []

    for bdt_bin in BDT_bin_names:
        counter = 0
        binName = "bin_{0}".format(bdt_bin).ljust(20)
        binNamesObs.append(binName)
        for p in nProc:
            srName = "bin_{0}_{1}".format(bdt_bin, p).ljust(20)
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
            if p == "fakes":
                title = "fkStat{0}_{1}".format(str(year)[-2:], iterator)
                yld = str(systematics_yields[title])
                #yld = str(fakeCRStatYld[iterator]) #implement fakeCRstatyld
                title = title.ljust(16-len(yld)) + "gmN " + yld
                rowTitles.append(title)
            else:
                #numParameters+=1
                title = "{0}_stat_{1}".format(p, iterator).ljust(17) + "lnN"
                rowTitles.append(title)

    for p in nProc:
        title = "{0}_syst".format(p).ljust(17) + "lnN"
        rowTitles.append(title)
        numParameters+=1
        
    #breakpoint()

    #dataframe I will print to datacard file
    dcard_df = pd.DataFrame(index = rowTitles, columns = dcColumns)
    #print("defined output dataframe")
    #now I want to know the stat uncertainty as a percentage of the yield
    for p in nProc:
        numParameters += 1
        for i in range(len(BDT_bin_names)):
            bdt_bin = BDT_bin_names[i]
            err = yield_dict["bin_{0}_{1}_error".format(bdt_bin, p)]
            yld = yield_dict["bin_{0}_{1}".format(bdt_bin, p)]
            #numParameters += 1
            if p != "fakes":
                if abs(yld) <= 0.01:
                    unc = 1.0
                else:
                    unc = (err / round(yld, 4)) + 1.0
                rTitle = "{0}_stat_{1}".format(p, i).ljust(17) + "lnN"

            else: #fakes CR Statistics
                unc = round(err, 4)
                title = "fkStat{0}_{1}".format(str(year)[-2:], i)
                CR_yld = str(systematics_yields[title])
                #yld = str(fakeCRStatYld[iterator]) #implement fakeCRstatyld
                rTitle = title.ljust(16-len(CR_yld)) + "gmN " + CR_yld
                
            cTitle = "bin_{0}_{1}".format(bdt_bin, p)
            for column in yield_dict.keys():
                if column==cTitle:
                    filler = str(unc).ljust(20)
                else:
                    filler = "-".ljust(20)
                if ("error" not in column) and ("Total" not in column):
                    dcard_df.at[rTitle,column.ljust(20)] = filler

    #print("filled stat uncertainties")
    #get MC yields in correct order of bins/processes
    rates = []
    observation = []
    for bdt_bin in BDT_bin_names:
        row = bdt_bin
        obsYld = yield_dict["bin_{0}_Total_Background".format(bdt_bin)]
        obsYld = round(obsYld,0)
        obsString = str(obsYld).ljust(20)
        observation.append(obsString)

        for p in nProc:
            yldString = str(yield_dict["bin_{0}_{1}".format(bdt_bin, p)]).ljust(20)
            rates.append(yldString)

    #filling dummy systematic uncertainties
    for p in nProc:
        if "signal" in p:
            unc = "0.8/1.2"
        elif p == "flips":
#             if year == 2016: unc = '1.1'
#             if year == 2017: unc = '1.4'
#             if year == 2018: unc = '1.3'
            unc = "0.7/1.3"
        elif p == "other":
            unc = "0.7/1.3"
        elif p == "fakes":
            unc = "0.6/1.4"
        #unc = str(yield_dict[p + "_error"]).ljust(20)
        rTitle = "{0}_syst".format(p).ljust(17) + "lnN"
        for column in yield_dict.keys():
            if p in column:
                filler = unc.ljust(20)
            else:
                filler = "-".ljust(20)
            if ("error" not in column) and ("Total" not in column):
                dcard_df.at[rTitle,column.ljust(20)] = filler
    #begin systematic uncertainties               
    fakeRateIter = 0
    numParameters += 1
    for column in dcard_df:
        rTitle = "fakeRate_syst".ljust(17) + "lnN"
        if "fakes" in column:
            #filler = str(1+fakeSystErr[fakeRateIter]/100)
            filler = str(1.09).ljust(20)
            fakeRateIter+=1
        else:
            filler = "-".ljust(20)
        dcard_df.at[rTitle,column]=filler

    for source in systematicSources:
        #systematics here are just the ratio of the "up" yield to the central yield, and the "down" yield to the central yield
        rTitle = "{}_{}".format(source, year).ljust(17) + "lnN"
        for column in dcard_df:
            if ("rares" in column) or ("signal" in column):
                #row = raresSyst_df.loc[(raresSyst_df["nLeptons"]==l) & (raresSyst_df["nJets"]==j) & (raresSyst_df["nBtags"]==b)]
                #up_yld = systematic_yields["bin_{0}_rares_{}_up".format(bdt_bin, source)]
                colTitleUp = "{}_{}_up".format(column.strip(), source)
                colTitleDown = "{}_{}_down".format(column.strip(), source)
                up_yld = systematics_yields[colTitleUp]
                down_yld = systematics_yields[colTitleDown]
                central_yld = yield_dict[column.strip()]
                filler = "{0:.3f}/{1:.3f}".format((down_yld/central_yld), (up_yld/central_yld)).ljust(20)
                #filler = str(round(down_yld/central_yld,3))+"/"+str(round(up_yld/central_yld,3))
#                     while len(filler)<20:
#                         filler+=" "

#             elif "signal" in column:
# #                 row = signalSyst_df.loc[(signalSyst_df["nLeptons"]==l) & (signalSyst_df["nJets"]==j) & (signalSyst_df["nBtags"]==b)]
# #                 colTitleUp = "{}_{}_up".format(s, source)
# #                 colTitleDown = "{}_{}_down".format(s, source)
#                 breakpoint()
#                 colTitleUp = "{}_{}_up".format(column, source)
#                 colTitleDown = "{}_{}_down".format(column, source)
#                 up_yld = systematics_yields["bin_{0}_{1}".format(bdt_bin, colTitleUp)]
#                 down_yld = systematics_yields["bin_{0}_{1}".format(bdt_bin, colTitleDown)]
#                 central_yld = yield_dict["bin_{0}_{1}".format(bdt_bin, column)]
#                 filler = "{0:.3f}/{1:.3f}".format((down_yld/central_yld), (up_yld/central_yld)).ljust(20)
#                 filler = str(round(row[colTitleDown],3))+"/"+str(round(row[colTitleUp],3))
#                 while len(filler)<20:
#                     filler+=" "

            else:
                filler = "-".ljust(20)
            dcard_df.at[rTitle,column]=filler
    #end systematic uncertainties
    #print("filled syst uncertainties")
    # outfile = open(outdir+outfileName,"w")
    # outfile.write(dcard_df.to_csv(sep="\t", index=True, header=False))
    # outfile.close()
    #define output file and write to output file
    outfile = open(outdir+outfileName,"w")
    #breakpoint()
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
    #numParameters = len(rowTitles)
    kmaxHeader = "kmax "+str(numParameters)+" number of nuisance parameters\n"
    comments = "#{} signal region data card\n#".format(signal)
    for b in range(len(BDT_bin_names)): 
        comments += "bin_{0}={1}\t".format(BDT_bin_names[b], BDT_bin_comments[b])
    comments += "\n"
    outfile.write(comments)
    outfile.write(imaxHeader)
    outfile.write(jmaxHeader)
    outfile.write(kmaxHeader)
    outfile.write("shapes * * FAKE\n\n")
    outfile.write(binHeadersObs)
    outfile.write(obsHeaders)
    outfile.write("\n")
    outfile.write(binHeaders)
    outfile.write(procHeaders)
    outfile.write(pInxHeaders)
    outfile.write(rateHeaders)
    outfile.write(dcard_df.to_csv(sep="\t", index=True, header=False))
    outfile.close()

flag_debug = False
if flag_debug:
    outdir = "/home/users/cmcmahon/public_html/BDT/datacards/debug/"
    BDT_bins = np.linspace(0.4, 1.0, 6)
    yield_dict = {
        "bin_0_signal": 1.,
        "bin_1_signal": 1.,
        "bin_2_signal": 1.,
        "bin_3_signal":1.,
        "bin_4_signal":1.,
        "bin_0_fakes": 1.,
        "bin_1_fakes": 1.,
        "bin_2_fakes": 1.,
        "bin_3_fakes":1.,
        "bin_4_fakes":1.,
        "bin_0_flips": 1.,
        "bin_1_flips": 1.,
        "bin_2_flips": 1.,
        "bin_3_flips":1.,
        "bin_4_flips":1.,
        "bin_0_rares": 1.,
        "bin_1_rares": 1.,
        "bin_2_rares": 1.,
        "bin_3_rares":1.,
        "bin_4_rares":1.,
        "bin_0_signal_error": 1.,
        "bin_1_signal_error": 1.,
        "bin_2_signal_error": 1.,
        "bin_3_signal_error":1.,
        "bin_4_signal_error":1.,
        "bin_0_fakes_error": 1.,
        "bin_1_fakes_error": 1.,
        "bin_2_fakes_error": 1.,
        "bin_3_fakes_error":1.,
        "bin_4_fakes_error":1.,
        "bin_0_flips_error": 1.,
        "bin_1_flips_error": 1.,
        "bin_2_flips_error": 1.,
        "bin_3_flips_error":1.,
        "bin_4_flips_error":1.,
        "bin_0_rares_error": 1.,
        "bin_1_rares_error": 1.,
        "bin_2_rares_error": 1.,
        "bin_3_rares_error":1.,
        "bin_4_rares_error":1.,
        "bin_0_Total_Background": 1.,
        "bin_1_Total_Background": 1.,
        "bin_2_Total_Background": 1.,
        "bin_3_Total_Background":1.,
        "bin_4_Total_Background":1.

    }
    #for b in range(1, len(BDT_bins)):
    signal = "signal_tuh_tch"
    make_BDT_datacard(yield_dict, BDT_bins, signal, outdir)
    