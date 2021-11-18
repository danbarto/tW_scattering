import shutil
import os

quantile_dict = {
    '0.025': '-2sig',
    '0.16': '-1sig',
    '0.5': 'expected',
    '0.84': '+1sig',
    '0.975': '+2sig',
    '-1.0': 'observed'
}

class dataCard:
    def __init__(self, releaseLocation='.'):
        self.reset()
        self.releaseLocation = os.path.abspath(releaseLocation)

    def reset(self):
        self.bins = []
        self.muted = {}
        self.uncertainties = []
        self.uncertaintyVal = {}
        self.uncertaintyString = {}
        self.processes = {}
        self.expectation = {}
        self.observation = {}
        self.contamination = {}
        self.niceNames = {}
        self.defWidth = 12
        self.precision = 7
        self.maxUncNameWidth = 30
        self.maxUncStrWidth = 30
        self.hasContamination=False
        self.rateParameters = []
        self.freeParameters = []
        self.precision = 10

    
    def setPrecision(self, prec):
        self.precision = prec

    def addBin(self, name, processes, niceName=""):
        if len(name)>30:
            print("Name for bin %s too long. Max. length is 30."%name)
            return
        if name in self.niceNames:
            print("Bin already there: %s"%name)
            return
        for p in processes:
            if len(p)>30:
                print("Name for process %s in bin %s is too long. Max. length is 30."%(p, name))
                return
        self.niceNames[name]=niceName
        self.bins.append(name)
        self.muted[name]=False
        self.processes[name] = ["signal"]+processes

    def addUncertainty(self, name, t, n=0):
        assert len(name)<self.maxUncNameWidth,  "That's too long: %s. Max. length is %i"%(name, self.maxUncNameWidth)
        if self.uncertainties.count(name):
            print("Uncertainty already there! (",name,")")
            return
        self.uncertainties.append(name)
        self.uncertaintyString[name] = t
        if t=="gmN":
            if n==0:
                print("gmN Uncertainty with n=0! Specify n as third argument: addUncertainty(..., 'gmN', n)")
                return
            self.uncertaintyString[name] = t+' '+str(n)
        if len(self.uncertaintyString[name])>self.maxUncStrWidth:
            print("That's too long:",self.uncertaintyString[name],"Max. length is", self.maxUncStrWidth)
            del self.uncertaintyString[name]
            return
    
    def addRateParameter(self, p, value, r):
        if [ a[0] for a in self.rateParameters ].count(p):
            print("Rate parameter for process %s already added!"%p)
            return
        self.rateParameters.append((p, value, r))

    def addFreeParameter(self, name, p, value, r):
        if [ a[0] for a in self.freeParameters ].count(name):
            print("Free parameter for process %s already added!"%p)
            return
        self.freeParameters.append((name, p, value, r))

    def specifyExpectation(self, b, p, exp):
        self.expectation[(b,p)] = round(exp, self.precision)

    def specifyObservation(self, b, obs):
        #if not isinstance(obs, int):
        #    print("Observation not an integer! (",obs,")")
        #    return
        self.observation[b] = obs

    def specifyContamination(self, b, cont):
        self.contamination[b] = cont
        self.hasContamination = True

    def specifyFlatUncertainty(self, u,  val):
        if u not in self.uncertainties:
            print("This uncertainty has not been added yet!",u,"Available:",self.uncertainties)
            return
        print("Adding ",u,"=",val,"for all bins and processes!")
        for b in self.bins:
            for p in self.processes[b]:
                self.uncertaintyVal[(u,b,p)] = round(val,self.precision)

    def specifyUncertainty(self, u, b, p, val):
        if u not in self.uncertainties:
            print("This uncertainty has not been added yet!",u,"Available:",self.uncertainties)
            return
        if b not in self.bins:
            print("This bin has not been added yet!",b,"Available:",self.bins)
            return
        if p not in self.processes[b]:
            print("Process ", p," is not in bin",b,". Available for ", b,":",self.processes[b])
            return
        if val<0:
            print("Warning! Found negative uncertainty %f for yield %f in %r. Reversing sign under the assumption that the correlation pattern is irrelevant (check!)."%(val, self.expectation[(b, p)], (u,b,p)))
            _val=1.0
        else:
            _val = val
        self.uncertaintyVal[(u,b,p)] = round(_val,self.precision)

    def getUncertaintyString(self, k):
        u, b, p = k
        if self.uncertaintyString[u].count('gmN'):
            if (u,b,p) in self.uncertaintyVal and self.uncertaintyVal[(u,b,p)]>0.:
                n = float(self.uncertaintyString[u].split(" ")[1])
                return self.mfs(self.expectation[(b, p)]/float(n))
            else: return '-'
        if (u,b,p) in self.uncertaintyVal:
            return self.mfs(self.uncertaintyVal[(u,b,p)])
        return '-'

    def checkCompleteness(self):
        for b in self.bins:
            if b not in self.observation or not self.observation[b]<float('inf'):
                print("No valid observation for bin",b)
                return False
            if self.hasContamination and (b not in self.contamination or not self.contamination[b] < float('inf')):
                print("No valid contamination for bin",b)
                return False
            if len(self.processes[b])==0:
                print("Warning, bin",b,"has no processes")
            for p in self.processes[b]:
                if (b,p) not in self.expectation or not self.expectation[(b,p)]<float('inf'):
                    print("No valid expectation for bin/process ",(b,p))
                    return False
            for k in list(self.uncertaintyVal.keys()):
                if not self.uncertaintyVal[k]<float('inf'):
                    print("Uncertainty invalid for",k,':',self.uncertaintyVal[k])
                    return False
        return True

    def mfs(self, f):
        return str(round(float(f),self.precision))

    def writeToFile(self, fname, shapeFile=False, noMCStat=False):
        import datetime, os
        if not self.checkCompleteness():
            print("Incomplete specification.")
            return
        allProcesses=[]
        numberID = {}
        i=1
        for b in self.bins:
            if not self.muted[b]:
                for p in self.processes[b]:
                    if not p in allProcesses and not p=='signal':
                        allProcesses.append(p)
                        numberID[p] = i
                        i+=1
        unmutedBins = [b for b in self.bins if not self.muted[b]]
        nBins = len(unmutedBins)
        numberID['signal'] = 0
        lspace = (self.maxUncStrWidth + self.maxUncNameWidth + 2)
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        
        with open(fname, 'w') as outfile:
            #outfile = file(fname, 'w')
            outfile.write('#cardFileWriter, %s'%datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")+'\n')
            outfile.write('imax %i'%nBins+'\n')
            outfile.write('jmax *\n')
            outfile.write('kmax *\n')
            outfile.write('\n')

            for b in self.bins:
                if not self.muted[b]:
                    outfile.write( '# '+b+': '+self.niceNames[b]+'\n')
                else:
                    outfile.write( '#Muted: '+b+': '+self.niceNames[b]+'\n')
            outfile.write( '\n')

            if shapeFile:
                outfile.write( 'shapes * * %s  $PROCESS $PROCESS_$SYSTEMATIC \n'%shapeFile)
                outfile.write( '\n')

            outfile.write( 'bin'.ljust(lspace)              +(' '.join([b.rjust(self.defWidth) for b in unmutedBins] ) ) +'\n')
            outfile.write( 'observation'.ljust(lspace)      +(' '.join([str(self.observation[b]).rjust(self.defWidth) for b in unmutedBins]) )+'\n')
            if self.hasContamination:
                outfile.write( 'contamination'.ljust(lspace)  +(' '.join([str(self.contamination[b]).rjust(self.defWidth) for b in unmutedBins]) )+'\n')
            outfile.write('\n')
            outfile.write( 'bin'.ljust(lspace)              +(' '.join( [' '.join([b.rjust(self.defWidth) for p in self.processes[b]] ) for b in unmutedBins]) ) +'\n')
            outfile.write( 'process'.ljust(lspace)          +(' '.join( [' '.join([p.rjust(self.defWidth) for p in self.processes[b]] ) for b in unmutedBins]) ) +'\n')
            outfile.write( 'process'.ljust(lspace)          +(' '.join( [' '.join([str(numberID[p]).rjust(self.defWidth) for p in self.processes[b]] ) for b in unmutedBins]) ) +'\n')
            outfile.write( 'rate'.ljust(lspace)             +(' '.join( [' '.join([self.mfs(self.expectation[(b,p)]).rjust(self.defWidth) for p in self.processes[b]] ) for b in unmutedBins]) )+'\n')
            outfile.write('\n')

            for u in self.uncertainties:
                outfile.write( u.ljust(self.maxUncNameWidth)+' '+self.uncertaintyString[u].ljust(self.maxUncStrWidth)+' '+
                                              ' '.join( [' '.join([self.getUncertaintyString((u,b,p)).rjust(self.defWidth) for p in self.processes[b]] ) for b in unmutedBins]) +'\n')

            for p in self.rateParameters:
                outfile.write('\n')
                for b in self.bins:
                    outfile.write('%s_norm_%s rateParam %s %s (@0*1) %s_norm\n'%(p[0], b, b, p[0], p[0]))
                outfile.write('%s_norm extArg %s %s\n'%(p[0], str(p[1]), str(p[2])))

            for p in self.freeParameters:
                outfile.write('\n')
                for b in self.bins:
                    outfile.write('%s rateParam %s %s %s %s\n'%(p[0], b, p[1], str(p[2]), str(p[3])))

            if shapeFile and not noMCStat:
                outfile.write('* autoMCStats 0 \n')

        #outfile.close()
        print("[cardFileWrite] Written card file %s"%fname)
        return fname

    def combineCards(self, cards, name='combined_card.txt'):

        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)

        years = list(cards.keys())
        cmd = ''
        for year in years:
            card_file = cards[year].split('/')[-1]
            print ("Copying card file to temp:", card_file)
            shutil.copyfile(cards[year], uniqueDirname+'/'+card_file)
            cmd += " dc_%s=%s"%(year, card_file)

        print (cmd)

        combineCommand  = "cd "+uniqueDirname+"; eval `scramv1 runtime -sh`; combineCards.py %s > combinedCard.txt; text2workspace.py combinedCard.txt --X-allow-no-signal -m 125"%(cmd)
        print ("Executing %s"%combineCommand)
        os.system(combineCommand)
        resFile = '/'+os.path.join(*cards[years[0]].split('/')[:-1]) + '/' + name
        f = resFile.split('/')[-1]
        resPath = resFile.replace(f, '')
        if not os.path.isdir(resPath):
            os.makedirs(resPath)
        print("Putting combined card into dir %s"%resPath)
        shutil.copyfile(uniqueDirname+"/combinedCard.txt", resFile)
        shutil.copyfile(uniqueDirname+"/combinedCard.root", resFile.replace(".txt",".root"))

        return resFile

    def readResFile(self, fname):
        import uproot
        f = uproot.open(fname)
        t = f["limit"]
        limits = t.array("limit")
        quantiles = t.array("quantileExpected")
        quantiles = quantiles.astype(str)
        limit = { quantile_dict[q]:limits[i] for i,q in enumerate(quantiles) }
        print (limit)
        return limit

    def calcLimit(self, fname=None, options="", verbose=False):
        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        if verbose: print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)

        if fname is not None:  # Assume card is already written when fname is not none
          filename = os.path.abspath(fname)
        else:
          filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
          self.writeToFile(filename)
        resultFilename = filename.replace('.txt','')+'.root'

        assert os.path.exists(filename), "File not found: %s"%filename
        
        
        combineCommand = "cd "+uniqueDirname+";eval `scramv1 runtime -sh`;combine --saveWorkspace -M AsymptoticLimits %s %s"%(options,filename)
        if verbose: print("Executing command:", combineCommand)
        os.system(combineCommand)

        tempResFile = uniqueDirname+"/higgsCombineTest.AsymptoticLimits.mH120.root"

        #print (tempResFile)
        
        try:
            res= self.readResFile(tempResFile)
            res['card'] = fname
        except:
            res=None
            print("[cardFileWrite] Did not succeed reading result.")
        if res:
            shutil.copyfile(tempResFile, resultFilename)
        
        #shutil.rmtree(uniqueDirname)
        return res

    def readNLLFile(self, fname):
        import uproot
        f = uproot.open(fname)
        t = f["limit"]
        nll = {}
        # prefit NLL
        nll["nll0"] = t.array('nll0')[0]
        # delta NLL to prefit (should always be negative since stuff is fitted)
        nll["nll"] = t.array('nll')[0]
        # absolute NLL postfit
        nll["nll_abs"] = nll["nll0"] + nll["nll"] #t.nll0 + t.nll
        return nll

    def calcNLL(self, fname=None, options=""):
        '''
        Does max likelihood fits, both with r=1 and a best-fit value
        '''
        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)
        if fname is not None:  # Assume card is already written when fname is not none
          filename = os.path.abspath(fname)
        else:
          filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
          self.writeToFile(filename)
        combineCommand  = "cd "+uniqueDirname+";eval `scramv1 runtime -sh`;combine -M MultiDimFit -n Nominal --saveNLL --expectSignal=1 --freezeParameters r --setParameters r=1 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 %s"%filename
        #combineCommand  = "cd "+uniqueDirname+";eval `scramv1 runtime -sh`;combine -M MultiDimFit -n Nominal --saveNLL --forceRecreateNLL --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --freezeParameters r %s %s"%(options,filename)
        os.system(combineCommand)

        #nll = self.readNLLFile(uniqueDirname+"/higgsCombineNominal.MultiDimFit.mH120.root")
        #nll["bestfit"] = nll["nll"]
        ##shutil.rmtree(uniqueDirname)

        import uproot
        import copy
        #with uproot.open(uniqueDirname+"/higgsCombineNominal.MultiDimFit.mH120.root") as f:
        with uproot.open(uniqueDirname+"/higgsCombineNominal.MultiDimFit.mH120.root") as f:
            tree = f['limit']
            result = copy.deepcopy( tree.arrays() )

        return result

    def nllScan(self, fname=None, rmin=0, rmax=5, npoints=11, options=""):
        import uuid, os
        import pandas
        import uproot
        import copy
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)
        if fname is not None:  # Assume card is already written when fname is not none
          filename = os.path.abspath(fname)
        else:
          filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
          self.writeToFile(filename)

        combineCommand  = "cd "+uniqueDirname+";eval `scramv1 runtime -sh`;combine -M MultiDimFit --algo grid --rMin %s --rMax %s --points %s --alignEdges 1 --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --saveNLL --forceRecreateNLL %s %s"%(rmin, rmax, npoints, options,filename)
        os.system(combineCommand)
        
        with uproot.open(uniqueDirname+"/higgsCombineTest.MultiDimFit.mH120.root") as f:
            tree = f['limit']
            result = copy.deepcopy( tree.arrays() )
    
        shutil.rmtree(uniqueDirname)

        return result

    def fit_diagnostics(self, fname=None, options=""):
        '''
        Does max likelihood fits, both with r=1 and a best-fit value
        '''
        import uuid, os
        import uproot
        from plots.helpers import get_yahist, finalizePlotDir
        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)
        if fname is not None:  # Assume card is already written when fname is not none
          filename = os.path.abspath(fname)
        else:
          filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
          self.writeToFile(filename)
        combineCommand  = "cd "+uniqueDirname+";eval `scramv1 runtime -sh`;combine --robustHesse 1 --forceRecreateNLL -M FitDiagnostics --saveShapes --saveNormalizations --saveOverall --saveWithUncertainties %s %s"%(options,filename)
        os.system(combineCommand)

        fit_list = ['shapes_prefit', 'shapes_fit_s', 'shapes_fit_b']

        res = uproot.open(uniqueDirname+'/fitDiagnostics.root')

        for fit in fit_list:
            histos = res[fit].keys()
            histos = [ x.replace(';1','') for x in histos if (x.count('data')==0 and x.count('covar')==0 and x.count('/')>0)]

            new_histos = { x:get_yahist(res['%s/%s'%(fit, x)], overflow=False) for x in histos }

            for new_hist in new_histos:

                region, process = new_hist.split('/')

                fig, ax = plt.subplots(1,1,figsize=(10,10))#, gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)

                hep.cms.label(
                    "Preliminary",
                    data=True,
                    #lumi=60.0,
                    loc=0,
                    ax=ax,
                )

                hep.histplot(
                    new_histos[new_hist].counts,
                    new_histos[new_hist].edges,
                    w2=new_histos[new_hist].errors,
                    histtype="step",
                    stack=False,
                    label='%s (%.0f)'%(new_hist, sum(new_histos[new_hist].counts)),
                    ax=ax)

                ax.set_ylabel(r'Events')
                ax.legend()

                plot_dir = os.path.expandvars('/home/users/$USER/public_html/tW_scattering/fitDiagnostics/%s/'%fit)
                finalizePlotDir(plot_dir)
                if not os.path.isdir(plot_dir):
                    os.makedirs(plot_dir)

                ext = ['png', 'pdf']
                for e in ext:
                    fig.savefig(plot_dir + '/%s_%s.'%(region, process)+e)


                del fig, ax

        return filename

    def calcNuisances(self, fname=None, options="", outputFileAddon = "", bonly=False):
        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)
        shutil.copyfile(os.path.join(os.environ['CMSSW_BASE'], 'src', 'Analysis', 'Tools', 'python', 'cardFileWriter', 'diffNuisances.py'), os.path.join(uniqueDirname, 'diffNuisances.py'))

        if fname is not None:  # Assume card is already written when fname is not none
          filename = os.path.abspath(fname)
        else:
          filename = fname if fname else os.path.join(uniqueDirname, ustr+".txt")
          self.writeToFile(filename)
        resultFilename      = filename.replace('.txt','')+'_nuisances.txt'
        resultFilenameFull  = filename.replace('.txt','')+'_nuisances_full.txt'
        resultFilename2     = filename.replace('.txt','')+'_nuisances.tex'
        resultFilename2Full = filename.replace('.txt','')+'_nuisances_full.tex'

        assert os.path.exists(filename), "File not found: %s"%filename

        combineCommand  = "cd "+uniqueDirname+";combine --robustHesse 1 --forceRecreateNLL -M FitDiagnostics --saveShapes --saveNormalizations --saveOverall --saveWithUncertainties %s %s"%(options,filename)
        combineCommand +=";python diffNuisances.py  fitDiagnostics.root &> nuisances.txt"
        combineCommand +=";python diffNuisances.py -a fitDiagnostics.root &> nuisances_full.txt"
        if bonly:
          combineCommand +=";python diffNuisances.py -bf latex fitDiagnostics.root &> nuisances.tex"
          combineCommand +=";python diffNuisances.py -baf latex fitDiagnostics.root &> nuisances_full.tex"
        else:
          combineCommand +=";python diffNuisances.py -f latex fitDiagnostics.root &> nuisances.tex"
          combineCommand +=";python diffNuisances.py -af latex fitDiagnostics.root &> nuisances_full.tex"
        print(combineCommand)
        os.system(combineCommand)

        if outputFileAddon: outputFileAddon = "_"+outputFileAddon
        shutil.copyfile(uniqueDirname+'/fitDiagnostics.root', fname.replace('.txt','%s_FD.root'%(outputFileAddon)))

        tempResFile      = uniqueDirname+"/nuisances%s.txt"%(outputFileAddon)
        tempResFileFull  = uniqueDirname+"/nuisances%s_full.txt"%(outputFileAddon)
        tempResFile2     = uniqueDirname+"/nuisances%s.tex"%(outputFileAddon)
        tempResFile2Full = uniqueDirname+"/nuisances%s_full.tex"%(outputFileAddon)
        shutil.copyfile(tempResFile, resultFilename)
        shutil.copyfile(tempResFileFull, resultFilenameFull)
        shutil.copyfile(tempResFile2, resultFilename2)
        shutil.copyfile(tempResFile2Full, resultFilename2Full)

        shutil.rmtree(uniqueDirname)
        return


    def calcSignif(self, fname="", options=""):
        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)
        combineCommand = "cd "+uniqueDirname+";combine --saveWorkspace -M ProfileLikelihood --uncapped 1 --significance --rMin -5 %s %s"%(options,fname)
        os.system(combineCommand)
        try:
            res= self.readResFile(uniqueDirname+"/higgsCombineTest.ProfileLikelihood.mH120.root")
            os.system("rm -rf "+uniqueDirname+"/higgsCombineTest.ProfileLikelihood.mH120.root")
        except:
            res=None
            print("Did not succeed.")
        shutil.rmtree(uniqueDirname)

        return res

    def cleanUp(self):
        for d in os.listdir(self.releaseLocation):
            if len(d) == len('43a8a7c4-0086-4ae8-94df-b1162165ddf4'):
                print ("Deleting: ", self.releaseLocation+'/'+d)
                shutil.rmtree(self.releaseLocation+'/'+d)


    def run_impacts(self, fname="", bkgOnly=False, observed=False, cores=8, plot_dir='./'):
        import uuid, os
        ustr          = str(uuid.uuid4())
        uniqueDirname = os.path.join(self.releaseLocation, ustr)
        print("Creating %s"%uniqueDirname)
        os.makedirs(uniqueDirname)

        shutil.copyfile(fname, uniqueDirname+'/'+fname.split('/')[-1])

        fname = uniqueDirname+'/'+fname.split('/')[-1]
        
        prepWorkspace   = "text2workspace.py %s -m 125"%fname
        workspace = fname.replace('.txt', '.root')

        print ('datacard: %s --> workspace %s'%(fname, workspace))

        if bkgOnly:
            robustFit       = "combineTool.py -M Impacts -d %s -m 125 -t -1 --expectSignal 0 --doInitialFit --robustFit 1 --rMin -10 --rMax 10"%workspace
            impactFits      = "combineTool.py -M Impacts -d %s -m 125 -t -1 --expectSignal 0 --robustFit 1 --doFits --parallel %s --rMin -10 --rMax 10"%(workspace, str(cores))
        elif observed:
            robustFit       = "combineTool.py -M Impacts -d %s -m 125 --doInitialFit --robustFit 1 --rMin -10 --rMax 10"%workspace
            impactFits      = "combineTool.py -M Impacts -d %s -m 125 --robustFit 1 --doFits --parallel %s --rMin -10 --rMax 10"%(workspace, str(cores))
        else:
            robustFit       = "combineTool.py -M Impacts -d %s -m 125 -t -1 --expectSignal 1 --doInitialFit --robustFit 1 --rMin -10 --rMax 10"%workspace
            impactFits      = "combineTool.py -M Impacts -d %s -m 125 -t -1 --expectSignal 1 --robustFit 1 --doFits --parallel %s --rMin -10 --rMax 10"%(workspace, str(cores))
        extractImpact   = "combineTool.py -M Impacts -d %s -m 125 -o impacts.json"%workspace
        plotImpacts     = "plotImpacts.py -i impacts.json -o impacts"
        combineCommand  = "cd %s;eval `scramv1 runtime -sh`;%s;%s;%s;%s;%s"%(uniqueDirname,prepWorkspace,robustFit,impactFits,extractImpact,plotImpacts)
        
        print ("Running the following command:")
        print (combineCommand)

        os.system(combineCommand)
        
        #if args.expected:
        #    s.name += '_expected'
        #if args.bkgOnly:
        #    s.name += '_bkgOnly'
        #if args.observed:
        #    s.name += '_observed'
        if not os.path.isdir(plot_dir): os.makedirs(plotDir)
        shutil.copyfile(uniqueDirname+'/impacts.pdf', "%s/impacts.pdf"%(plot_dir))
        #elif args.year:
        #    shutil.copyfile(combineDirname+'/impacts.pdf', "%s/%s_%s%s.pdf"%(plotDir,s.name,args.year,'_signalInjected' if args.signalInjection else ''))
        #else:
        #    shutil.copyfile(combineDirname+'/impacts.pdf', "%s/%s.pdf"%(plotDir,s.name))
        #logger.info("Copied result to %s"%plotDir)
    
        #if args.removeDir:
        #    logger.info("Removing directory in release location")
        #    rmtree(combineDirname)
