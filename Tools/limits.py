import re

notdata = re.compile('(?!pseudodata)')
notsignal = re.compile('(?!tW_scattering)')

def myRebin(var, nbins, binsize, threshold):
    #values = output[var]['tW_scattering'].sum("dataset").values()[()]
    values = output[var]['tW_scattering'].sum("dataset").values()[()]
    bin_boundaries = [0]
    last_index = 0

    for i in range(nbins): # loop over the bins like a cave man
        #output['leadingForward_p']['tW_scattering'].sum("dataset").values()[()]
        if values[last_index:i].sum() > threshold:
            bin_boundaries.append(i)
            last_index = i
    return np.array(bin_boundaries)*binsize

def makeCardFromHist(hist_name, nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True):
    print ("Writing cards using histogram:", hist_name)
    card_dir = os.path.expandvars('$TWHOME/data/cards/')
    if not os.path.isdir(card_dir):
        os.makedirs(card_dir)
    
    data_card = card_dir+hist_name+ext+'_card.txt'
    shape_file = card_dir+hist_name+ext+'_shapes.root'
    
    histogram = out_cache[hist_name].copy()
    #histogram = histogram.rebin('mass', bins[hist_name]['bins'])
    
    # scale some processes
    scales = { 
        'ttbar': nonprompt_scale, 
        'tW_scattering': signal_scale,
        'TTW': bkg_scale, # only scale the most important backgrounds
        'TTZ': bkg_scale,
        'TTH': bkg_scale,
    }
    histogram.scale(scales, axis='dataset')
    
    #observation = hist.export1d(histogram['pseudodata'].integrate('dataset'), overflow=overflow)
    observation = export1d(histogram[notdata].integrate('dataset'), overflow=overflow)
    tw          = export1d(histogram['tW_scattering'].integrate('dataset'), overflow=overflow)
    onlyttx     = re.compile('(TTW|TTZ|TTH|TTTT|diboson|DY)')
    bkg         = export1d(histogram[onlyttx].integrate('dataset'), overflow=overflow)
    nonprompt   = export1d(histogram['ttbar'].integrate('dataset'), overflow=overflow)
    
    file = uproot.recreate(shape_file, compression=uproot.ZLIB(4))
    
    file["signal"]    = tw
    file["nonprompt"] = nonprompt
    file["bkg"]       = bkg
    file["data_obs"]  = observation
    
    # Get the total yields to write into a data card
    totals = {}
    
    totals['signal']      = histogram['tW_scattering'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['bkg']         = histogram[onlyttx].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['nonprompt']   = histogram['ttbar'].integrate('dataset').values(overflow=overflow)[()].sum()
    #totals['observation'] = histogram['pseudodata'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['observation'] = histogram[notdata].integrate('dataset').values(overflow=overflow)[()].sum()
    
    print ("{:30}{:.2f}".format("Signal expectation:",totals['signal']) )
    print ("{:30}{:.2f}".format("Non-prompt background:",totals['nonprompt']) )
    print ("{:30}{:.2f}".format("t(t)X(X)/rare background:",totals['bkg']) )
    print ("{:30}{:.2f}".format("Observation:", totals['observation']) )
    
    
    # set up the card
    card = dataCard()
    card.reset()
    card.setPrecision(3)
    
    # add the uncertainties (just flat ones for now)
    card.addUncertainty('lumi', 'lnN')
    card.addUncertainty('ttx', 'lnN')
    card.addUncertainty('fake', 'lnN')
    
    # add the single bin
    card.addBin('Bin0', [ 'bkg', 'nonprompt' ], 'Bin0')
    card.specifyExpectation('Bin0', 'signal', totals['signal'] )
    card.specifyExpectation('Bin0', 'bkg', totals['bkg'] )
    card.specifyExpectation('Bin0', 'nonprompt', totals['nonprompt'] )
    
    # set uncertainties
    if systematics:
        card.specifyUncertainty('ttx', 'Bin0', 'bkg', 1.20 )
        card.specifyUncertainty('fake', 'Bin0', 'nonprompt', 1.25 )
        card.specifyFlatUncertainty('lumi', 1.03)
    
    # observation
    card.specifyObservation('Bin0', int(round(totals['observation'],0)))
    
    print ("Done.\n")
    
    return card.writeToFile(data_card, shapeFile=shape_file)

if __name__ == '__main__':

    '''
    This is probably broken, but an example of how to use the above functions

    '''

    from Tools.dataCard import *
    from Tools.helpers import export1d

    year = 2018

    card_SR_ideal_noSyst = makeCardFromHist('mjj_max_tight', nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='_idealNoSyst', systematics=False)

    card = dataCard()

    import mplhep
    plt.style.use(mplhep.style.CMS)
    
    plt.figure()

    plt.plot(results_SR['r'][1:], results_SR['deltaNLL'][1:]*2, label=r'Expected ($M_{jj}$) SR', c='black')#, linewidths=2)


    plt.xlabel(r'$r$')
    plt.ylabel(r'$-2\Delta  ln L$')
    plt.legend()

    card.cleanUp()
