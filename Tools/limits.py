import os
import re
from coffea import hist
import uproot3
import numpy as np
from Tools.dataCard import *

notdata = re.compile('(?!pseudodata)')
notsignal = re.compile('(?!topW_v2)')

def myRebin(var, nbins, binsize, threshold):
    #values = output[var]['topW_v2'].sum("dataset").values()[()]
    values = output[var]['topW_v2'].sum("dataset").values()[()]
    bin_boundaries = [0]
    last_index = 0

    for i in range(nbins): # loop over the bins like a cave man
        #output['leadingForward_p']['topW_v2'].sum("dataset").values()[()]
        if values[last_index:i].sum() > threshold:
            bin_boundaries.append(i)
            last_index = i
    return np.array(bin_boundaries)*binsize

def makeCardFromHist(out_cache, hist_name, nonprompt_scale=1, signal_scale=1, bkg_scale=1, overflow='all', ext='', systematics=True):
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
        'topW_v2': signal_scale,
        'TTW': bkg_scale, # only scale the most important backgrounds
        'TTZ': bkg_scale,
        'TTH': bkg_scale,
    }
    histogram.scale(scales, axis='dataset')
    
    ## making a histogram for pseudo observation. this hurts, but rn it seems to be the best option
    data_counts = np.asarray(np.round(histogram[notdata].integrate('dataset').values(overflow=overflow)[()], 0), int)
    data_hist = histogram['topW_v2']
    data_hist.clear()
    data_hist_bins = data_hist.axes()[1]
    for i, edge in enumerate(data_hist_bins.edges(overflow=overflow)):
        if i >= len(data_counts): break
        for y in range(data_counts[i]):
            data_hist.fill(**{'dataset': 'data', data_hist_bins.name: edge+0.0001})
            

    other_sel   = re.compile('(TTTT|diboson|DY|rare)')
    ##observation = hist.export1d(histogram['pseudodata'].integrate('dataset'), overflow=overflow)
    #observation = hist.export1d(data_hist['data'].integrate('dataset'), overflow=overflow)
    observation = hist.export1d(histogram[notdata].integrate('dataset'), overflow=overflow)
    tw          = hist.export1d(histogram['topW_v2'].integrate('dataset'), overflow=overflow)
    ttw         = hist.export1d(histogram['TTW'].integrate('dataset'), overflow=overflow)
    ttz         = hist.export1d(histogram['TTZ'].integrate('dataset'), overflow=overflow)
    tth         = hist.export1d(histogram['TTH'].integrate('dataset'), overflow=overflow)
    rare        = hist.export1d(histogram[other_sel].integrate('dataset'), overflow=overflow)
    nonprompt   = hist.export1d(histogram['ttbar'].integrate('dataset'), overflow=overflow)
    
    fout = uproot3.recreate(shape_file)

    fout["signal"]    = tw
    fout["nonprompt"] = nonprompt
    fout["ttw"]       = ttw
    fout["ttz"]       = ttz
    fout["tth"]       = tth
    fout["rare"]      = rare
    fout["data_obs"]  = observation
    fout.close()
    
    # Get the total yields to write into a data card
    totals = {}
    
    totals['signal']      = histogram['topW_v2'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['ttw']         = histogram['TTW'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['ttz']         = histogram['TTZ'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['tth']         = histogram['TTH'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['rare']        = histogram['rare'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['nonprompt']   = histogram['ttbar'].integrate('dataset').values(overflow=overflow)[()].sum()
    ##totals['observation'] = histogram['pseudodata'].integrate('dataset').values(overflow=overflow)[()].sum()
    #totals['observation'] = int(sum(data_hist['data'].sum('dataset').values(overflow=overflow)[()]))
    totals['observation'] = histogram[notdata].integrate('dataset').values(overflow=overflow)[()].sum()
    
    print ("{:30}{:.2f}".format("Signal expectation:",totals['signal']) )
    print ("{:30}{:.2f}".format("Non-prompt background:",totals['nonprompt']) )
    print ("{:30}{:.2f}".format("t(t)X(X)/rare background:",totals['ttw']+totals['ttz']+totals['tth']+totals['rare']) )
    print ("{:30}{:.2f}".format("Observation:", totals['observation']) )
    
    
    # set up the card
    card = dataCard()
    card.reset()
    card.setPrecision(3)
    
    # add the uncertainties (just flat ones for now)
    card.addUncertainty('lumi', 'lnN')
    card.addUncertainty('ttw_norm', 'lnN')
    card.addUncertainty('ttz_norm', 'lnN')
    card.addUncertainty('tth_norm', 'lnN')
    card.addUncertainty('rare_norm', 'lnN')
    card.addUncertainty('fake', 'lnN')
    
    # add the single bin
    card.addBin('Bin0', [ 'ttw', 'ttz', 'tth', 'rare', 'nonprompt' ], 'Bin0')
    card.specifyExpectation('Bin0', 'signal', totals['signal'] )
    card.specifyExpectation('Bin0', 'ttw', totals['ttw'] )
    card.specifyExpectation('Bin0', 'ttz', totals['ttz'] )
    card.specifyExpectation('Bin0', 'tth', totals['tth'] )
    card.specifyExpectation('Bin0', 'rare', totals['rare'] )
    card.specifyExpectation('Bin0', 'nonprompt', totals['nonprompt'] )
    
    # set uncertainties
    if systematics:
        card.specifyUncertainty('ttw_norm', 'Bin0', 'ttw', 1.15 )
        card.specifyUncertainty('ttz_norm', 'Bin0', 'ttz', 1.10 )
        card.specifyUncertainty('tth_norm', 'Bin0', 'tth', 1.20 )
        card.specifyUncertainty('rare_norm', 'Bin0', 'rare', 1.20 )
        card.specifyUncertainty('fake', 'Bin0', 'nonprompt', 1.25 )
        card.specifyFlatUncertainty('lumi', 1.03)
    
    ## observation
    #card.specifyObservation('Bin0', int(round(totals['observation'],0)))
    card.specifyObservation('Bin0', totals['observation'])
    
    print ("Done.\n")
    
    return card.writeToFile(data_card, shapeFile=shape_file)

if __name__ == '__main__':

    '''
    This is probably broken, but an example of how to use the above functions

    '''

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
