import os
import re
from coffea import hist
import uproot3
import numpy as np
from Tools.dataCard import *

from yahist import Hist1D
from Tools.yahist_to_root import yahist_to_root

def get_pdf_unc(output, hist_name, process, rebin=None, hessian=True, quiet=True):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    '''
    if not hessian:
        print ("Can't handle mc replicas.")
        return False
    
    
    # now get the actual values
    tmp_central = output[hist_name].copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow='all')[()]
    pdf_unc = np.zeros_like(central)
    
    
    for i in range(1,101):
        tmp_variation = output['%s_pdf_%s'%(hist_name, i)]
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        pdf_unc += (tmp_variation[process].sum('dataset').values(overflow='all')[()]-central)**2

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')

    pdf_unc = np.sqrt(pdf_unc)

    up_hist = Hist1D.from_bincounts(
        central+pdf_unc,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        central-pdf_unc,
        edges,
    )

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(pdf_unc):
            print (i, round(val/central[i],2))
    
        print (central)

    return  up_hist, down_hist

def get_scale_unc(output, hist_name, process, rebin=None, quiet=True):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    From auto documentation of NanoAODv8
    
    OBJ: TBranch LHEScaleWeight LHE scale variation weights (w_var / w_nominal);
    [0] is MUF="0.5" MUR="0.5"; [1] is MUF="1.0" MUR="0.5"; [2] is MUF="2.0" MUR="0.5";
    [3] is MUF="0.5" MUR="1.0"; [4] is MUF="1.0" MUR="1.0"; [5] is MUF="2.0" MUR="1.0";
    [6] is MUF="0.5" MUR="2.0"; [7] is MUF="1.0" MUR="2.0"; [8] is MUF="2.0" MUR="2.0"
    
    --> take 0, 1, 3 for down variations
    --> take 5, 7, 8 for up variations
    --> 4 is central, if needed
    
    '''
    
    # now get the actual values
    tmp_central = output[hist_name].copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow='all')[()]
    
    
    scale_unc = np.zeros_like(central)
    for i in [0,1,3,5,7,8]:
        '''
        Using the full envelope.
        Don't know how to make a sensible envelope of up/down separately,
        without getting vulnerable to weird one-sided uncertainties.
        '''
        tmp_variation = output['%s_scale_%s'%(hist_name, i)].copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        scale_unc = np.maximum(
            scale_unc,
            np.abs(tmp_variation[process].sum('dataset').values(overflow='all')[()]-central)
        )

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')
    
    up_hist = Hist1D.from_bincounts(
        central+scale_unc,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        central-scale_unc,
        edges,
    )

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(scale_unc):
            print (i, round(val/central[i],2))
    
    return  up_hist, down_hist

def get_unc(output, hist_name, process, unc, rebin=None, quiet=True):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    '''
    
    # now get the actual values
    tmp_central = output[hist_name].copy()
    tmp_up      = output[hist_name+unc+'Up'].copy()
    tmp_down    = output[hist_name+unc+'Down'].copy()
    
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
        tmp_up      = tmp_up.rebin(rebin.name, rebin)
        tmp_down    = tmp_down.rebin(rebin.name, rebin)
        
    central  = tmp_central[process].sum('dataset').values(overflow='all')[()]
    up_unc   = tmp_up[process].sum('dataset').values(overflow='all')[()]
    down_unc = tmp_down[process].sum('dataset').values(overflow='all')[()]   
    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')

    up_hist = Hist1D.from_bincounts(
        up_unc,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        down_unc,
        edges,
    )
    
    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(up_unc):
            print (i, round(abs(up_unc[i]-down_unc[i])/(2*central[i]),2))
                
    return  up_hist, down_hist

def regroup_and_rebin(histo, rebin, mapping):
    tmp = histo.copy()
    tmp = tmp.rebin(rebin.name, rebin)
    tmp = tmp.group("dataset", hist.Cat("dataset", "new grouped dataset"), mapping)
    return tmp

def get_systematics(output, hist):
    systematics = []
    for proc in ['signal', 'TTW', 'TTZ', 'TTH']:
        systematics += [
            ('jes',     get_unc(output, hist, proc, '_pt_jesTotal'), proc),
            ('b',       get_unc(output, hist, proc, '_b'), proc),
            ('light',   get_unc(output, hist, proc, '_l'), proc),
            ('PU',      get_unc(output, hist, proc, '_PU'), proc),
        ]

    for proc in ['TTW', 'TTZ', 'TTH']:
        systematics += [
            ('pdf', get_pdf_unc(output, hist, proc), proc),
        ]

    systematics += [
        ('scale_TTW', get_scale_unc(output, hist, 'TTW'), 'TTW'),
        ('scale_TTH', get_scale_unc(output, hist, 'TTH'), 'TTH'),
        ('scale_TTZ', get_scale_unc(output, hist, 'TTZ'), 'TTZ'),
        #('ttz_norm', 1.10, 'TTZ'),
        #('tth_norm', 1.20, 'TTH'),
        ('rare_norm', 1.20, 'rare'),
        ('nonprompt_norm', 1.30, 'nonprompt'),
        ('chargeflip_norm', 1.20, 'chargeflip'),
        ('conversion_norm', 1.20, 'conversion')
    ]

    return systematics

def makeCardFromHist(
    out_cache,
    hist_name,
    scales={'nonprompt':1, 'signal':1},
    overflow='all',
    ext='',
    systematics={},
    signal_hist=None,
    integer=False, quiet=False,
):
    
    '''
    make a card file from a processor output
    signal_hist overrides the default signal histogram if provided
    '''

    if not quiet:
        print ("Writing cards using histogram:", hist_name)
    card_dir = os.path.expandvars('$TWHOME/data/cards/')
    if not os.path.isdir(card_dir):
        os.makedirs(card_dir)
    
    data_card = card_dir+hist_name+ext+'_card.txt'
    shape_file = card_dir+hist_name+ext+'_shapes.root'
    
    histogram = out_cache[hist_name].copy()
    #histogram = histogram.rebin('mass', bins[hist_name]['bins'])
    
    # scale some processes
    histogram.scale(scales, axis='dataset')
    
    ## making a histogram for pseudo observation. this hurts, but rn it seems to be the best option
    data_counts = np.asarray(np.round(histogram.integrate('dataset').values(overflow=overflow)[()], 0), int)
    data_hist = histogram['signal']
    data_hist.clear()
    data_hist_bins = data_hist.axes()[1]
    for i, edge in enumerate(data_hist_bins.edges(overflow=overflow)):
        if i >= len(data_counts): break
        for y in range(data_counts[i]):
            data_hist.fill(**{'dataset': 'data', data_hist_bins.name: edge+0.0001})


    fout = uproot3.recreate(shape_file)

    processes = [ p[0] for p in list(histogram.values().keys()) if p[0] != 'signal']  # ugly conversion
    
    for process in processes + ['signal']:
        if (signal_hist is not None) and process=='signal':
            fout[process] = hist.export1d(signal_hist.integrate('dataset'), overflow=overflow)
        else:
            fout[process] = hist.export1d(histogram[process].integrate('dataset'), overflow=overflow)

    if integer:
        fout["data_obs"]  = hist.export1d(data_hist.integrate('dataset'), overflow=overflow)
    else:
        fout["data_obs"]  = hist.export1d(histogram.integrate('dataset'), overflow=overflow)

    
    # Get the total yields to write into a data card
    totals = {}
    
    for process in processes + ['signal']:
        if (signal_hist is not None) and process=='signal':
            totals[process] = signal_hist.integrate('dataset').values(overflow=overflow)[()].sum()
        else:
            totals[process] = histogram[process].integrate('dataset').values(overflow=overflow)[()].sum()
    
    if integer:
        totals['observation'] = data_hist.integrate('dataset').values(overflow=overflow)[()].sum()  # this is always with the SM signal
    else:
        totals['observation'] = histogram.integrate('dataset').values(overflow=overflow)[()].sum()  # this is always with the SM signal
    
    if not quiet:
        for process in processes + ['signal']:
            print ("{:30}{:.2f}".format("Expectation for %s:"%process, totals[process]) )
        
        print ("{:30}{:.2f}".format("Observation:", totals['observation']) )
    
    
    # set up the card
    card = dataCard()
    card.reset()
    card.setPrecision(3)
    
    # add the single bin
    card.addBin('Bin0', processes, 'Bin0')
    for process in processes + ['signal']:
        card.specifyExpectation('Bin0', process, totals[process] )
    
    # add the uncertainties (just flat ones for now)
    card.addUncertainty('lumi', 'lnN')
    if systematics:
        for systematic, mag, proc in systematics:
            if isinstance(mag, type(())):
                card.addUncertainty(systematic, 'shape')
                print ("Adding shape uncertainty %s for process %s."%(systematic, proc))
                if len(mag)>1:
                    fout[proc+'_'+systematic+'Up']   = yahist_to_root(mag[0], systematic+'Up', systematic+'Up')
                    fout[proc+'_'+systematic+'Down'] = yahist_to_root(mag[1], systematic+'Down', systematic+'Down')
                else:
                    fout[proc+'_'+systematic] = yahist_to_root(mag[0], systematic, systematic)
                card.specifyUncertainty(systematic, 'Bin0', proc, 1)
            else:
                card.addUncertainty(systematic, 'lnN')
                card.specifyUncertainty(systematic, 'Bin0', proc, mag)
            
    fout.close()

    card.specifyFlatUncertainty('lumi', 1.03)
    
             ## observation
    #card.specifyObservation('Bin0', int(round(totals['observation'],0)))
    card.specifyObservation('Bin0', totals['observation'])
    
    if not quiet:
        print ("Done.\n")
    
    return card.writeToFile(data_card, shapeFile=shape_file)




notdata = re.compile('(?!pseudodata)')
notsignal = re.compile('(?!topW_v2)')

def makeCardFromHist_ret(out_cache, hist_name, scales={'nonprompt':1, 'signal':1}, overflow='all', ext='', systematics=True, categories=False, bsm_hist=None, tw_name='topW_v2', quiet=False):
    print ("Writing cards using histogram:", hist_name)
    card_dir = os.path.expandvars('$TWHOME/data/cards/')
    if not os.path.isdir(card_dir):
        os.makedirs(card_dir)
    
    data_card = card_dir+hist_name+ext+'_card.txt'
    shape_file = card_dir+hist_name+ext+'_shapes.root'
    
    histogram = out_cache[hist_name].copy()
    #histogram = histogram.rebin('mass', bins[hist_name]['bins'])
    
    # scale some processes
    histogram.scale(scales, axis='dataset')
    
    ## making a histogram for pseudo observation. this hurts, but rn it seems to be the best option
    data_counts = np.asarray(np.round(histogram[notdata].integrate('dataset').values(overflow=overflow)[()], 0), int)
    data_hist = histogram[tw_name]
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
    tw          = hist.export1d(histogram[tw_name].integrate('dataset'), overflow=overflow) if bsm_hist is None else hist.export1d(bsm_hist.integrate('dataset'), overflow=overflow)
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
    
    if bsm_hist is not None:
        totals['signal']      = bsm_hist.integrate('dataset').values(overflow=overflow)[()].sum()
    else:
        totals['signal']      = histogram[tw_name].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['ttw']         = histogram['TTW'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['ttz']         = histogram['TTZ'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['tth']         = histogram['TTH'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['rare']        = histogram['rare'].integrate('dataset').values(overflow=overflow)[()].sum()
    totals['nonprompt']   = histogram['ttbar'].integrate('dataset').values(overflow=overflow)[()].sum()
    ##totals['observation'] = histogram['pseudodata'].integrate('dataset').values(overflow=overflow)[()].sum()
    #totals['observation'] = int(sum(data_hist['data'].sum('dataset').values(overflow=overflow)[()]))
    totals['observation'] = histogram[notdata].integrate('dataset').values(overflow=overflow)[()].sum()
    
    if not quiet:
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
        card.specifyUncertainty('ttw_norm', 'Bin0', 'ttw', 1.15 if not categories else 1.10 )  # this is the prompt category
        card.specifyUncertainty('ttz_norm', 'Bin0', 'ttz', 1.10 if not categories else 1.10 )  # lost lepton
        card.specifyUncertainty('tth_norm', 'Bin0', 'tth', 1.20 if not categories else 1.25 )  # nonprompt
        card.specifyUncertainty('rare_norm', 'Bin0', 'rare', 1.20 if not categories else 1.10 )  # charge flip
        card.specifyUncertainty('fake', 'Bin0', 'nonprompt', 1.25 if not categories else 1.10 )  # nothing
        card.specifyFlatUncertainty('lumi', 1.03)
    else:
        # use just very small systematics...
        card.specifyUncertainty('ttw_norm', 'Bin0', 'ttw', 1.03 )  # this is the prompt category
        card.specifyUncertainty('ttz_norm', 'Bin0', 'ttz', 1.03 )  # lost lepton
        card.specifyUncertainty('tth_norm', 'Bin0', 'tth', 1.03 )  # nonprompt
        card.specifyUncertainty('rare_norm', 'Bin0', 'rare', 1.03 )  # charge flip
        card.specifyUncertainty('fake', 'Bin0', 'nonprompt', 1.03 )  # nothing
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
