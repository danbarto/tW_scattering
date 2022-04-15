import os
import re
from coffea import hist
import uproot
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
        #print (tmp_variation.values())
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        pdf_unc += (tmp_variation[process].sum('dataset').values(overflow='all')[()]-central)**2

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')

    pdf_unc = np.sqrt(pdf_unc)

    up_hist = Hist1D.from_bincounts(
        (central+pdf_unc)/central,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        (central-pdf_unc)/central,
        edges,
    )

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(pdf_unc):
            print (i, round(val/central[i],2))
    
        print (central)

    return  up_hist, down_hist

def get_scale_unc(output, hist_name, process, rebin=None, quiet=True, keep_norm=False):
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

        norm = 1
        if keep_norm:
            proc_norm_central   = output['norm'][process].sum('dataset').sum('one').values(overflow='all')[()]
            #print (proc_norm_central)
            proc_norm_var       = output['_scale_%s'%i][process].sum('dataset').sum('one').values(overflow='all')[()]
            #print (proc_norm_var)
            norm = (proc_norm_central)/proc_norm_var

        tmp_variation = output['%s_scale_%s'%(hist_name, i)].copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        scale_unc = np.maximum(
            scale_unc,
            np.abs(tmp_variation[process].sum('dataset').values(overflow='all')[()] * norm - central)
        )

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')
    
    up_hist = Hist1D.from_bincounts(
        (central+scale_unc)/central,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        (central-scale_unc)/central,
        edges,
    )

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(scale_unc):
            print (i, round(val/central[i],2))
    
    return  up_hist, down_hist

def get_ISR_unc(output, hist_name, process, rebin=None, quiet=True):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    From auto documentation of NanoAODv8
    
    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    
    --> take 0, 2 for ISR variations
    
    '''
    
    # now get the actual values
    tmp_central = output[hist_name].copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow='all')[()]
    
    for i in [0,2]:
        tmp_variation = output['%s_PS_%s'%(hist_name, i)].copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        if i == 2:
            up_unc = tmp_variation[process].sum('dataset').values(overflow='all')[()]
        if i == 0:
            down_unc = tmp_variation[process].sum('dataset').values(overflow='all')[()]

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')
    
    up_hist = Hist1D.from_bincounts(
        up_unc/central,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        down_unc/central,
        edges,
    )

    return  up_hist, down_hist

def get_FSR_unc(output, hist_name, process, rebin=None, quiet=True):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    From auto documentation of NanoAODv8
    
    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    
    --> take 1, 3 for ISR variations
    
    '''
    
    # now get the actual values
    tmp_central = output[hist_name].copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow='all')[()]
    
    for i in [1,3]:
        tmp_variation = output['%s_PS_%s'%(hist_name, i)].copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        if i == 3:
            up_unc = tmp_variation[process].sum('dataset').values(overflow='all')[()]
        if i == 1:
            down_unc = tmp_variation[process].sum('dataset').values(overflow='all')[()]

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow='all')
    
    up_hist = Hist1D.from_bincounts(
        up_unc/central,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        down_unc/central,
        edges,
    )

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
        up_unc/central,
        edges,
    )
    
    down_hist = Hist1D.from_bincounts(
        down_unc/central,
        edges,
    )
    
    if not quiet:
        print (process)
        print ("Rel. uncertainties:")
        for i, val in enumerate(up_unc):
            print (i, round(abs(up_unc[i]-down_unc[i])/(2*central[i]),2))

    return  up_hist, down_hist

def regroup_and_rebin(histo, rebin, mapping):
    tmp = histo.copy()
    tmp = tmp.rebin(rebin.name, rebin)
    tmp = tmp.group("dataset", hist.Cat("dataset", "new grouped dataset"), mapping)
    return tmp

def get_systematics(output, hist, year, correlated=False, signal=True):
    if correlated:
        year = "cor"
    systematics = []

    all_processes = ['TTW', 'TTZ', 'TTH']
    if signal: all_processes += ['signal']

    for proc in all_processes:
        systematics += [
            ('jes_%s'%year,     get_unc(output, hist, proc, '_pt_jesTotal', quiet=True), proc),
            ('b_%s'%year,       get_unc(output, hist, proc, '_b', quiet=True), proc),
            ('light_%s'%year,   get_unc(output, hist, proc, '_l', quiet=True), proc),
            ('PU',      get_unc(output, hist, proc, '_PU', quiet=True), proc),
        ]

    for proc in ['TTW', 'TTZ', 'TTH']:
        systematics += [
            ('pdf', get_pdf_unc(output, hist, proc), proc),  # FIXME not keep_norm yet
            ('FSR', get_FSR_unc(output, hist, proc), proc),
        ]

    systematics += [
        ('scale_TTW', get_scale_unc(output, hist, 'TTW', keep_norm=True), 'TTW'),
        ('scale_TTH', get_scale_unc(output, hist, 'TTH', keep_norm=True), 'TTH'),
        ('scale_TTZ', get_scale_unc(output, hist, 'TTZ', keep_norm=True), 'TTZ'),
        ('ISR_TTW', get_ISR_unc(output, hist, 'TTW'), 'TTW'),
        ('ISR_TTH', get_ISR_unc(output, hist, 'TTH'), 'TTH'),
        ('ISR_TTZ', get_ISR_unc(output, hist, 'TTZ'), 'TTZ'),
        ##('ttz_norm', 1.10, 'TTZ'),
        ##('tth_norm', 1.20, 'TTH'),
        ('rare_norm', 1.20, 'rare'),
        ('nonprompt_norm', 1.30, 'nonprompt'),
        ('chargeflip_norm', 1.20, 'chargeflip'),
        ('conversion_norm', 1.20, 'conversion')
    ]
    return systematics

def add_signal_systematics(output, hist, year, correlated=False, systematics=[], proc='signal'):
    if correlated:
        year = "cor"
    systematics += [
        ('jes_%s'%year,     get_unc(output, hist, proc, '_pt_jesTotal'), proc),
        ('b_%s'%year,       get_unc(output, hist, proc, '_b'), proc),
        ('light_%s'%year,   get_unc(output, hist, proc, '_l'), proc),
        ('PU',      get_unc(output, hist, proc, '_PU'), proc),
    ]
    return systematics

def makeCardFromHist(
        out_cache,
        hist_name,
        scales={'nonprompt':1, 'signal':1},  # this scales everything, also the expected observation
        bsm_scales={'TTZ':1},  # this only scales the expected background + signal, but leaves expected observation at 1
        overflow='all',
        ext='',
        systematics={},
        signal_hist=None,
        bsm_vals=None,
        sm_vals=None,
        integer=False,
        quiet=False,
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
    histogram_sm = out_cache[hist_name].copy()
    #histogram = histogram.rebin('mass', bins[hist_name]['bins'])
    
    # scale some processes
    histogram.scale(scales, axis='dataset')
    histogram_sm.scale(scales, axis='dataset')
    histogram.scale(bsm_scales, axis='dataset')
    
    ## making a histogram for pseudo observation. this hurts, but rn it seems to be the best option
    data_counts = np.asarray(np.round(histogram_sm.integrate('dataset').values(overflow=overflow)[()], 0), int)
    data_hist = histogram['signal']
    data_hist.clear()
    data_hist_bins = data_hist.axes()[1]
    for i, edge in enumerate(data_hist_bins.edges(overflow=overflow)):
        if i >= len(data_counts): break
        for y in range(data_counts[i]):
            data_hist.fill(**{'dataset': 'data', data_hist_bins.name: edge+0.0001})


    fout = uproot.recreate(shape_file)

    processes = [ p[0] for p in list(histogram.values().keys()) if p[0] != 'signal']  # ugly conversion

    # Start with making the central value histograms.
    # NOTE: this works!
    #
    for process in processes + ['signal']:
        if (signal_hist is not None) and process=='signal':
            #fout[process] = hist.export1d(signal_hist.integrate('dataset'), overflow=overflow)
            fout[process] = signal_hist.integrate('dataset').to_hist()
        elif (bsm_vals is not None) and process=='signal':
            fout[process] = yahist_to_root(bsm_vals, 'signal', 'signal', overflow='all')
        else:
            #fout[process] = hist.export1d(histogram[process].integrate('dataset'), overflow=overflow)
            fout[process] = histogram[process].integrate('dataset').to_hist()

    if integer:
        #fout["data_obs"]  = hist.export1d(data_hist.integrate('dataset'), overflow=overflow)
        fout["data_obs"]  = data_hist.integrate('dataset').to_hist()
    else:
        if sm_vals is not None:
            bkg_tmp = histogram_sm.integrate('dataset').values(overflow=overflow)[()]
            sig_tmp = histogram_sm['signal'].integrate('dataset').values(overflow=overflow)[()]
            tmp_hist = Hist1D.from_bincounts(
                bkg_tmp+sm_vals.counts-sig_tmp,
                data_hist_bins.edges(overflow=overflow),
            )
            fout["data_obs"]  = yahist_to_root(tmp_hist, 'data_obs', 'data_obs', overflow=overflow)
        else:
            #fout["data_obs"]  = hist.export1d(histogram_sm.integrate('dataset'), overflow=overflow)
            fout["data_obs"]  = histogram_sm.integrate('dataset').to_hist()

    
    # Get the total yields to write into a data card
    # NOTE: this works!
    #
    totals = {}
    
    for process in processes + ['signal']:
        if (signal_hist is not None) and process=='signal':
            totals[process] = signal_hist.integrate('dataset').values(overflow=overflow)[()].sum()
        elif (bsm_vals is not None) and process=='signal':
            print ("sum in limit tool", sum(bsm_vals.counts))
            totals[process] = sum(bsm_vals.counts)
        else:
            totals[process] = histogram[process].integrate('dataset').values(overflow=overflow)[()].sum()
    
    if integer:
        totals['observation'] = data_hist.integrate('dataset').values(overflow=overflow)[()].sum()  # this is always with the SM signal
    else:
        if sm_vals is not None:
            totals['observation'] = histogram_sm.integrate('dataset').values(overflow=overflow)[()].sum() \
                + sum(sm_vals.counts) \
                - histogram['signal'].integrate('dataset').values(overflow=overflow)[()].sum()
        else:
            totals['observation'] = histogram_sm.integrate('dataset').values(overflow=overflow)[()].sum()  # this is always with the SM signal
    
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
                # if systematics are shapes we need to scale them similar to the expectations
                if proc in scales:
                    scale = scales[proc]
                else:
                    scale = 1
                if proc in bsm_scales:
                    scale *= bsm_scales[proc]

                card.addUncertainty(systematic, 'shape')
                print ("Adding shape uncertainty %s for process %s."%(systematic, proc))

                if len(mag)>1:
                    # NOTE: if we switch to relative uncertainties we need the central values too.
                    # NOTE need to make a new yahist with scaled counts. fuck
                    # However, here at least I don't care about the unceratinties
                    if proc == 'signal' and bsm_vals:
                        central = bsm_vals.counts
                    else:
                        central = histogram[proc].integrate('dataset').values(overflow=overflow)[()]
                    val = np.nan_to_num(mag[0].counts, nan=1.0) * scale * central
                    val_h = Hist1D.from_bincounts(val, mag[0].edges)
                    incl_rel = sum(val_h.counts)/sum(central)
                    print ("Integrated systematic uncertainty %s for %s:"%(systematic, proc))
                    print (" - central prediction: %.2f"%sum(central))
                    print (" - relative uncertainty: %.2f"%incl_rel)

                    fout[proc+'_'+systematic+'Up']   = yahist_to_root(
                        #mag[0] * scale * histogram[process].integrate('dataset').values(overflow=overflow)[()],
                        val_h,
                        systematic+'Up',
                        systematic+'Up',
                    )
                    val = np.nan_to_num(mag[1].counts, nan=1.0) * scale * central
                    val_h = Hist1D.from_bincounts(val, mag[1].edges)
                    #incl_rel = sum(val_h.counts)/sum(central)
                    fout[proc+'_'+systematic+'Down'] = yahist_to_root(
                        #mag[1]*scale * histogram[process].integrate('dataset').values(overflow=overflow)[()],
                        val_h,
                        systematic+'Down',
                        systematic+'Down',
                    )
                else:
                    val = np.nan_to_num(mag[0].counts, nan=1.0) * scale * histogram[process].integrate('dataset').values(overflow=overflow)[()]
                    val_h = Hist1D.from_bincounts(val, mag[0].edges)
                    fout[proc+'_'+systematic] = yahist_to_root(
                        #mag[0] * scale * histogram[process].integrate('dataset').values(overflow=overflow)[()],
                        val_h,
                        systematic,
                        systematic,
                    )

                card.specifyUncertainty(systematic, 'Bin0', proc, 1)
            else:
                # this type systematic does not need any scaling
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
