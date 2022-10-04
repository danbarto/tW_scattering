import os
import re
from coffea import hist
import uproot
import numpy as np
from Tools.dataCard import *
from Tools.helpers import make_bh

def get_norms(dataset, samples, mapping, name='pdf', weight='LHEPdfWeight'):
    '''
    this function does not need the actual histograms but the stored meta data
    from the samples database
    '''
    # samples[mapping['UL16']['TTW'][0]]['xsec']/samples[mapping['UL16']['TTW'][0]]['LHEPdfWeight'][1]
    #
    #norms = {f'{name}_{i}': 0 for x,i in enumerate(samples[mapping[dataset][0]])}
    #
    total = sum([ samples[s]['xsec']/samples[s][weight] for s in mapping[dataset] ])
    norms = {f'{name}_{i}': x for i,x in enumerate(total/total[0])}
    return norms

def get_pdf_unc(output, hist_name, process, eft_point, rebin=None, hessian=True, quiet=True, overflow='all', norms=None):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    '''
    if not hessian:
        print ("Can't handle mc replicas.")
        return False
    
    
    # now get the actual values
    tmp_central = output[hist_name][(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    pdf_unc = np.zeros_like(central)
    
    
    for i in range(1,101):
        tmp_variation = output[hist_name][(process, eft_point, 'central', f'pdf_{i}')].sum('EFT', 'systematic', 'prediction').copy()
        #print (tmp_variation.values())
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)

        norm = norms[f'pdf_{i}'] if (norms is not None) else 1
        pdf_unc += (tmp_variation[process].sum('dataset').values(overflow=overflow)[()]*norm-central)**2

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    pdf_unc = np.sqrt(pdf_unc)

    up_hist = make_bh((central+pdf_unc)/central, (central+pdf_unc)/central, edges)
    down_hist = make_bh((central-pdf_unc)/central, (central-pdf_unc)/central, edges)

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(pdf_unc):
            print (i, round(val/central[i],2))
    
        print (central)

    return  up_hist, down_hist

def get_scale_unc(output, hist_name, process, eft_point, rebin=None, quiet=True, overflow='all', norms=None):
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
    tmp_central = output[hist_name][(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]

    scale_unc = np.zeros_like(central)
    for i in [0,1,3,5,7,8]:
        '''
        Using the full envelope.
        Don't know how to make a sensible envelope of up/down separately,
        without getting vulnerable to weird one-sided uncertainties.
        '''

        norm = norms[f'scale_{i}'] if (norms is not None) else 1
        #if norms:
        #    proc_norm = norms[f'scale_{i}']

        tmp_variation = output[hist_name][(process, eft_point, 'central', f'scale_{i}')].sum('EFT', 'systematic', 'prediction').copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)

        tmp_var = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]
        print (i, sum(tmp_var)/sum(central))

        scale_unc = np.maximum(
            scale_unc,
            np.abs(tmp_var * norm - central)
        )

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    up_hist = make_bh((central+scale_unc)/central, (central+scale_unc)/central, edges)
    down_hist = make_bh((central-scale_unc)/central, (central-scale_unc)/central, edges)

    if not quiet:
        print ("Rel. uncertainties:")
        for i, val in enumerate(scale_unc):
            print (i, round(val/central[i],2))
    
    return  up_hist, down_hist


def get_ISR_unc(output, hist_name, process, eft_point, rebin=None, quiet=True, overflow='all', norm=None):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties

    From auto documentation of NanoAODv8

    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    --> take 0, 2 for ISR variations
    '''

    # now get the actual values
    tmp_central = output[hist_name][(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]

    for i in [0,2]:
        tmp_variation = output[hist_name][(process, eft_point, 'central', f'PS_{i}')].sum('EFT', 'systematic', 'prediction').copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        if i == 2:
            up_unc = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]
        if i == 0:
            down_unc = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    up_hist = make_bh(up_unc/central, up_unc/central, edges)
    down_hist = make_bh(down_unc/central, down_unc/central, edges)

    if not quiet:
        print (process)
        print ("Rel. uncertainties:")
        for i, val in enumerate(up_unc):
            print (i, round(abs(up_unc[i]-down_unc[i])/(2*central[i]),2))

    return  up_hist, down_hist


def get_FSR_unc(output, hist_name, process, eft_point, rebin=None, quiet=True, overflow='all', norm=None):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    From auto documentation of NanoAODv8
    
    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    --> take 1, 3 for FSR variations
    '''
    
    # now get the actual values
    tmp_central = output[hist_name][(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    
    for i in [1,3]:
        tmp_variation = output[hist_name][(process, eft_point, 'central', f'PS_{i}')].sum('EFT', 'systematic', 'prediction').copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)
        if i == 3:
            up_unc = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]
        if i == 1:
            down_unc = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]

    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    up_hist = make_bh(up_unc/central, up_unc/central, edges)
    down_hist = make_bh(down_unc/central, down_unc/central, edges)

    if not quiet:
        print (process)
        print ("Rel. uncertainties:")
        for i, val in enumerate(up_unc):
            print (i, round(abs(up_unc[i]-down_unc[i])/(2*central[i]),2))

    return  up_hist, down_hist

def get_unc(output, hist_name, process, unc, eft_point, rebin=None, quiet=True, overflow='all'):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    '''
    
    # now get the actual values
    tmp_central = output[hist_name][(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    tmp_up      = output[hist_name][(process, eft_point, 'central', unc+'_up')].sum('EFT', 'systematic', 'prediction').copy()
    tmp_down    = output[hist_name][(process, eft_point, 'central', unc+'_down')].sum('EFT', 'systematic', 'prediction').copy()
    
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
        tmp_up      = tmp_up.rebin(rebin.name, rebin)
        tmp_down    = tmp_down.rebin(rebin.name, rebin)
        
    central  = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    up_unc   = tmp_up[process].sum('dataset').values(overflow=overflow)[()]
    down_unc = tmp_down[process].sum('dataset').values(overflow=overflow)[()]
    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    up_hist = make_bh(up_unc/central, up_unc/central, edges)
    down_hist = make_bh(down_unc/central, down_unc/central, edges)

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

def get_systematics(output, hist, year, eft_point, correlated=False, signal=True, overflow='all'):
    if correlated:
        year = "cor"
    systematics = []

    all_processes = ['TTW', 'TTZ', 'TTH', 'rare', 'diboson']
    if signal: all_processes += ['signal']

    for proc in all_processes:
        systematics += [
            ('jes_%s'%year,     get_unc(output, hist, proc, 'jes',  eft_point, overflow=overflow, quiet=True), proc),
            ('b_%s'%year,       get_unc(output, hist, proc, 'b',    eft_point, overflow=overflow, quiet=True), proc),
            ('light_%s'%year,   get_unc(output, hist, proc, 'l',    eft_point, overflow=overflow, quiet=True), proc),
            ('mu_%s'%year,      get_unc(output, hist, proc, 'mu',   eft_point, overflow=overflow, quiet=True), proc),
            ('ele_%s'%year,     get_unc(output, hist, proc, 'ele',  eft_point, overflow=overflow, quiet=True), proc),
            ('PU',              get_unc(output, hist, proc, 'PU',   eft_point, overflow=overflow, quiet=True), proc),
        ]

    for proc in ['TTW', 'TTZ', 'TTH']:
        systematics += [
            ('pdf', get_pdf_unc(output, hist, proc, eft_point, overflow=overflow), proc),  # FIXME not keep_norm yet
            ('FSR', get_FSR_unc(output, hist, proc, eft_point, overflow=overflow), proc),
        ]

    #systematics += [
    #    ('scale_TTW', get_scale_unc(output, hist, 'TTW', keep_norm=True), 'TTW'),
    #    ('scale_TTH', get_scale_unc(output, hist, 'TTH', keep_norm=True), 'TTH'),
    #    ('scale_TTZ', get_scale_unc(output, hist, 'TTZ', keep_norm=True), 'TTZ'),
    #    ('ISR_TTW', get_ISR_unc(output, hist, 'TTW'), 'TTW'),
    #    ('ISR_TTH', get_ISR_unc(output, hist, 'TTH'), 'TTH'),
    #    ('ISR_TTZ', get_ISR_unc(output, hist, 'TTZ'), 'TTZ'),
    #    ##('ttz_norm', 1.10, 'TTZ'),
    #    ##('tth_norm', 1.20, 'TTH'),
    #    #('rare_norm', 1.20, 'rare'),
    #    ('nonprompt_norm', 1.30, 'nonprompt'),
    #    ('chargeflip_norm', 1.20, 'chargeflip'),
    #    ('conversion_norm', 1.20, 'conversion')
    #]
    return systematics

def add_signal_systematics(output, hist, year, eft_point, correlated=False, systematics=[], proc='signal'):
    if correlated:
        year = "cor"
    systematics += [
        ('jes_%s'%year,     get_unc(output, hist, proc, 'jes', eft_point), proc),
        ('b_%s'%year,       get_unc(output, hist, proc, 'b', eft_point), proc),
        ('light_%s'%year,   get_unc(output, hist, proc, 'l', eft_point), proc),
        ('PU',              get_unc(output, hist, proc, 'PU', eft_point), proc),
    ]
    return systematics

def makeCardFromHist(
        histograms,
        #hist_name,
        scales={'nonprompt':1, 'signal':1},  # this scales everything, also the expected observation
        bsm_scales={'TTZ':1},  # this only scales the expected background + signal, but leaves expected observation at 1
        ext='',
        systematics={},
        bsm_hist=None,
        #signal_hist=None,
        #bsm_vals=None,
        #sm_vals=None,
        integer=False,
        quiet=False,
        blind=True,
        data=None,
):
    
    '''
    make a card file from histograms
    histogams: python dictionary with 2D histograms: dataset axis and feature axis
    signal_hist overrides the default signal histogram if provided
    '''

    if not quiet:
        print ("Writing cards now")
    card_dir = os.path.expandvars('$TWHOME/data/cards/')
    if not os.path.isdir(card_dir):
        os.makedirs(card_dir)
    
    data_card = card_dir+ext+'_card.txt'
    shape_file = card_dir+ext+'_shapes.root'

    processes = [ p for p in histograms.keys() if p != 'signal' ]

    # make a copy of the histograms
    h_tmp = {}
    h_tmp_bsm = {}
    for p in processes + ['signal']:
        h_tmp[p] = histograms[p].copy()
        h_tmp[p].scale(scales, axis='dataset')  # scale according to the processes
        h_tmp[p] = h_tmp[p].sum('dataset')  # reduce to 1D histogram

        h_tmp_bsm[p] = histograms[p].copy()
        h_tmp_bsm[p].scale(scales, axis='dataset')  # scale according to the processes
        h_tmp_bsm[p].scale(bsm_scales, axis='dataset')  # scale according to the processes
        h_tmp_bsm[p] = h_tmp_bsm[p].sum('dataset')  # reduce to 1D histogram

    # get an axis
    axis = h_tmp["signal"].axes()[0]

    # make observation boost histogram
    from Tools.helpers import make_bh

    # FIXME: decide on how to handle overflows.
    #for p in processes + ['signal']:
    #    print (p)
    #    print (h_tmp[p].values()[()])
    total = np.sum([h_tmp[p].values()[()] for p in processes + ['signal']], axis=0)
    total_int = np.round(total, 0).astype(int)

    pdata_hist = make_bh(
        sumw  = total,
        sumw2 = total,
        edges = axis.edges(),
    )

    # Now replace processes with BSM, if we have a histogram / values
    if bsm_hist:
        h_tmp_bsm['signal'] = bsm_hist
    else:
        h_tmp_bsm['signal'] = h_tmp['signal'].to_hist()

    fout = uproot.recreate(shape_file)

    # we write out the BSM histograms!
    for p in processes:
        fout[p] = h_tmp_bsm[p].to_hist()

    fout['signal'] = h_tmp_bsm['signal']
    fout['data_obs'] = pdata_hist  # this should work directly

    # Get the total yields to write into a data card
    #
    totals = {}

    for p in processes + ['signal']:
        totals[p] = h_tmp_bsm[p].values()[()].sum()

    totals['observation'] = pdata_hist.values().sum()

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
        # FIXME: update to boost histogram :(
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
                        central = bsm_vals.values()
                    else:
                        central = histogram[proc].integrate('dataset').values()[()]
                    val = np.nan_to_num(mag[0].counts, nan=1.0) * scale * central
                    val_h = make_bh(val, val, mag[0].edges)
                    incl_rel = sum(val_h.counts)/sum(central)
                    print ("Integrated systematic uncertainty %s for %s:"%(systematic, proc))
                    print (" - central prediction: %.2f"%sum(central))
                    print (" - relative uncertainty: %.2f"%incl_rel)

                    fout[proc+'_'+systematic+'Up']   = val_h
                    val = np.nan_to_num(mag[1].counts, nan=1.0) * scale * central
                    val_h = make_bh(val, val, mag[1].edges)
                    #incl_rel = sum(val_h.counts)/sum(central)
                    fout[proc+'_'+systematic+'Down'] = val_h
                else:
                    val = np.nan_to_num(mag[0].counts, nan=1.0) * scale * histogram[process].integrate('dataset').values()[()]
                    val_h = make_bh(val, val, mag[0].edges)
                    fout[proc+'_'+systematic] = val_h

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
