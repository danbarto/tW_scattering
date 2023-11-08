import os
import re
from coffea import hist
import uproot
import numpy as np
from analysis.Tools.dataCard import *
from analysis.Tools.helpers import make_bh
from analysis.Tools.defaults import variations_jet_all_list
import boost_histogram as bh

wildcard = re.compile(r'.', re.UNICODE)

def get_norms(dataset, samples, mapping, name='pdf', weight='LHEPdfWeight'):
    '''
    this function does not need the actual histograms but the stored meta data
    from the samples database
    '''
    central = sum([ samples.db[s].xsec/samples.db[s].sumWeight for s in mapping[dataset] ])
    total   = sum([ samples.db[s].xsec/np.array(getattr(samples.db[s], weight)[:101]) for s in mapping[dataset] ])  # some PDF sets don't include the alpha_S variations...
    norms   = {f'{name}_{i}': x for i,x in enumerate(total/central)}
    return norms

def get_pdf_unc(
        histogram,
        process,
        eft_point,
        rebin=None,
        hessian=True,
        quiet=True,
        overflow='all',
        norms=None,
        select_indices=False,
):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    '''
    if select_indices:
        if isinstance(select_indices, int):
            indices = [select_indices]
        else:
            indices = select_indices
    else:
        indices = range(1,101)
    if not hessian:
        print ("Can't handle mc replicas.")
        return False

    # now get the actual values
    tmp_central = histogram[(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    pdf_unc = np.zeros_like(central)

    for i in indices:
        tmp_variation = histogram[(process, eft_point, 'central', f'pdf_{i}')].sum('EFT', 'systematic', 'prediction').copy()
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
    
        #print (central)

    return  up_hist, down_hist

def get_scale_unc(histogram, process, eft_point,
                  rebin=None,
                  quiet=True,
                  overflow='all',
                  norms=None,
                  indices=[0,1,3,5,7,8],
                  ):
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
    tmp_central = histogram[(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]

    scale_unc = np.zeros_like(central)
    for i in indices:
        '''
        Using the full envelope.
        Don't know how to make a sensible envelope of up/down separately,
        without getting vulnerable to weird one-sided uncertainties.
        '''

        norm = norms[f'scale_{i}'] if (norms is not None) else 1
        #if norms:
        #    proc_norm = norms[f'scale_{i}']

        tmp_variation = histogram[(process, eft_point, 'central', f'scale_{i}')].sum('EFT', 'systematic', 'prediction').copy()
        if rebin:
            tmp_variation = tmp_variation.rebin(rebin.name, rebin)

        tmp_var = tmp_variation[process].sum('dataset').values(overflow=overflow)[()]
        #print (i, sum(tmp_var)/sum(central))

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


def get_ISR_unc(histogram, process, eft_point, rebin=None, quiet=True, overflow='all', norm=None):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties

    From auto documentation of NanoAODv8

    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    --> take 0, 2 for ISR variations
    '''
    # now get the actual values
    tmp_central = histogram[(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]

    for i in [0,2]:
        tmp_variation = histogram[(process, eft_point, 'central', f'PS_{i}')].sum('EFT', 'systematic', 'prediction').copy()
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


def get_FSR_unc(histogram, process, eft_point, rebin=None, quiet=True, overflow='all', norm=None):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    
    From auto documentation of NanoAODv8
    
    PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
    --> take 1, 3 for FSR variations
    '''
    # now get the actual values
    tmp_central = histogram[(process, eft_point, 'central', 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
    central = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    
    for i in [1,3]:
        tmp_variation = histogram[(process, eft_point, 'central', f'PS_{i}')].sum('EFT', 'systematic', 'prediction').copy()
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

def get_unc(
        histogram,
        process,
        unc,
        eft_point,
        rebin=None,
        quiet=True,
        overflow='all',
        prediction='central',
        symmetric=False,
):
    '''
    takes a coffea output, histogram name, process name and bins if histogram should be rebinned.
    returns a histogram that can be used for systematic uncertainties
    '''
    # now get the actual values
    tmp_central = histogram[(process, eft_point, prediction, 'central')].sum('EFT', 'systematic', 'prediction').copy()
    if not symmetric:
        tmp_up      = histogram[(process, eft_point, prediction, unc+'_up')].sum('EFT', 'systematic', 'prediction').copy()
        tmp_down    = histogram[(process, eft_point, prediction, unc+'_down')].sum('EFT', 'systematic', 'prediction').copy()
    else:
        tmp_up      = histogram[(process, eft_point, prediction, unc)].sum('EFT', 'systematic', 'prediction').copy()

    
    if rebin:
        tmp_central = tmp_central.rebin(rebin.name, rebin)
        tmp_up      = tmp_up.rebin(rebin.name, rebin)
        if not symmetric:
            tmp_down    = tmp_down.rebin(rebin.name, rebin)
        
    central  = tmp_central[process].sum('dataset').values(overflow=overflow)[()]
    up_unc   = tmp_up[process].sum('dataset').values(overflow=overflow)[()]
    if not symmetric:
        down_unc = tmp_down[process].sum('dataset').values(overflow=overflow)[()]
    edges    = tmp_central[process].sum('dataset').axes()[0].edges(overflow=overflow)

    up_hist = make_bh(up_unc/central, up_unc/central, edges)
    if not symmetric:
        down_hist = make_bh(down_unc/central, down_unc/central, edges)
    else:
        down_hist = make_bh(central/up_unc, central/up_unc, edges)

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

def get_systematics(histogram, year, eft_point,
                    correlated=False,
                    signal=True,
                    overflow='all',
                    samples=None,
                    mapping=None,
                    rebin=None,
                    ):
    if correlated:
        year = "cor"
    systematics = []

    all_processes = ['TTW', 'TTZ', 'TTH', 'rare', 'diboson']
    if signal: all_processes += ['signal']

    for proc in all_processes:
        systematics += [
            #('jes_%s'%year,     get_unc(histogram, proc, 'jesTotal',  eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            #('jer_%s'%year,     get_unc(histogram, proc, 'jer',  eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ('b_%s'%year,       get_unc(histogram, proc, 'b',    eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ('light_%s'%year,   get_unc(histogram, proc, 'l',    eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ('mu_%s'%year,      get_unc(histogram, proc, 'mu',   eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ('ele_%s'%year,     get_unc(histogram, proc, 'ele',  eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ('PU',              get_unc(histogram, proc, 'PU',   eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
        ]

        for var in variations_jet_all_list:
            systematics += [
                (f'{var}_{year}',     get_unc(histogram, proc, var,  eft_point, rebin=rebin, overflow=overflow, quiet=True), proc),
            ]

    for proc in ['TTW', 'TTZ', 'TTH', 'rare']:  # FIXME extend to all MC driven estimates. diboson is broken because of weight length mismatch of ZZ sample...
        systematics += [
            #('pdf', get_pdf_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow, norms=get_norms(proc, samples, mapping, name='pdf', weight='LHEPdfWeight')), proc),
            ('FSR', get_FSR_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow), proc),
            ('ISR', get_ISR_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow), proc),
            ('scale_%s'%proc, get_scale_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow, norms=get_norms(proc, samples, mapping, name='scale', weight='LHEScaleWeight')), proc),
        ]
        for i in range(1,101):
            systematics.append(
                (f'pdf_{i}', get_pdf_unc(histogram, proc, eft_point,
                                         rebin=rebin, overflow=overflow,
                                         select_indices = i,
                                         norms=get_norms(proc, samples, mapping, name='pdf', weight='LHEPdfWeight')), proc)
            )

    systematics += [
        ('fake_el_closure', get_unc(histogram, wildcard, 'fake_el_closure', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True), 'nonprompt'),
        ('fake_mu_closure', get_unc(histogram, wildcard, 'fake_mu_closure', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True), 'nonprompt'),
        ('fake_el', get_unc(histogram, wildcard, 'fake_el', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True), 'nonprompt'),
        ('fake_mu', get_unc(histogram, wildcard, 'fake_mu', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True), 'nonprompt'),
        ('fake_mu_pt1', get_unc(histogram, wildcard, 'fake_mu_pt1', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_mu_pt2', get_unc(histogram, wildcard, 'fake_mu_pt2', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_el_pt1', get_unc(histogram, wildcard, 'fake_el_pt1', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_el_pt2', get_unc(histogram, wildcard, 'fake_el_pt2', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_mu_be1', get_unc(histogram, wildcard, 'fake_mu_be1', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_mu_be2', get_unc(histogram, wildcard, 'fake_mu_be2', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_el_be1', get_unc(histogram, wildcard, 'fake_el_be1', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
        ('fake_el_be2', get_unc(histogram, wildcard, 'fake_el_be2', eft_point, prediction='np_est_data', rebin=rebin, overflow=overflow, quiet=True, symmetric=True), 'nonprompt'),
    ]

    systematics += [
        #('signal_norm', 1.10, 'signal'),
        ('signal_pdf_norm', 1.012, 'signal'),
        ('signal_scale_norm', 1.17, 'signal'),
        #('ttz_norm', 1.10, 'TTZ'),
        ('ttz_scale_norm', 1.113, 'TTZ'),
        ('ttz_pdf_norm', 1.028, 'TTZ'),
        ('ttz_as_norm', 1.028, 'TTZ'),
        #('ttw_norm', 1.15, 'TTW'),
        ('ttw_pdf_norm', 1.01, 'TTW'),
        ('ttw_scale_norm', 1.11, 'TTW'),  # NOTE could be asymetrized.
        #('tth_norm', 1.15, 'TTH'),
        ('tth_pdf_norm', 1.030, 'TTH'),
        ('tth_as_norm', 1.020, 'TTH'),
        #('tth_scale_norm', [0.915, 1.058], 'TTH'),  # NOTE this is not very secure
        ('tth_scale_norm', 1.092, 'TTH'),
        ('rare_norm', 1.20, 'rare'),
        ('diboson_norm', 1.20, 'diboson'),
        #
        #('nonprompt_norm', 1.30, 'nonprompt'),
        ('chargeflip_norm', 1.20, 'chargeflip'),
        ('conv_norm', 1.20, 'conv')
    ]
    return systematics

def add_signal_systematics(histogram, year, eft_point,
                           correlated=False,
                           systematics=[],
                           proc='signal',
                           overflow='all',
                           samples=None,
                           mapping=None,
                           rebin=None,
                           ):
    if correlated:
        year = "cor"
    systematics += [
        #('jes_%s'%year,     get_unc(histogram, proc, 'jesTotal',  eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        #('jer_%s'%year,     get_unc(histogram, proc, 'jer',  eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        ('b_%s'%year,       get_unc(histogram, proc, 'b',    eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        ('light_%s'%year,   get_unc(histogram, proc, 'l',    eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        ('mu_%s'%year,      get_unc(histogram, proc, 'mu',   eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        ('ele_%s'%year,     get_unc(histogram, proc, 'ele',  eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        ('PU',              get_unc(histogram, proc, 'PU',   eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
        #('pdf',             get_pdf_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow, norms=get_norms(proc, samples, mapping, name='pdf', weight='LHEPdfWeight')), "signal"),
        ('scale_signal',    get_scale_unc(histogram, proc, eft_point, rebin=rebin, overflow=overflow, norms=get_norms(proc, samples, mapping, name='scale', weight='LHEScaleWeight'), indices=[0,1,3,4,6,7]), "signal"),
    ]
    for i in range(1,101):
        systematics.append(
            (f'pdf_{i}', get_pdf_unc(histogram, proc, eft_point,
                                        rebin=rebin, overflow=overflow,
                                        select_indices = i,
                                        norms=get_norms(proc, samples, mapping, name='pdf', weight='LHEPdfWeight')), "signal")
        )
    for var in variations_jet_all_list:
        systematics += [
            (f'{var}_{year}',     get_unc(histogram, proc, var,  eft_point, rebin=rebin, overflow=overflow, quiet=True), "signal"),
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
        data=False,
):
    
    '''
    make a card file from histograms
    histogams: python dictionary with 2D histograms: dataset axis and feature axis
    signal_hist overrides the default signal histogram if provided
    '''

    if not quiet:
        print ("Writing cards now")
    card_dir = os.path.abspath(os.path.expandvars('data/cards/')) + '/'
    print ("#### CARD DIR ####")
    print (card_dir)
    if not os.path.isdir(card_dir):
        os.makedirs(card_dir)
    
    data_card = card_dir+ext+'_card.txt'
    shape_file = card_dir+ext+'_shapes.root'

    processes = [ p for p in histograms.keys() if p != 'signal']

    # make a copy of the histograms
    h_tmp = {}
    h_tmp_bsm = {}
    for p in processes + ['signal']:
        # SM values --> sum of this will be the observation for the expected limits
        h_tmp[p] = histograms[p].copy()
        h_tmp[p].scale(scales, axis='dataset')  # scale according to the processes
        h_tmp[p] = h_tmp[p].sum('dataset')  # reduce to 1D histogram

        # BSM values -> expectation for both expected and observed limits
        h_tmp_bsm[p] = histograms[p].copy()
        h_tmp_bsm[p].scale(scales, axis='dataset')  # scale according to the processes
        h_tmp_bsm[p].scale(bsm_scales, axis='dataset')  # scale according to the processes
        h_tmp_bsm[p] = h_tmp_bsm[p].sum('dataset')  # reduce to 1D histogram

    if data:
        print("Found observation histogram")
        h_tmp["observation"] = data.copy()
        h_tmp["observation"] = h_tmp["observation"]#.sum('dataset')  # reduce to 1D histogram

    # get an axis
    axis = h_tmp["signal"].axes()[0]

    # make observation boost histogram
    from analysis.Tools.helpers import make_bh

    # FIXME: decide on how to handle overflows.
    total = np.sum([h_tmp[p].values()[()] for p in processes + ['signal']], axis=0)
    total_int = np.round(total, 0).astype(int)

    # this is how we get the expected results
    if not blind:
        pdata_hist = make_bh(
            sumw  = h_tmp["observation"].values()[()],
            sumw2 = h_tmp["observation"].values()[()],
            edges = axis.edges(),
        )
    else:
        pdata_hist = make_bh(
            sumw  = total,
            sumw2 = total,
            edges = axis.edges(),
        )

    fout = uproot.recreate(shape_file)

    # Replace signal histogram with BSM histogram, if we have a histogram
    if bsm_hist:
        if isinstance(bsm_hist, bh.Histogram):
            h_tmp_bsm['signal'] = bsm_hist
            out_hist_tmp = h_tmp_bsm['signal']
        else:
            # it is not a boost histogram for both SS and trilep
            h_tmp_bsm['signal'] = bsm_hist.sum('dataset')
            out_hist_tmp = h_tmp_bsm['signal'].to_hist()

        out_hist_tmp.view().value = np.maximum(out_hist_tmp.view().value, 0.01*np.ones_like(out_hist_tmp.view().value))
        fout['signal'] = out_hist_tmp
    else:
        h_tmp_bsm['signal'] = h_tmp['signal']
        out_hist_tmp = h_tmp['signal'].to_hist()
        out_hist_tmp.view().value = np.maximum(out_hist_tmp.view().value, 0.01*np.ones_like(out_hist_tmp.view().value))
        fout['signal'] = out_hist_tmp

    # we write out the BSM histograms!
    for p in processes:
        out_hist_tmp = h_tmp_bsm[p].to_hist()
        out_hist_tmp.view().value = np.maximum(out_hist_tmp.view().value, 0.01*np.ones_like(out_hist_tmp.view().value))
        fout[p] = out_hist_tmp

    #fout['signal'] = h_tmp_bsm['signal'].to_hist()
    fout['data_obs'] = pdata_hist  # this should work directly

    # Get the total _expected_ yields to write into a data card
    totals = {}

    for p in processes + ['signal']:
        totals[p] = np.maximum(h_tmp_bsm[p].values()[()], 0.01*np.ones_like(h_tmp_bsm[p].values()[()])).sum()

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

    card.specifyObservation('Bin0', totals['observation'])
    
    ## add the uncertainties
    if systematics:
        for systematic, mag, proc in systematics:
            if isinstance(mag, type(())):
                # systematic up/down histograms are given as relative uncertainties (values 1+/-sigma)
                # for the data cards these need to be scaled to the expected values
                # REMINDER: h_tmp_bsm are the expected histograms
                if proc in scales:
                    scale = scales[proc]
                else:
                    scale = 1
                if proc in bsm_scales:
                    scale *= bsm_scales[proc]

                if not card.checkUncertaintyExists(systematic):
                    card.addUncertainty(systematic, 'shape')
                    print ("Adding shape uncertainty %s for process %s."%(systematic, proc))

                if len(mag)>1:
                    central = h_tmp_bsm[proc].values()[()]  # get BSM scaled prediction

                    val = np.nan_to_num(mag[0].values(), nan=1.0) * central
                    val = np.maximum(val, 0.02*np.ones_like(val))
                    val_h = make_bh(val, val, mag[0].axes[0].edges)
                    incl_rel = sum(val_h.values())/sum(central)
                    print ("Integrated systematic uncertainty %s for %s:"%(systematic, proc))
                    print (" - central prediction: %.2f"%sum(central))
                    print (" - relative uncertainty: %.2f"%incl_rel)

                    fout[proc+'_'+systematic+'Up']   = val_h
                    val = np.nan_to_num(mag[1].values(), nan=1.0) * central
                    val = np.maximum(val, np.zeros_like(val))
                    val_h = make_bh(val, val, mag[1].axes[0].edges)
                    fout[proc+'_'+systematic+'Down'] = val_h
                else:
                    val = np.nan_to_num(mag[0].values(), nan=1.0) * h_tmp_bsm[proc].values()[()]
                    val = np.maximum(val, 0.02*np.ones_like(val))
                    val_h = make_bh(val, val, mag[0].axes[0].edges)
                    fout[proc+'_'+systematic] = val_h

                card.specifyUncertainty(systematic, 'Bin0', proc, 1)

            elif isinstance(mag, type([])):
                # FIXME this is not implemented yet. can't have asymmetric lnNs in my card file writer
                if not card.checkUncertaintyExists(systematic):
                    card.addUncertainty(systematic, 'lnN')
                    print ("Adding lnN uncertainty %s for process %s."%(systematic, proc))
                card.specifyUncertainty(systematic, 'Bin0', proc, (mag[0], mag[1]))
            else:
                if not card.checkUncertaintyExists(systematic):
                    card.addUncertainty(systematic, 'lnN')
                    print ("Adding lnN uncertainty %s for process %s."%(systematic, proc))
                card.specifyUncertainty(systematic, 'Bin0', proc, mag)
            
    fout.close()

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
