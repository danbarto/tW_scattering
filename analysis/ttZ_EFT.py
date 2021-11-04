'''
Load NanoGEN samples in NanoAOD event factory
- Compare LO vs NLO ttZ samples
- Compare on-shell vs off-shell ttZ samples
- Compare to a benchmark point to verify that reweighting works #FIXME
'''

import warnings
warnings.filterwarnings("ignore")

import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory
from coffea import hist

from Tools.reweighting import get_coordinates_and_ref, get_coordinates
from Tools.EFT_tools import make_scan
from Tools.HyperPoly import *
from Tools.gen import find_first_parent, get_lepton_fromW, get_neutrino_fromW, get_lepton_filter, get_lepton_from
from Tools.objects import choose
from Tools.helpers import get_four_vec_fromPtEtaPhiM

from processor.default_accumulators import dataset_axis

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from yahist import Hist1D, Hist2D

from reweighting_sanity_check import add_uncertainty

def get_Z(ev):
    gp = ev.GenPart
    return gp[((abs(gp.pdgId)==23) & ((gp.statusFlags & (1 << 13)) > 0))]

def get_leptonic_Z_sel(ev):
    gp = ev.GenPart
    gp_lep = gp[((abs(gp.pdgId)==11)|(abs(gp.pdgId)==13)|(abs(gp.pdgId)==15))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==23]
    return (ak.num(gp_lep_fromW)==2)

def get_leptonic_Z(ev):
    gp = ev.GenPart
    gp_lep = gp[((abs(gp.pdgId)==11)|(abs(gp.pdgId)==13)|(abs(gp.pdgId)==15))]
    gp_lep_fromW = gp_lep[abs(gp_lep.parent.pdgId)==23]
    leptons_from_Z = gp_lep_fromW[ak.num(gp_lep_fromW)==2]
    leptons_from_Z['p4'] = get_four_vec_fromPtEtaPhiM(leptons_from_Z, leptons_from_Z.pt, leptons_from_Z.eta, leptons_from_Z.phi, leptons_from_Z.mass)
    return choose(leptons_from_Z,2)

def get_LT(ev):
    return ak.sum(
        ak.concatenate([
            ak.from_regular(ev.GenMET.pt[:,np.newaxis]),
            ev.GenDressedLepton.pt[:,0:2]
        ], axis=1),
        axis=1,
    )

def histo_values(histo, weight):
    return histo[weight].sum('dataset').values(overflow='all')[()]

if __name__ == '__main__':

    # Load samples
    base_dir = "/nfs-7/userdata/dspitzba/"
    
    res = {}

    res['ttZ_NLO'] = {
        'file': base_dir + "merged_ttZ_NLO.root",
        'events': NanoEventsFactory.from_root(base_dir + "merged_ttZ_NLO.root").events()
    }

    res['ttZ_LO'] = {
        'file': base_dir + "merged_ttZ_LO.root",
        'events': NanoEventsFactory.from_root(base_dir + "merged_ttZ_LO.root").events()
    }
    
    #res['ttll_LO'] = {
    #    'file': base_dir + "merged_ttll_LO.root",
    #    'events': NanoEventsFactory.from_root(base_dir + "merged_ttll_LO.root").events()
    #}

    # Get HyperPoly parametrization.
    # this is sample independent as long as the coordinates are the same.
    
    hp = HyperPoly(2)
    
    coordinates, ref_coordinates = get_coordinates_and_ref(res['ttZ_NLO']['file'])
    hp.initialize( coordinates, ref_coordinates )
    
    pt_axis   = hist.Bin("pt",     r"p", 7, 100, 800)
    #pt_axis   = hist.Bin("pt",     r"p", 7, 0, 200)

    xsecs = {'ttZ_NLO': 0.930, 'ttZ_LO': 0.663}

    for r in res:

        print (r)

        tree    = uproot.open(res[r]['file'])["Events"]
        ev      = res[r]['events']
        name    = r

        weights = [ x.replace('LHEWeight_','') for x in tree.keys() if x.startswith('LHEWeight_c') ]

        trilep = (ak.num(ev.GenDressedLepton)==3)
        #trilep = (ak.num(ev.GenDressedLepton)>=0)

        res[name]['selection'] = trilep
        res[name]['hist'] = hist.Hist("met", dataset_axis, pt_axis)

        res[name]['hist'].fill(
            dataset='stat',
            pt = get_LT(ev[trilep]),
            weight = ev[trilep].genWeight/sum(ev.genWeight)
        )

        res[name]['central'] = res[name]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][0]
        res[name]['w2'] = res[name]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][1]

        for w in weights:
        
            res[name]['hist'].fill(
                dataset=w,
                pt = get_LT(ev[trilep]),
                #pt = ev[trilep].GenMET.pt,
                weight=xsecs[r]*getattr(ev[trilep].LHEWeight, w)*ev[trilep].genWeight/sum(ev.genWeight)
            )

        res[r]['coeff'] = hp.get_parametrization( [histo_values(res[name]['hist'], w) for w in weights] )

        # points are given as [ctZ, cpt, cpQM, cpQ3, ctW, ctp]
        print ("SM point:", hp.eval(res[r]['coeff'], [0,0,0,0,0,0]))
        print ("BSM point:", hp.eval(res[r]['coeff'], [2,0,0,0,0,0]))


    
    
    ## k-factor
    # because there are differences of the EFT effects at LO and NLO we are not arriving back at the same SM point when weighting back?
    # Does that make sense?
    # Yes, because we artificially fix the LO and NLO x-secs at our benchmark point to the same value the way we set weights.
    # So, we use a k factor the fix the LO and NLO x-secs at the SM point to the same value.

    # LO x-sec at ref point: 0.663 +- 0.00256 pb
    # NLO x-sec at ref point: 0.930 +- 0.00132 pb

    # FIXME negative weights! what to do about those?
    # is it a problem for the histogram based reweighting???
    # or is it a deeper problem?
    # or no problem at all?

    k = sum(res['ttZ_NLO']['hist']['ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p_nlo'].sum('dataset').values(overflow='all')[()])/sum(res['ttZ_LO']['hist']['ctZ_0p_cpt_0p_cpQM_0p_cpQ3_0p_ctW_0p_ctp_0p'].sum('dataset').values(overflow='all')[()])
    edges   = res['ttZ_NLO']['hist'].axis('pt').edges(overflow='all')

    print ("Found a k-factor of %.3f"%k)

    fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)
    
    hep.cms.label(
        "Preliminary",
        data=True,
        #year=2018,
        lumi=60.0,
        loc=0,
        ax=ax,
    )

    num     = hp.eval(res['ttZ_NLO']['coeff'], [0,0,0,0,0,0]), res['ttZ_NLO']['w2']
    denom   = hp.eval(res['ttZ_LO']['coeff'], [0,0,0,0,0,0])*k, res['ttZ_LO']['w2']

    num_vals = res['ttZ_NLO']['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()]
    denom_vals = res['ttZ_LO']['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()]
    denom_unc_rel = np.sqrt(denom_vals[1])/denom_vals[0]
    denom_unc_abs = denom_unc_rel*denom[0]

    num_unc_rel = np.sqrt(num_vals[1])/num_vals[0]
    num_unc_abs = num_unc_rel*num[0]

    num_h   = Hist1D.from_bincounts(num[0], edges, num_unc_abs, )
    denom_h = Hist1D.from_bincounts(denom[0], edges, denom_unc_abs, )
    ratio   = num_h.divide(denom_h)

    hep.histplot(
        num[0],
        edges,
        #w2=num[1],
        histtype="step",
        label=[r'ttZ (NLO), $c_{\varphi t}=0$'],
        color=['red'],
        ax=ax)

    hep.histplot(
        denom[0],
        edges,
        #w2=denom[1],
        histtype="step",
        linestyle='--',
        #linewidth=1.5,
        label=[r'ttZ (LO), $c_{\varphi t}=0$'],
        color=['red'],
        ax=ax)

    hep.histplot(
        ratio.counts,
        edges,
        #w2=ratio.errors,
        histtype="errorbar",
        color='red',
        ax=rax,
    )

    add_uncertainty(denom[0], denom_unc_abs, ax, edges, ratio=False, color='red', hatch='///')
    add_uncertainty(num[0], np.sqrt(num[1]), ax, edges, ratio=False, color='red', hatch="\\\\")
    add_uncertainty(ratio.counts, ratio.errors, rax, edges, ratio=False, color='red', hatch="|||")


    num     = hp.eval(res['ttZ_NLO']['coeff'], [0,4,0,0,0,0]), res['ttZ_NLO']['w2']
    denom   = hp.eval(res['ttZ_LO']['coeff'], [0,4,0,0,0,0])*k, res['ttZ_LO']['w2']

    denom_unc_abs = denom_unc_rel*denom[0]

    num_unc_abs = num_unc_rel*num[0]

    num_h   = Hist1D.from_bincounts(num[0], edges, num_unc_abs, )
    denom_h = Hist1D.from_bincounts(denom[0], edges, denom_unc_abs, )
    ratio   = num_h.divide(denom_h)


    hep.histplot(
        num[0],
        edges,
        #w2=num[1],
        histtype="step",
        label=[r'ttZ (NLO), $c_{\varphi t}=4$'],
        color=['green'],
        ax=ax)

    hep.histplot(
        denom[0],
        edges,
        #w2=denom[1],
        histtype="step",
        linestyle='--',
        #linewidth=1.5,
        label=[r'ttZ (LO), $c_{\varphi t}=4$'],
        color=['green'],
        ax=ax)

    hep.histplot(
        ratio.counts,
        edges,
        #w2=ratio.errors,
        histtype="errorbar",
        color='green',
        ax=rax,
    )

    add_uncertainty(denom[0], denom_unc_abs, ax, edges, ratio=False, color='green', hatch='///')
    add_uncertainty(num[0], np.sqrt(num[1]), ax, edges, ratio=False, color='green', hatch="\\\\")
    add_uncertainty(ratio.counts, ratio.errors, rax, edges, ratio=False, color='green', hatch="///")

    rax.set_ylim(0,1.99)
    rax.set_xlabel(r'$p_{T}(l_1)+p_{T}(l_2)+p_{T}^{miss}\ (GeV)$')
    rax.set_ylabel(r'Ratio')
    ax.set_ylabel(r'Events')
    ax.set_yscale('log')
    ax.set_ylim(0.00003,0.09)

    ax.legend()

    x1, y1 = [-100, 1000], [1, 1]
    plt.xlim(0, 900)
    #plt.ylim(-2, 8)
    plt.plot(x1, y1, marker = 'o', color='black')

    plt.show()
    
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LT_cpt.png')
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LT_cpt.pdf')

    # FIXME WIP
    ## E^2 scaling for ttZ
    # just an example.
    points = make_scan(operator='cpQM', C_min=-20, C_max=20, step=1)

    c_values = []
    for i in range(0,41):
        print (i-20, hp.eval(res['ttZ_LO']['coeff'], points[i]['point']))
        c_values.append(i-20)

    pred_matrix = np.array([ np.array(hp.eval(res['ttZ_LO']['coeff'],points[i]['point'])) for i in range(41) ])

    fig, ax = plt.subplots()
    hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60.0+41.5+35.9,
        loc=0,
        ax=ax,
    )
    
    plt.plot(c_values, np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[20,:]), label=r'inclusive', c='green')
    plt.plot(c_values, np.sum(pred_matrix[:,7:], axis=1)/np.sum(pred_matrix[20,7:]), label=r'$L_{T} \geq 700\ GeV$', c='blue')
    
    plt.xlabel(r'$C_{\varphi Q}^{-}$')
    plt.ylabel(r'$\sigma/\sigma_{SM}$')
    plt.legend()

    ax.set_ylim(0,10)
    
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LO_cpQM_scaling.pdf')
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LO_cpQM_scaling.png')

    # just an example.
    points = make_scan(operator='cpt', C_min=-20, C_max=20, step=1)

    c_values = []
    for i in range(0,41):
        print (i-20, hp.eval(res['ttZ_LO']['coeff'], points[i]['point']))
        c_values.append(i-20)

    pred_matrix = np.array([ np.array(hp.eval(res['ttZ_LO']['coeff'],points[i]['point'])) for i in range(41) ])

    fig, ax = plt.subplots()
    
    hep.cms.label(
        "Work in progress",
        data=True,
        #year=2018,
        lumi=60.0+41.5+35.9,
        loc=0,
        ax=ax,
    )
    
    plt.plot(c_values, np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[20,:]), label=r'inclusive', c='green')
    plt.plot(c_values, np.sum(pred_matrix[:,7:], axis=1)/np.sum(pred_matrix[20,7:]), label=r'$L_{T} \geq 700\ GeV$', c='blue')
    
    plt.xlabel(r'$C_{\varphi t}$')
    plt.ylabel(r'$\sigma/\sigma_{SM}$')
    plt.legend()

    ax.set_ylim(0,10)
    
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LO_cpt_scaling.pdf')
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/LO_cpt_scaling.png')

