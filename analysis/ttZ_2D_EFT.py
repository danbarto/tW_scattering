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
from plots.helpers import finalizePlotDir

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
    base_dir = "/ceph/cms/store/user/dspitzba/NanoGEN/"
    plot_dir = "/home/users/sjeon/public_html/tW_scattering/ttZ_EFT_v2/"
    finalizePlotDir(plot_dir)

    # NOTE: use these /ceph/cms/store/user/dspitzba/ProjectMetis/TTZ_EFT_NLO_fixed_RunIISummer20_NanoGEN_NANO_v12/output_1.root samples?

    res = {}

    res['ttZ_NLO_2D'] = {
        'file': base_dir + "../ProjectMetis/TTZ_EFT_NLO_fixed_RunIISummer20_NanoGEN_NANO_v12/output_1.root",
        'events': NanoEventsFactory.from_root(base_dir + "../ProjectMetis/TTZ_EFT_NLO_fixed_RunIISummer20_NanoGEN_NANO_v12/output_1.root").events()
    }
    
    hp = HyperPoly(2)
    
    coordinates, ref_coordinates = get_coordinates_and_ref(res['ttZ_NLO_2D']['file'], is2D=True)
    hp.initialize( coordinates, ref_coordinates )
    
    pt_axis   = hist.Bin("pt",     r"p", 7, 100, 800)
    #pt_axis   = hist.Bin("pt",     r"p", 7, 0, 200)

    xsecs = {'ttZ_NLO': 0.930, 'ttZ_LO': 0.663, 'ttZ_NLO_2D':0.930}

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
        print ("SM point:", hp.eval(res[r]['coeff'], [0,0]))
        print ("BSM point:", hp.eval(res[r]['coeff'], [1,1]))
    
    
    # just an example.
    points = make_scan(operator='cpQM', C_min=-20, C_max=20, step=1, is2D=True)

    c_values = []
    for i in range(0,41):
        print (i-20, hp.eval(res['ttZ_NLO_2D']['coeff'], points[i]['point']))
        c_values.append(i-20)

    pred_matrix = np.array([ np.array(hp.eval(res['ttZ_NLO_2D']['coeff'],points[i]['point'])) for i in range(41) ])

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


    fig.savefig(plot_dir+'NLO_cpQM_scaling.pdf')
    fig.savefig(plot_dir+'NLO_cpQM_scaling.png')

    # just an example.
    points = make_scan(operator='cpt', C_min=-20, C_max=20, step=1, is2D=True)

    c_values = []
    for i in range(0,41):
        print (i-20, hp.eval(res['ttZ_NLO_2D']['coeff'], points[i]['point']))
        c_values.append(i-20)

    pred_matrix = np.array([ np.array(hp.eval(res['ttZ_NLO_2D']['coeff'],points[i]['point'])) for i in range(41) ])

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
    
    fig.savefig(plot_dir+'NLO_cpt_scaling.pdf')
    fig.savefig(plot_dir+'NLO_cpt_scaling.png')
