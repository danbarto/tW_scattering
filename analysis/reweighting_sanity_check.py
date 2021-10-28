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

    res['ttZ_LO'] = {
        'file': base_dir + "merged_ttZ_LO.root",
        'events': NanoEventsFactory.from_root(base_dir + "merged_ttZ_LO.root").events()
    }

    res['ttZ_LO_ctz5'] = {
        'file': base_dir + "merged_ttZ_LO_ctz5.root",
        'events': NanoEventsFactory.from_root(base_dir + "merged_ttZ_LO_ctz5.root").events()
    }
    
    # Get HyperPoly parametrization.
    # this is sample independent as long as the coordinates are the same.
    
    hp = HyperPoly(2)
    
    coordinates, ref_coordinates = get_coordinates_and_ref(res['ttZ_LO']['file'])
    hp.initialize( coordinates, ref_coordinates )
    
    pt_axis   = hist.Bin("pt",     r"p", 7, 100, 800)
    #pt_axis   = hist.Bin("pt",     r"p", 7, 0, 200)

    # this needs to be the x-sec at the ref point of course!!!
    xsecs = {'ttZ_LO': 0.663, 'ttZ_LO_ctz5': 1.588}

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
            weight = xsecs[r]*ev[trilep].genWeight/sum(ev.genWeight)
        )

        res[name]['central'] = res[name]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][0]
        res[name]['w2'] = res[name]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][1]

        if r == 'ttZ_LO':

            for w in weights:
            
                res[name]['hist'].fill(
                    dataset=w,
                    pt = get_LT(ev[trilep]),
                    #pt = ev[trilep].GenMET.pt,
                    weight=xsecs[r]*getattr(ev[trilep].LHEWeight, w)*ev[trilep].genWeight/sum(ev.genWeight)
                )

            res[r]['coeff'] = hp.get_parametrization( [histo_values(res[name]['hist'], w) for w in weights] )



    edges   = res['ttZ_LO']['hist'].axis('pt').edges(overflow='all')

    fig, (ax, rax) = plt.subplots(2,1,figsize=(10,10), gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05}, sharex=True)
    
    hep.cms.label(
        "Preliminary",
        data=True,
        #year=2018,
        lumi=60.0,
        loc=0,
        ax=ax,
    )

    num     = res['ttZ_LO_ctz5']['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()]
    denom   = hp.eval(res['ttZ_LO']['coeff'], [5,0,0,0,0,0]), res['ttZ_LO']['w2']
    ratio   = num[0]/denom[0]

    hep.histplot(
        num[0],
        edges,
        #w2=num[1],
        histtype="step",
        label=[r'ttZ (LO), $c_{tZ}=5$'],
        color=['red'],
        ax=ax)

    hep.histplot(
        denom[0],
        edges,
        #w2=denom[1],
        histtype="step",
        linestyle='--',
        #linewidth=1.5,
        label=[r'ttZ (LO), $c_{tZ}=5$, reweighted'],
        color=['red'],
        ax=ax)

    hep.histplot(
        ratio,
        edges,
        #w2=ratio.errors,
        histtype="errorbar",
        color='black',
        ax=rax,
    )

    rax.set_ylim(0,1.99)
    rax.set_xlabel(r'$L_T\ (l_1+l_2+p_{T}^{miss})\ (GeV)$')
    rax.set_ylabel(r'Ratio')
    ax.set_ylabel(r'Events')
    ax.set_yscale('log')

    ax.legend()

    plt.show()
    
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/sanity_check.png')
    fig.savefig('/home/users/dspitzba/public_html/tW_scattering/ttZ_EFT/sanity_check.pdf')

