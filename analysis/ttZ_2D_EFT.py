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
from Tools.EFT_tools import make_scan_2D
from Tools.HyperPoly import *
from Tools.gen import find_first_parent, get_lepton_fromW, get_neutrino_fromW, get_lepton_filter, get_lepton_from
from Tools.objects import choose
from Tools.helpers import get_four_vec_fromPtEtaPhiM
from plots.helpers import finalizePlotDir

from processor.default_accumulators import dataset_axis

import matplotlib.pyplot as plt
from matplotlib import cm
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
    base_dir = "/home/users/sjeon/ttw/CMSSW_10_6_19/src/"
    plot_dir = "/home/users/sjeon/public_html/tW_scattering/ttZ_EFT_2D/"
    finalizePlotDir(plot_dir)

    res = {}

    res['ttZ_NLO_2D'] = {
        'file': base_dir + "ttZ_EFT_NLO.root",
        'events': NanoEventsFactory.from_root(base_dir + "ttZ_EFT_NLO.root").events()
    }

    res['ttZ_LO_2D'] = {
        'file': base_dir + "ttZ_EFT_LO.root",
        'events': NanoEventsFactory.from_root(base_dir + "ttZ_EFT_LO.root").events()
    }
    
    is2D = {
        'ttZ_NLO_2D':True,
        'ttZ_LO_2D':True,
    }

    results = {}


    # Get HyperPoly parametrization.
    # this is sample independent as long as the coordinates are the same.

    xsecs = {'ttZ_NLO_2D': 0.930, 'ttZ_LO_2D': 0.663}
    
    for r in res:

        print (r)

        hp = HyperPoly(2)

        tree    = uproot.open(res[r]['file'])["Events"]

        coordinates, ref_coordinates = get_coordinates_and_ref(res[r]['file'],is2D[r])
        hp.initialize( coordinates, ref_coordinates )

        weights = [ x.replace('LHEWeight_','') for x in tree.keys() if x.startswith('LHEWeight_c') ]

        # define selections
        trilep = (ak.num(res[r]['events'].GenDressedLepton)==3)
        ev = res[r]['events'][trilep] 
        LTlim = 700
        ptZlim = 400
        LTmask = (get_LT(ev) >= LTlim)
        ptZmask = ak.flatten(get_Z(ev).pt >= ptZlim)
      
        print('out of',len(ev),'events total, there are...')
        print(len(ev[LTmask]), 'events with LT>%d'%LTlim)
        print(len(ev[ptZmask]), 'events with ptZ>%d'%ptZlim) 
        
        # fill histogram
        pt_axis = hist.Bin("pt", r"p", 1, 100, 800)
        res[r]['hist'] = hist.Hist("met", dataset_axis, pt_axis)

        res[r]['hist'].fill(
            dataset='stat',
            pt = get_LT(ev),
            weight = ev.genWeight/sum(ev.genWeight)
        )
        
        res[r]['central'] = res[r]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][0]
        res[r]['w2'] = res[r]['hist']['stat'].sum('dataset').values(overflow='all', sumw2=True)[()][1]

        for w in weights:
            res[r]['hist'].fill(
                dataset=w,
                pt = get_LT(ev),
                weight=xsecs[r]*getattr(ev.LHEWeight, w)*ev.genWeight/sum(ev.genWeight)
            )

        # calculate coefficients
        allvals = [histo_values(res[r]['hist'], w) for w in weights]
        res[r]['coeff'] = hp.get_parametrization(allvals)

        # print sample SM/BSM point
        # points are given as [ctZ, cpt, cpQM, cpQ3, ctW, ctp]
        if is2D[r]:
            print ("SM point:", hp.eval(res[r]['coeff'], [0,0]))
            print ("BSM point:", hp.eval(res[r]['coeff'], [1,1]))
        else:
            print ("SM point:", hp.eval(res[r]['coeff'], [0,0,0,0,0,0]))
            print ("BSM point:", hp.eval(res[r]['coeff'], [2,0,0,0,0,0]))

        # get c axis points
        points = make_scan_2D(operators=['cpt','cpQM'], C_min=-20, C_max=20, step=1)
        cpt_vals = []
        cpQM_vals = []
        for i in range(0,41):
            cpt_vals.append(i-20)
            cpQM_vals.append(i-20)
 
        # calculate and store results 
        pred_matrix = [ [ hp.eval(res[r]['coeff'],points[i1][i2]['point']) for i1 in range(41) ] for i2 in range(41)]

        results[r] = {}
        results[r]['inc'] = np.sum(pred_matrix, axis=2)/np.sum(pred_matrix[20][20])
        #results[r][c]['LTtail'] = np.sum(pred_matrix[:,LTmask], axis=1)/np.sum(pred_matrix[20,LTmask])
        #results[r][c]['ptZtail'] = np.sum(pred_matrix[:,ptZmask], axis=1)/np.sum(pred_matrix[20,ptZmask])

        print(results[r]['inc'])
 
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #fig, ax = plt.subplots()
 
        ax.plot_surface(cpt_vals, cpQM_vals, results[r]['inc'], cmap=cm.coolwarm)
        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        #plt.zlabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()
     
        ax.set_ylim(0,10)
     
        fig.savefig(plot_dir+r[4:]+'_scaling.pdf')
        fig.savefig(plot_dir+r[4:]+'_scaling.png')
