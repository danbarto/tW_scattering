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
    base_dir = "/home/users/sjeon/ttw/CMSSW_10_6_19/src/"
    #base_dir = "/ceph/cms/store/user/sjeon/NanoGEN/"
    plot_dir = "/home/users/sjeon/public_html/tW_scattering/ttZ_EFT_v2/"
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
        
        # calculate coefficients
        allvals = [getattr(ev.LHEWeight, w) for w in weights]
        res[r]['coeff'] = hp.get_parametrization(allvals)
        # print sample SM/BSM point
        # points are given as [ctZ, cpt, cpQM, cpQ3, ctW, ctp]
        if is2D[r]:
            print ("SM point:", hp.eval(res[r]['coeff'], [0,0]))
            print ("BSM point:", hp.eval(res[r]['coeff'], [1,1]))
        else:
            print ("SM point:", hp.eval(res[r]['coeff'], [0,0,0,0,0,0]))
            print ("BSM point:", hp.eval(res[r]['coeff'], [2,0,0,0,0,0]))

        
        results[r] = {}

        for c in ['cpQM', 'cpt']:
 
            # get c axis points
            points = make_scan(operator=c, C_min=-20, C_max=20, step=1, is2D=is2D[r])
            print(points)
            c_values = []
            for i in range(0,41):
                c_values.append(i-20)

            # calculate and store results 
            pred_matrix = np.array([ np.array(hp.eval(res[r]['coeff'],points[i]['point'])) for i in range(41) ])
    
            results[r][c] = {}
            results[r][c]['inc'] = np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[20,:])
            results[r][c]['LTtail'] = np.sum(pred_matrix[:,LTmask], axis=1)/np.sum(pred_matrix[20,LTmask])
            results[r][c]['ptZtail'] = np.sum(pred_matrix[:,ptZmask], axis=1)/np.sum(pred_matrix[20,ptZmask])

            # plot
            fig, ax = plt.subplots()
            hep.cms.label(
                "Work in progress",
                data=True,
                #year=2018,
                lumi=60.0+41.5+35.9,
                loc=0,
                ax=ax,
            )
            plt.plot(c_values, results[r][c]['inc'], label=r'inclusive', c='black')
            plt.plot(c_values, results[r][c]['LTtail'], label=r'$L_{T} \geq %d\ GeV$'%LTlim, c='blue')
            plt.plot(c_values, results[r][c]['ptZtail'], label=r'$p_{T,Z} \geq %d\ GeV$'%ptZlim, c='red')

            if c == 'cpQM': 
                plt.xlabel(r'$C_{\varphi Q}^{-}$')
            else:
                plt.xlabel(r'$C_{\varphi t}$')
            plt.ylabel(r'$\sigma/\sigma_{SM}$')
            plt.legend()
    
            ax.set_ylim(0,10)
    
            fig.savefig(plot_dir+r[4:]+'_'+c+'_scaling.pdf')
            fig.savefig(plot_dir+r[4:]+'_'+c+'_scaling.png')

# comparison plots NLO vs.LO
for c in ['cpQM', 'cpt']:
    for t in ['inc','LTtail','ptZtail']:
        c_values = []
        for i in range(0,41):
            c_values.append(i-20)
        
        fig, ax = plt.subplots()
        hep.cms.label(
            "Work in progress",
            data=True,
            #year=2018,
            lumi=60.0+41.5+35.9,
            loc=0,
            ax=ax,
        )
        plt.plot(c_values, results['ttZ_NLO_2D'][c][t], label=r'NLO', c='green')
        plt.plot(c_values, results['ttZ_LO_2D'][c][t], label=r'LO', c='darkviolet')
        if c == 'cpQM':
            plt.xlabel(r'$C_{\varphi Q}^{-}$')
        else:
            plt.xlabel(r'$C_{\varphi t}$')
        plt.ylabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()

        ax.set_ylim(0,10)
        
        fig.savefig(plot_dir+'compare_'+t+'_'+c+'.pdf')
        fig.savefig(plot_dir+'compare_'+t+'_'+c+'.png')

