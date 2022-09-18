'''
Load NanoGEN samples in NanoAOD event factory
- Compare LO vs NLO ttZ samples
- Compare on-shell vs off-shell ttZ samples
- Compare to a benchmark point to verify that reweighting works #FIXME
- Find plane fit equation for cpt and cpQM
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


# ==================================
# ======== HELPER FUNCTIONS ========
# ==================================

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


# ==================================
# =========== MAIN CODE ============
# ==================================

if __name__ == '__main__':

    # Load samples

    base_dir = "/ceph/cms/store/user/sjeon/NanoGEN/"
    plot_dir = "/home/users/sjeon/public_html/tW_scattering/ttZ_EFT_v2/"
    finalizePlotDir(plot_dir)


    # root files

    res = {}

    res['ttZ_NLO'] = {
        'filename': "ttZ_EFT_NLO_all.root",
        #'filename': "ttZ_EFT_NLO.root",
        # this root file is smaller than the _all one
        'is2D'    : True,
        'xsecs'   : 0.930
        }

    res['ttZ_LO']  = {
        'filename': "ttZ_EFT_LO_all.root",
        #'filename': "ttZ_EFT_LO.root",
        'is2D'    : True,
        'xsecs'   : 0.663
         }

    # ptZ and LT limits

    LTlim = 700
    ptZlim = 400


    for r in res:
        print (r)

        res[r]['file'] = base_dir + res[r]['filename']
        res[r]['events'] = NanoEventsFactory.from_root(base_dir + res[r]['filename']).events()
        res[r]['data'] = {}

        hp = HyperPoly(2)

        tree = uproot.open(res[r]['file'])['Events']

        coordinates, ref_coordinates = get_coordinates_and_ref(res[r]['file'],res[r]['is2D'])
        hp.initialize(coordinates, ref_coordinates)

        weights = [ x.replace('LHEWeight_','') for x in tree.keys() if x.startswith('LHEWeight_c') ]


        # ========= Define and count selections =========

        trilep = (ak.num(res[r]['events'].GenDressedLepton)==3)
        ev = res[r]['events'][trilep]
        print('%d trilep events in total'%len(ev))

        LTmask = (get_LT(ev) >= LTlim)
        ptZmask = ak.flatten(get_Z(ev).pt >= ptZlim)
        res[r]['N_LT'] = np.sum(LTmask)
        res[r]['N_ptZ'] = np.sum(ptZmask)
        res[r]['N_LT_weighted'] = np.sum(ev.genWeight[LTmask])
        res[r]['N_ptZ_weighted'] = np.sum(ev.genWeight[ptZmask])
        print('%d (%d weighted) events with LT>%d'%(res[r]['N_LT'],res[r]['N_LT_weighted'],LTlim))
        print('%d (%d weighted) events with ptZ>%d'%(res[r]['N_ptZ'],res[r]['N_ptZ_weighted'],ptZlim))


        # =========== Calculate & plot 1D fit ===========

        # get coefficients
        allvals = [getattr(ev.LHEWeight, w) for w in weights]
        unweighted = hp.get_parametrization(allvals)
        res[r]['coeff'] = [unweighted[u]*ev.genWeight for u in range(len(unweighted))]


        for c in ['cpQM', 'cpt']:

            # get c axis points
            points = make_scan(operator=c, C_min=-20, C_max=20, step=1, is2D=res[r]['is2D'])
            c_vals = np.linspace(-20,20,41)

            # get plot data
            pred_matrix = np.array([np.array(hp.eval(res[r]['coeff'],points[i]['point'])) for i in range(41)])
            res[r]['data'][c] = {}
            data = res[r]['data'][c]
            data['inc'] = np.sum(pred_matrix, axis=1)/np.sum(pred_matrix[20,:])
            data['LTtail'] = np.sum(pred_matrix[:,LTmask], axis=1)/np.sum(pred_matrix[20,LTmask])
            data['ptZtail'] = np.sum(pred_matrix[:,ptZmask], axis=1)/np.sum(pred_matrix[20,ptZmask])

            # plot
            fig, ax = plt.subplots()
            hep.cms.label(
                "Work in progress",
                data=True,
                lumi=60.0+41.5+35.9,
                loc=0,
                ax=ax,
            )
            plt.plot(c_vals, data['inc'], label=r'inclusive', c='black')
            plt.plot(c_vals, data['LTtail'], label=r'$L_{T} \geq %d\ GeV$'%LTlim, c='blue')
            plt.plot(c_vals, data['ptZtail'], label=r'$p_{T,Z} \geq %d\ GeV$'%ptZlim, c='red')
            
            plt.plot([],[],' ',label="# ev (LT): %d(%d)"%(res[r]['N_LT'],res[r]['N_LT_weighted']))
            plt.plot([],[],' ',label="# ev (ptZ): %d(%d)"%(res[r]['N_ptZ'],res[r]['N_ptZ_weighted']))
            plt.xlabel(r'$C_{\varphi Q}^{-}$') if c == 'cpQM' else plt.xlabel(r'$C_{\varphi t}$')
            plt.ylabel(r'$\sigma/\sigma_{SM}$')
            plt.legend()

            ax.set_ylim(0,10)

            fig.savefig(plot_dir+r[4:]+'_'+c+'_scaling.pdf')
            fig.savefig(plot_dir+r[4:]+'_'+c+'_scaling.png')

    # comparison plots NLO vs.LO
    c_vals = np.linspace(-20,20,41)
    
    for c in ['cpQM', 'cpt']:
        for sel in ['inc','LTtail','ptZtail']:
            
            fig, ax = plt.subplots()
            hep.cms.label(
                "Work in progress",
                data=True,
                #year=2018,
                lumi=60.0+41.5+35.9,
                loc=0,
                ax=ax,
            )
    
            plt.plot(c_vals, res['ttZ_NLO']['data'][c][sel], label=r'NLO', c='green')
            plt.plot(c_vals, res['ttZ_LO']['data'][c][sel], label=r'LO', c='darkviolet')
    
            plt.xlabel(r'$C_{\varphi Q}^{-}$') if c == 'cpQM' else plt.xlabel(r'$C_{\varphi t}$')
            plt.ylabel(r'$\sigma/\sigma_{SM}$')
            N = 'N_LT' if sel == 'LTtail' else 'N_ptZ'
            plt.plot([], [], ' ', label="# ev (LT): %d(%d)"%(res['ttZ_NLO'][N],res['ttZ_NLO'][N+'_weighted']))
            plt.plot([], [], ' ', label="# ev (LT): %d(%d)"%(res['ttZ_LO'][N],res['ttZ_LO'][N+'_weighted']))
            plt.legend()
    
            ax.set_ylim(0,10)
            
            fig.savefig(plot_dir+'compare_'+sel+'_'+c+'.pdf')
            fig.savefig(plot_dir+'compare_'+sel+'_'+c+'.png')
