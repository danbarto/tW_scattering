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
from scipy.optimize import curve_fit

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
    #base_dir = "/home/users/sjeon/ttw/CMSSW_10_6_19/src/"
    base_dir = "/ceph/cms/store/user/sjeon/NanoGEN/"
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
        pt_axis = hist.Bin("pt", r"p", 1, 0, 5000)
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
        allvals = [getattr(ev.LHEWeight, w) for w in weights]
        res[r]['coeff'] = hp.get_parametrization(allvals)

        allvals_fit = [histo_values(res[r]['hist'], w) for w in weights]
        res[r]['coeff_fit'] = hp.get_parametrization(allvals_fit)

        # print sample SM/BSM point
        # points are given as [ctZ, cpt, cpQM, cpQ3, ctW, ctp]
        #if is2D[r]:
        #    print ("SM point:", hp.eval(res[r]['coeff'], [0,0]))
        #    print ("BSM point:", hp.eval(res[r]['coeff'], [1,1]))
        #else:
        #    print ("SM point:", hp.eval(res[r]['coeff'], [0,0,0,0,0,0]))
        #    print ("BSM point:", hp.eval(res[r]['coeff'], [2,0,0,0,0,0]))

        # get c axis points
        points = make_scan_2D(operators=['cpt','cpQM'], C_min=-20, C_max=20, step=1)
        cpt_mesh = np.linspace(-20,20,41)
        cpQM_mesh = np.linspace(-20,20,41)
        cpt_mesh, cpQM_mesh = np.meshgrid(cpt_mesh,cpQM_mesh)

        # calculate and store results 
        pred_matrix = [ [ hp.eval(res[r]['coeff'],points[i1][i2]['point']) for i1 in range(41) ] for i2 in range(41)]
        pred_matrix_fit = [ [ hp.eval(res[r]['coeff_fit'],points[i1][i2]['point']) for i1 in range(41) ] for i2 in range(41)]

        #print(pred_matrix)
        results[r] = {}
        results[r]['inc'] = np.sum(pred_matrix*ev.genWeight, axis=2)/np.sum(pred_matrix[20][20]*ev.genWeight)
        results[r]['inc_fit'] = np.sum(pred_matrix_fit, axis=2)/np.sum(pred_matrix_fit[20][20])

        print('point | real data | fit')
        for p in [[5,8],[26,30],[1,40]]:
            i = p[0]-20
            j = p[1]-20
            print('%d,%d | %.2f | %.2f'%(i,j,results[r]['inc'][p[0]][p[1]],results[r]['inc_fit'][p[0]][p[1]]))

        # plot colormap
        fig, ax = plt.subplots()
        hep.cms.label(
                "Work in progress",
                data=True,
                lumi=60.0+41.5+35.9,
                loc=0,
                ax=ax,
            )
        im = ax.imshow(results[r]['inc'],
                  interpolation='gaussian',
                  cmap='viridis',
                  origin='lower',
                  extent=[-20,20,-20,20])
        plt.colorbar(im)

        #ax.plot_surface(cpt_values, cpQM_values, results[r]['inc'], cmap=cm.coolwarm)
        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        #plt.zlabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()

        fig.savefig(plot_dir+r[4:]+'_scaling.pdf')
        fig.savefig(plot_dir+r[4:]+'_scaling.png')

        # fit and plot 3D
        cpt = np.array(cpt_mesh).flatten()
        cpQM = np.array(cpQM_mesh).flatten()
        #A = np.array([cpt**4,cpQM**4,cpt**2*cpQM**2,cpt**2,cpQM**2,cpt*cpQM,cpt*0+1]).T
        #B = results[r]['inc'].flatten()
        #coeff, residuals, rank, s = np.linalg.lstsq(A, B)
        #results[r]['fit_coeff'] = coeff

        coeff = np.sum(res[r]['coeff_fit'],axis=1)
        def plane_func(xt,xQM,A,B,C,D,E,F):
            return A+B*xt+C*xQM+D*xt**2+E*xt*xQM+F*xQM**2
            #return (A*xt**4)+(B*xQM**4)+(C*xt**2*xQM**2)+(D*xt**2)+(E*xQM**2)+(F*xt*xQM)+G
        plot_func = plane_func(cpt_mesh,cpQM_mesh,
                       coeff[0],coeff[1],coeff[2],coeff[3],coeff[4],coeff[5])
        print(plot_func)
        fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax2.plot_surface(cpt_mesh,cpQM_mesh,plot_func,cmap=cm.coolwarm)
        ax2.set_xlim3d(-20,20)
        ax2.set_ylim3d(-20,20)
        ax2.set_zlim3d(0,10)
        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        ax2.text2D(0.0, 0.8,
            r'${:.3f}$' '\n'
            r'$+{:.3f}x_{{t}}$' '\n'
            r'$+{:.3f}x_{{QM}}$' '\n'
            r'$+{:.3f}x_t^2$' '\n'
            r'$+{:.3f}x_tx_{{QM}}$' '\n'
            r'$+{:.3f}x_{{QM}}^2$'.format(*coeff),
            transform=ax2.transAxes, wrap=True, fontsize='small')
        #fig.savefig(plot_dir+r[4:]+'_fit.pdf')
        fig2.savefig(plot_dir+r[4:]+'_fit.png')

        # plot 3D
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection='3d')
        ax3.set_xlim3d(-20,20) 
        ax3.set_ylim3d(-20,20)
        ax3.set_zlim3d(0,10)
        ax3.plot_surface(cpt_mesh, cpQM_mesh, results[r]['inc'], cmap=cm.coolwarm)
        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        #plt.zlabel(r'$\sigma/\sigma_{SM}$')
        plt.legend()
        fig3.savefig(plot_dir+r[4:]+'_data.png')

