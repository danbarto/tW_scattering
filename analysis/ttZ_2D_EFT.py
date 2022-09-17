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
        'is2D'    : True,
        'xsecs'   : 0.930
        }

    res['ttZ_LO']  = {
    'filename': "ttZ_EFT_LO_all.root",
        'is2D'    : True,
        'xsecs'   : 0.663
         }


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


        # =========== Calculate & plot 2D fit ===========

        # get c axis points
        points = make_scan_2D(operators=['cpt','cpQM'], C_min=-20, C_max=20, step=1)
        cpt_mesh = np.linspace(-20,20,41)
        cpQM_mesh = np.linspace(-20,20,41)
        cpt_mesh, cpQM_mesh = np.meshgrid(cpt_mesh,cpQM_mesh)

        # get event-by-event coeffs & plot data
        allvals = [getattr(ev.LHEWeight, w) for w in weights]
        unweighted = hp.get_parametrization(allvals)
        res[r]['coeff'] = [unweighted[u]*ev.genWeight for u in range(len(unweighted))]

        pred_matrix = [ [ hp.eval(res[r]['coeff'],points[i1][i2]['point']) for i2 in range(41) ] for i1 in range(41)]

        # get 2D fit coeffs & plot data
        pt_axis = hist.Bin("pt", r"p", 1, 0, 5000)
        res[r]['hist'] = hist.Hist("met", dataset_axis, pt_axis)

        for w in weights:
            res[r]['hist'].fill(
                dataset=w,
                pt = get_LT(ev),
                weight=res[r]['xsecs']*getattr(ev.LHEWeight, w)*ev.genWeight/sum(ev.genWeight)
            )

        allvals_fit = [histo_values(res[r]['hist'], w) for w in weights]
        res[r]['coeff_fit'] = hp.get_parametrization(allvals_fit)

        pred_matrix_fit = [ [ hp.eval(res[r]['coeff_fit'],points[i1][i2]['point']) for i2 in range(41) ] for i1 in range(41)]

        res[r]['data']['inc'] = np.sum(pred_matrix,axis=2)/np.sum(pred_matrix[20][20])
        res[r]['data']['inc_fit'] = np.sum(pred_matrix_fit, axis=2)/np.sum(pred_matrix_fit[20][20])

        # sanity check
        print('point | real data | fit')
        for p in [[5,8],[26,30],[1,40]]:
            i = p[0]-20
            j = p[1]-20
            print('%d,%d | %.2f | %.2f'%(i,j,res[r]['data']['inc'][p[0]][p[1]],res[r]['data']['inc_fit'][p[0]][p[1]]))

        # Heatmap plot of data
        fig, ax = plt.subplots()
        hep.cms.label(
                "WIP",
                data=True,
                lumi=60.0+41.5+35.9,
                loc=0,
                ax=ax,
            )
        im = ax.imshow(res[r]['data']['inc'],
                  interpolation='gaussian',
                  cmap='viridis',
                  origin='lower',
                  extent=[-20,20,-20,20])
        plt.colorbar(im)

        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        plt.legend()

        fig.savefig(plot_dir+r[4:]+'_scaling.pdf')
        fig.savefig(plot_dir+r[4:]+'_scaling.png')

        # 3D plot of fit
        cpt = np.array(cpt_mesh).flatten()
        cpQM = np.array(cpQM_mesh).flatten()

        coeff = np.sum(res[r]['coeff_fit'],axis=1)
        def plane_func(xt,xQM,A,B,C,D,E,F):
            return A+B*xt+C*xQM+D*xt**2+E*xt*xQM+F*xQM**2
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
        fig2.savefig(plot_dir+r[4:]+'_fit.pdf')
        fig2.savefig(plot_dir+r[4:]+'_fit.png')

        # 3D plot of data
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection='3d')
        ax3.set_xlim3d(-20,20) 
        ax3.set_ylim3d(-20,20)
        ax3.set_zlim3d(0,10)
        ax3.plot_surface(cpt_mesh, cpQM_mesh, res[r]['data']['inc'], cmap=cm.coolwarm)
        plt.ylabel(r'$C_{\varphi Q}^{-}$')
        plt.xlabel(r'$C_{\varphi t}$')
        plt.legend()
        fig3.savefig(plot_dir+r[4:]+'_data.pdf')
        fig3.savefig(plot_dir+r[4:]+'_data.png')
