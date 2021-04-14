import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist

import numpy as np

from Tools.config_helpers import loadConfig, make_small

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from plots.helpers import makePlot

from klepto.archives import dir_archive


if __name__ == '__main__':


    small = False
    cfg = loadConfig()

    plot_dir = os.path.expandvars(cfg['meta']['plots'])
    
    cacheName = 'SS_analysis'
    if small: cacheName += '_small'
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cacheName), serialized=True)

    cache.load()

    output = cache.get('simple_output')
    
    # defining some new axes for rebinning.
    N_bins = hist.Bin('multiplicity', r'$N$', 10, -0.5, 9.5)
    N_bins_red = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
    mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 20, 0, 200)
    pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 30, 0, 300)
    pt_bins_coarse = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 300)
    pt_bins_ext = hist.Bin('pt', r'$p_{T}\ (GeV)$', 10, 0, 1000)
    eta_bins = hist.Bin('eta', r'$\eta $', 25, -5.0, 5.0)
    score_bins = hist.Bin("score",          r"N", 25, 0, 1)
    
    my_labels = {
        'topW_v3': 'top-W scat.',
        'topW_EFT_cp8': r'EFT, $C_{\varphi t}=8$',
        'topW_EFT_mix': 'EFT, operator mix',
        'TTZ': r'$t\bar{t}Z$',
        'TTW': r'$t\bar{t}W$',
        'TTH': r'$t\bar{t}H$',
        'diboson': 'VV/VVV',
        'ttbar': r'$t\bar{t}$',
    }
    
    my_colors = {
        'topW_v3': '#FF595E',
        'topW_EFT_cp8': '#000000',
        'topW_EFT_mix': '#0F7173',
        'TTZ': '#FFCA3A',
        'TTW': '#8AC926',
        'TTH': '#34623F',
        'diboson': '#525B76',
        'ttbar': '#1982C4',
    }

    makePlot(output, 'nGenL', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson'],
         signals=[],
         omit=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         overlay=output['nGenL']['topW_v3'],
         save=os.path.expandvars(plot_dir+'/SS/nGenL_test'),
        )

    makePlot(output, 'nGenL', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nGenL'),
        )

    makePlot(output, 'nGenTau', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nGenTau'),
        )

    makePlot(output, 'nLepFromW', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromW'),
        )

    makePlot(output, 'nLepFromZ', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromZ'),
        )

    makePlot(output, 'nLepFromTau', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromTau'),
        )

    makePlot(output, 'nLepFromTop', 'multiplicity',
         data=[],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nLepFromTop'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n1',
         data=[],
         bins=None, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nChargeFlip'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n2',
         data=[],
         bins=None, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nNonprompt'),
        )

    makePlot(output, 'chargeFlip_vs_nonprompt', 'n2',
         data=[],
         bins=None, log=True, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/nNonprompt_log'),
        )

    makePlot(output, 'node', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=N_bins_red, log=False, normalize=False, axis_label=r'node',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node'),
        )

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node0_score'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=True, normalize=False, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/lead_lep_pt'),
        )

    makePlot(output, 'lead_lep', 'pt',
         data=[],
         bins=pt_bins_coarse, log=False, normalize=False, axis_label=r'$p_{T}\ lead \ lep\ (GeV)$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'TTH', 'TTZ', 'ttbar'],
         save=os.path.expandvars(plot_dir+'/SS/lead_lep_pt_signals'),
        )

    makePlot(output, 'node1_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node1_score'),
        )

    makePlot(output, 'node2_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node2_score'),
        )

    makePlot(output, 'node3_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node3_score'),
        )

    makePlot(output, 'node4_score', 'score',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         #data=[],
         bins=score_bins, log=False, normalize=True, axis_label=r'score',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         #omit=['DoubleMuon', 'MuonEG', 'EGamma'],
         omit=[],
         save=os.path.expandvars(plot_dir+'/SS/ML_node4_score'),
        )

    makePlot(output, 'MET', 'pt',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=pt_bins_coarse, log=False, normalize=True, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar'],
         signals=['topW_v3'],
         omit=['topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt'),
        )

    makePlot(output, 'MET', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_coarse, log=False, shape=True, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         ymax=0.4,
         order=['TTW', 'topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         #signals=['topW_v3'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt_shape'),
        )

    makePlot(output, 'MET', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_coarse, log=False, axis_label=r'$p_{T}^{miss}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/MET_pt_signals'),
        )

    makePlot(output, 'ST', 'pt',
         data=[],
         normalize=False,
         bins=pt_bins_ext, log=False, axis_label=r'$S_{T}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['ttbar', 'diboson', 'TTZ', 'TTH', 'DoubleMuon', 'MuonEG', 'EGamma'],
         save=os.path.expandvars(plot_dir+'/SS/ST_signals'),
        )

    makePlot(output, 'PV_npvsGood', 'multiplicity',
         data=['DoubleMuon', 'MuonEG', 'EGamma'],
         bins=None, log=False, normalize=True, axis_label=r'$N_{PV}$',
         new_colors=my_colors, new_labels=my_labels,
         order=['diboson', 'TTW', 'TTH', 'TTZ', 'ttbar', 'topW_v3'],
         signals=[],
         omit=['topW_EFT_cp8', 'topW_EFT_mix'],
         save=os.path.expandvars(plot_dir+'/SS/PV_npvsGood'),
        )


    ## shapes

    makePlot(output, 'node0_score', 'score',
         data=[],
         bins=score_bins, log=False, normalize=False, axis_label=r'score', shape=True, ymax=0.35,
         new_colors=my_colors, new_labels=my_labels,
         order=['TTW'],
         signals=['topW_v3', 'topW_EFT_cp8', 'topW_EFT_mix'],
         omit=['DoubleMuon', 'MuonEG', 'EGamma', 'diboson', 'ttbar', 'TTH', 'TTZ'],
         save=os.path.expandvars(plot_dir+'/SS/ML_node0_score_shape'),
        )


    fig, ax  = plt.subplots(1,1,figsize=(10,10) )
    ax = hist.plot2d(
        output['chargeFlip_vs_nonprompt']['ttbar'].sum('n_ele').sum('dataset'),
        xaxis='n1',
        ax=ax,
        text_opts={'format': '%.3g'},
        patch_opts={},
    )
    ax.set_xlabel(r'$N_{charge flips}$')
    ax.set_ylabel(r'$N_{nonprompt}$')
    fig.savefig(plot_dir+'/SS/nChargeFlip_vs_nNonprompt_ttbar')

