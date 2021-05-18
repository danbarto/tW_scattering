import awkward as ak
from coffea.analysis_tools import PackedSelection
from Tools.objects import choose, cross

def SS_selection(lep1, lep2):
    selection = PackedSelection()

    is_dilep   = ((ak.num(lep1) + ak.num(lep2))==2)
    pos_charge = ((ak.sum(lep1.pdgId, axis=1) + ak.sum(lep2.pdgId, axis=1))<0)
    neg_charge = ((ak.sum(lep1.pdgId, axis=1) + ak.sum(lep2.pdgId, axis=1))>0)

    dilep2    = choose(lep2, 2)
    dilep1   = choose(lep1, 2)
    dilep   = cross(lep2, lep1)

    is_SS = ( ak.any((dilep2['0'].charge * dilep2['1'].charge)>0, axis=1) | \
              ak.any((dilep1['0'].charge * dilep1['1'].charge)>0, axis=1) | \
              ak.any((dilep['0'].charge * dilep['1'].charge)>0, axis=1) )

    selection.add('SS', is_SS)
    ss_reqs = ['SS']

    ss_reqs_d = {sel: True for sel in ss_reqs}
    ss_selection = selection.require(**ss_reqs_d)
    return ss_selection
