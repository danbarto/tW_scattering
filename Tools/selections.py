'''
Maybe standard selections should go in here?
'''
import awkward as ak

from coffea.analysis_tools import Weights, PackedSelection
from Tools.triggers import getTriggers, getFilters
from Tools.objects import choose, cross, choose3

class Selection:
    def __init__(self, **kwargs):
        '''
        kwargs should be:
        ele (loose and tight)
        mu
        jets: all, central, forward, b-tag
        met
        
        '''
        self.__dict__.update(kwargs)


        # not yet sure whether this should go here, or later
        self.filters   = getFilters(self.events, year=self.year, dataset=self.dataset)


    def dilep_baseline(self, omit=[], cutflow=None, tight=False, SS=True):
        '''
        give it a cutflow object if you want it to be filed.
        cuts in the omit list will not be applied
        '''
        self.selection = PackedSelection()

        is_dilep   = ((ak.num(self.ele) + ak.num(self.mu))==2)
        pos_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))<0)
        neg_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))>0)
        lep0pt     = ((ak.num(self.ele[(self.ele.pt>25)]) + ak.num(self.mu[(self.mu.pt>25)]))>0)
        lep1pt     = ((ak.num(self.ele[(self.ele.pt>20)]) + ak.num(self.mu[(self.mu.pt>20)]))>1)
        lepveto    = ((ak.num(self.ele_veto) + ak.num(self.mu_veto))==2)

        dimu    = choose(self.mu, 2)
        diele   = choose(self.ele, 2)
        dilep   = cross(self.mu, self.ele)

        if SS:
            is_SS = ( ak.any((dimu['0'].charge * dimu['1'].charge)>0, axis=1) | \
                      ak.any((diele['0'].charge * diele['1'].charge)>0, axis=1) | \
                      ak.any((dilep['0'].charge * dilep['1'].charge)>0, axis=1) )
        else:
            is_OS = ( ak.any((dimu['0'].charge * dimu['1'].charge)<0, axis=1) | \
                      ak.any((diele['0'].charge * diele['1'].charge)<0, axis=1) | \
                      ak.any((dilep['0'].charge * dilep['1'].charge)<0, axis=1) )

        lepton = ak.concatenate([self.ele, self.mu], axis=1)
        lepton_pdgId_pt_ordered = ak.fill_none(
            ak.pad_none(
                lepton[ak.argsort(lepton.pt, ascending=False)].pdgId, 2, clip=True),
        0)

        triggers  = getTriggers(self.events,
            ak.flatten(lepton_pdgId_pt_ordered[:,0:1]),
            ak.flatten(lepton_pdgId_pt_ordered[:,1:2]), year=self.year, dataset=self.dataset)

        ht = ak.sum(self.jet_all.pt, axis=1)
        st = self.met.pt + ht + ak.sum(self.mu.pt, axis=1) + ak.sum(self.ele.pt, axis=1)

        self.selection.add('lepveto',       lepveto)
        self.selection.add('dilep',         is_dilep)
        self.selection.add('filter',        self.filters)
        self.selection.add('trigger',       triggers)
        self.selection.add('p_T(lep0)>25',  lep0pt)
        self.selection.add('p_T(lep1)>20',  lep1pt)
        if SS:
            self.selection.add('SS',            is_SS )
        else:
            self.selection.add('OS',            is_OS )
        self.selection.add('N_jet>3',       (ak.num(self.jet_all)>3) )
        self.selection.add('N_jet>4',       (ak.num(self.jet_all)>4) )
        self.selection.add('N_central>2',   (ak.num(self.jet_central)>2) )
        self.selection.add('N_central>3',   (ak.num(self.jet_central)>3) )
        self.selection.add('N_btag>0',      (ak.num(self.jet_btag)>0) )
        self.selection.add('N_fwd>0',       (ak.num(self.jet_fwd)>0) )
        self.selection.add('MET>30',        (self.met.pt>30) )
        self.selection.add('MET>50',        (self.met.pt>50) )
        self.selection.add('ST>600',        (st>600) )

        ss_reqs = [
            'filter',
            'lepveto',
            'dilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS' if SS else 'OS',
            'N_jet>3',
            'N_central>2',
            'N_btag>0',
            'MET>30',
            'N_fwd>0',
        ]
        
        if tight:
            ss_reqs += [
                'N_jet>4',
                'N_central>3',
                'ST>600',
                'MET>50',
                #'delta_eta',
            ]

        ss_reqs_d = { sel: True for sel in ss_reqs if not sel in omit }
        ss_selection = self.selection.require(**ss_reqs_d)

        if cutflow:
            #
            cutflow_reqs_d = {}
            for req in ss_reqs:
                cutflow_reqs_d.update({req: True})
                cutflow.addRow( req, self.selection.require(**cutflow_reqs_d) )

        return ss_selection


    def trilep_baseline(self, omit=[], cutflow=None, tight=False):
        '''
        give it a cutflow object if you want it to be filed.
        cuts in the omit list will not be applied
        '''
        self.selection = PackedSelection()

        is_trilep  = ((ak.num(self.ele) + ak.num(self.mu))==3)
        los_trilep = ((ak.num(self.ele) + ak.num(self.mu))>=2)
        pos_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))<0)
        neg_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))>0)
        lep0pt     = ((ak.num(self.ele[(self.ele.pt>25)]) + ak.num(self.mu[(self.mu.pt>25)]))>0)
        lep1pt     = ((ak.num(self.ele[(self.ele.pt>20)]) + ak.num(self.mu[(self.mu.pt>20)]))>1)
        lepveto    = ((ak.num(self.ele_veto) + ak.num(self.mu_veto))==3)

        dimu    = choose(self.mu, 2)
        diele   = choose(self.ele, 2)
        dimu_veto = choose(self.mu_veto,2)
        diele_veto = choose(self.ele_veto,2)
        #dilep   = cross(self.mu, self.ele)

        OS_dimu = dimu[(dimu['0'].charge*dimu['1'].charge < 0)]
        OS_diele = diele[(diele['0'].charge*diele['1'].charge < 0)]
        OS_dimu_veto = dimu_veto[(dimu_veto['0'].charge*dimu_veto['1'].charge < 0)]
        OS_diele_veto = diele_veto[(diele_veto['0'].charge*diele_veto['1'].charge < 0)]
        
        SFOS = ak.concatenate([OS_diele_veto, OS_dimu_veto], axis=1)

        offZ = (ak.all(abs(OS_dimu.mass-91.2)>10, axis=1) & ak.all(abs(OS_diele.mass-91.2)>10, axis=1))
        offZ_veto = (ak.all(abs(OS_dimu_veto.mass-91.2)>10, axis=1) & ak.all(abs(OS_diele_veto.mass-91.2)>10, axis=1))

        lepton = ak.concatenate([self.ele, self.mu], axis=1)
        lepton_pdgId_pt_ordered = ak.fill_none(ak.pad_none(lepton[ak.argsort(lepton.pt, ascending=False)].pdgId, 2, clip=True), 0)
        dilep = choose(lepton,2)
        SS_dilep = (dilep['0'].charge*dilep['1'].charge > 0)
        los_trilep_SS = (ak.any(SS_dilep, axis=1))

        vetolepton   = ak.concatenate([self.ele_veto, self.mu_veto], axis=1)    
        vetotrilep = choose3(vetolepton, 3)
        pos_trilep = ak.any((vetotrilep['0'].charge+vetotrilep['1'].charge+vetotrilep['2'].charge > 0),axis=1)
        neg_trilep = ak.any((vetotrilep['0'].charge+vetotrilep['1'].charge+vetotrilep['2'].charge < 0),axis=1)
        
        triggers  = getTriggers(self.events,
            ak.flatten(lepton_pdgId_pt_ordered[:,0:1]),
            ak.flatten(lepton_pdgId_pt_ordered[:,1:2]), year=self.year, dataset=self.dataset)

        ht = ak.sum(self.jet_all.pt, axis=1)
        st = self.met.pt + ht + ak.sum(self.mu.pt, axis=1) + ak.sum(self.ele.pt, axis=1)
        st_veto = self.met.pt + ht + ak.sum(self.mu_veto.pt, axis=1) + ak.sum(self.ele_veto.pt, axis=1)

        lep0pt_veto     = ((ak.num(self.ele_veto[(self.ele_veto.pt>25)]) + ak.num(self.mu_veto[(self.mu_veto.pt>25)]))>0)
        lep1pt_veto     = ((ak.num(self.ele_veto[(self.ele_veto.pt>20)]) + ak.num(self.mu_veto[(self.mu_veto.pt>20)]))>1)

        self.selection.add('lepveto',       lepveto)
        self.selection.add('trilep',        los_trilep_SS)
        self.selection.add('filter',        self.filters)
        self.selection.add('trigger',       triggers)
        self.selection.add('p_T(lep0)>25',  lep0pt_veto)
        self.selection.add('p_T(lep1)>20',  lep1pt_veto)
        self.selection.add('N_jet>2',       (ak.num(self.jet_all)>2) )
        self.selection.add('N_jet>3',       (ak.num(self.jet_all)>3) )
        self.selection.add('N_central>1',   (ak.num(self.jet_central)>1) )
        self.selection.add('N_central>2',   (ak.num(self.jet_central)>2) )
        self.selection.add('N_btag>0',      (ak.num(self.jet_btag)>0 ))
        self.selection.add('N_fwd>0',       (ak.num(self.jet_fwd)>0) )
        self.selection.add('MET>50',        (self.met.pt>50) )
        self.selection.add('ST>600',        (st_veto>600) )
        self.selection.add('offZ',          offZ_veto )
        #self.selection.add('SFOS>=1',          ak.num(SFOS)==0)
        #self.selection.add('charge_sum',          neg_trilep)
        
        reqs = [
            'filter',
            'lepveto',
            'trilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'offZ',
            'MET>50',
            'N_jet>2',
            'N_central>1',
            'N_btag>0',
            'N_fwd>0',
            #'SFOS>=1',
            #'charge_sum'
        ]
        
        if tight:
            reqs += [
                'N_jet>3',
                'N_central>2',
                'ST>600',
                #'MET>50',
                #'delta_eta',
            ]

        reqs_d = { sel: True for sel in reqs if not sel in omit }
        selection = self.selection.require(**reqs_d)

        self.reqs = [ sel for sel in reqs if not sel in omit ]

        if cutflow:
            #
            cutflow_reqs_d = {}
            for req in reqs:
                cutflow_reqs_d.update({req: True})
                cutflow.addRow( req, self.selection.require(**cutflow_reqs_d) )

        return selection


if __name__ == '__main__':
    
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    from coffea.analysis_tools import Weights, PackedSelection
    from Tools.samples import fileset_2018
    
    # the below command will change to .from_root in coffea v0.7.0
    ev = NanoEventsFactory.from_root(fileset_2018['TTW'][0], schemaclass=NanoAODSchema).events()
    
    sel = Selection(
        dataset = "TTW",
        events = ev,
        year = 2018,
        ele = ev.Electron,
        ele_veto = ev.Electron,
        mu = ev.Muon,
        mu_veto = ev.Muon,
        jet_all = ev.Jet,
        jet_central = ev.Jet,
        jet_btag = ev.Jet,
        jet_fwd = ev.Jet,
        met = ev.MET,
    )

    trilep = sel.trilep_baseline(omit=['N_btag>0', 'N_fwd>0'], tight=False)
    print ("Found %s raw events in trilep selection"%sum(trilep))
    print ("Applied the following requirements:")
    print (sel.reqs)
