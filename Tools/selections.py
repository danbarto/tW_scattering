'''
Maybe standard selections should go in here?
'''
import awkward as ak

from coffea.analysis_tools import Weights, PackedSelection
from Tools.triggers import getTriggers, getFilters
from Tools.objects import choose, cross, choose3

def get_pt(lep):
    mask_tight    = (lep.id!=1)*1  # either veto (and not fakeable) or tight
    mask_fakeable = (lep.id==1)*1

    return lep.pt*mask_tight + lep.conePt*mask_fakeable

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


    def dilep_baseline(self, omit=[], cutflow=None, tight=False, SS=True, DY=False):
        '''
        give it a cutflow object if you want it to be filed.
        cuts in the omit list will not be applied
        '''
        self.selection = PackedSelection()

        lepton = ak.concatenate([self.ele, self.mu], axis=1)

        is_dilep   = ( ((ak.num(self.ele) + ak.num(self.mu))==2) & ((ak.num(self.ele_veto) + ak.num(self.mu_veto))==2) )
        pos_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))<0)
        neg_charge = ((ak.sum(self.ele.pdgId, axis=1) + ak.sum(self.mu.pdgId, axis=1))>0)
        lep0pt     = ((ak.num(self.ele[(get_pt(self.ele)>25)]) + ak.num(self.mu[(get_pt(self.mu)>25)]))>0)
        lep1pt     = ((ak.num(self.ele[(get_pt(self.ele)>20)]) + ak.num(self.mu[(get_pt(self.mu)>20)]))>1)
        #lepsel     = ((ak.num(self.ele_tight) + ak.num(self.mu_tight))==2)

        dimu    = choose(self.mu, 2)
        diele   = choose(self.ele, 2)
        dilep   = choose(lepton, 2)

        is_SS = ( ak.sum(lepton.charge, axis=1)!=0 )
        is_OS = ( ak.sum(lepton.charge, axis=1)==0 )

        triggers  = getTriggers(self.events, year=self.year, dataset=self.dataset, era=self.era)
        #triggers  = getTriggers(self.events, year=self.year, dataset=self.dataset)

        ht = ak.sum(self.jet_all.pt, axis=1)
        st = self.met.pt + ht + ak.sum(self.mu.pt, axis=1) + ak.sum(self.ele.pt, axis=1)
        
        min_mll = ak.all(dilep.mass>12, axis=1)

        #self.selection.add('lepsel',        lepsel)
        self.selection.add('dilep',         is_dilep)
        self.selection.add('filter',        self.filters)
        self.selection.add('trigger',       triggers)
        self.selection.add('p_T(lep0)>25',  lep0pt)
        self.selection.add('p_T(lep1)>20',  lep1pt)
        self.selection.add('SS',            is_SS )
        self.selection.add('OS',            is_OS )
        self.selection.add('N_jet>1',       (ak.num(self.jet_all)>1) )
        self.selection.add('N_jet>3',       (ak.num(self.jet_all)>3) )
        self.selection.add('N_jet>4',       (ak.num(self.jet_all)>4) )
        self.selection.add('N_central>1',   (ak.num(self.jet_central)>1) )
        self.selection.add('N_central>2',   (ak.num(self.jet_central)>2) )
        self.selection.add('N_central>3',   (ak.num(self.jet_central)>3) )
        self.selection.add('N_btag=0',      (ak.num(self.jet_btag)==0) )
        self.selection.add('N_btag>0',      (ak.num(self.jet_btag)>0) )
        self.selection.add('N_light>0',     (ak.num(self.jet_light)>0) )
        self.selection.add('N_fwd>0',       (ak.num(self.jet_fwd)>0) )
        self.selection.add('MET>30',        (self.met.pt>30) )
        self.selection.add('MET>50',        (self.met.pt>50) )
        self.selection.add('ST>600',        (st>600) )
        self.selection.add('min_mll',       (min_mll) )
        
        reqs = [
            'filter',
         #   'lepsel',
            'dilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS' if SS else 'OS',
            'N_jet>3',
            'N_central>2',
            'N_btag>0',
            'N_light>0',
            'MET>30',
            'N_fwd>0',
            'min_mll'
        ]

        reqs_DY = [
            'filter',
            'dilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'OS',
            'N_jet>1',
            'N_central>1',
            'N_btag=0',
            'min_mll'
        ]
        
        if tight:
            reqs += [
                'N_jet>4',
                'N_central>3',
                'ST>600',
                'MET>50',
                #'delta_eta',
            ]

        if DY: reqs = reqs_DY

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


    def trilep_baseline(self, omit=[], cutflow=None, tight=False):
        '''
        give it a cutflow object if you want it to be filed.
        cuts in the omit list will not be applied
        every quantity in trilep should be calculated from loose leptons
        '''
        self.selection = PackedSelection()

        is_trilep  = ( ((ak.num(self.ele_veto) + ak.num(self.mu_veto))>=3) & ((ak.num(self.ele) + ak.num(self.mu))>=3) )
        lep0pt     = ((ak.num(self.ele_veto[(get_pt(self.ele_veto)>25)]) + ak.num(self.mu_veto[(get_pt(self.mu_veto)>25)]))>0)
        lep1pt     = ((ak.num(self.ele_veto[(get_pt(self.ele_veto)>20)]) + ak.num(self.mu_veto[(get_pt(self.mu_veto)>20)]))>1)
        # FIXME here we need to have a 10 GeV threshold on the third lepton

        dimu    = choose(self.mu_veto,2)
        diele   = choose(self.ele_veto,2)

        OS_dimu     = dimu[(dimu['0'].charge*dimu['1'].charge < 0)]
        OS_diele    = diele[(diele['0'].charge*diele['1'].charge < 0)]
        
        SFOS = ak.concatenate([OS_diele, OS_dimu], axis=1)  # do we have SF OS?

        offZ = (ak.all(abs(OS_dimu.mass-91.2)>10, axis=1) & ak.all(abs(OS_diele.mass-91.2)>10, axis=1))
        onZ = (ak.all(abs(OS_dimu.mass-91.2)<10, axis=1) & ak.all(abs(OS_diele.mass-91.2)<10, axis=1))

        lepton_tight = ak.concatenate([self.ele, self.mu], axis=1)
        SS_dilep = ( ak.sum(lepton_tight.charge, axis=1)!=0 )  # this makes sure that at least the SS leptons are tight, or all 3 leptons are tight

        # get lepton vectors for trigger
        lepton = ak.concatenate([self.ele_veto, self.mu_veto], axis=1)

        vetolepton   = ak.concatenate([self.ele_veto, self.mu_veto], axis=1)    
        vetotrilep = choose3(vetolepton, 3)

        pos_trilep =  ( ak.sum(lepton.charge, axis=1)>0 )
        neg_trilep =  ( ak.sum(lepton.charge, axis=1)<0 )
        
        #triggers  = getTriggers(self.events, year=self.year, dataset=self.dataset, era=self.era)
        triggers  = getTriggers(self.events, year=self.year, dataset=self.dataset)

        ht = ak.sum(self.jet_all.pt, axis=1)
        st = self.met.pt + ht + ak.sum(self.mu.pt, axis=1) + ak.sum(self.ele.pt, axis=1)
        st_veto = self.met.pt + ht + ak.sum(self.mu_veto.pt, axis=1) + ak.sum(self.ele_veto.pt, axis=1)

        self.selection.add('trilep',        is_trilep)
        self.selection.add('SS_dilep',      SS_dilep)
        self.selection.add('p_T(lep0)>25',  lep0pt)
        self.selection.add('p_T(lep1)>20',  lep1pt)
        self.selection.add('filter',        self.filters)
        self.selection.add('trigger',       triggers)
        self.selection.add('N_jet>2',       (ak.num(self.jet_all)>2) )
        self.selection.add('N_jet>3',       (ak.num(self.jet_all)>3) )
        self.selection.add('N_central>1',   (ak.num(self.jet_central)>1) )
        self.selection.add('N_central>2',   (ak.num(self.jet_central)>2) )
        self.selection.add('N_btag>0',      (ak.num(self.jet_btag)>0 ))
        self.selection.add('N_fwd>0',       (ak.num(self.jet_fwd)>0) )
        self.selection.add('MET>50',        (self.met.pt>50) )
        self.selection.add('ST>600',        (st_veto>600) )
        self.selection.add('offZ',          offZ )
        self.selection.add('onZ',           onZ )
        #self.selection.add('SFOS>=1',          ak.num(SFOS)==0)
        #self.selection.add('charge_sum',          neg_trilep)
        
        reqs = [
            'filter',
            'trilep',
            'p_T(lep0)>25',
            'p_T(lep1)>20',
            'trigger',
            'SS_dilep',
            #'offZ',
            'onZ',
            'MET>50',
            'N_jet>2',
            'N_central>1',
            #'N_btag>0',
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
    import warnings
    warnings.filterwarnings("ignore")

    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    from coffea.analysis_tools import Weights, PackedSelection
    from Tools.samples import fileset_2018
    from Tools.objects import Collections
    from Tools.helpers import get_four_vec_fromPtEtaPhiM, getCutFlowTable
    from Tools.cutflow import Cutflow

    from coffea import processor
    from coffea.analysis_tools import Weights
    from processor.default_accumulators import desired_output as output

    output.update({'TTW': processor.defaultdict_accumulator(int)})

    year = 2018

    # the below command will change to .from_root in coffea v0.7.0
    ev = NanoEventsFactory.from_root(fileset_2018['TTW'][0], schemaclass=NanoAODSchema).events()
    ev.metadata['dataset'] = 'TTW'

    weight = Weights( len(ev) )
    weight.add("weight", ev.weight)

    ## Muons
    mu_v     = Collections(ev, "Muon", "vetoTTH", year=year).get()  # these include all muons, tight and fakeable
    mu_t     = Collections(ev, "Muon", "tightSSTTH", year=year).get()
    mu_f     = Collections(ev, "Muon", "fakeableSSTTH", year=year).get()
    muon     = ak.concatenate([mu_t, mu_f], axis=1)
    muon['p4'] = get_four_vec_fromPtEtaPhiM(muon, get_pt(muon), muon.eta, muon.phi, muon.mass, copy=False)
    mu_v['p4'] = get_four_vec_fromPtEtaPhiM(mu_v, get_pt(mu_v), mu_v.eta, mu_v.phi, mu_v.mass, copy=False) 
    # the muon object automatically has the right id member, but for veto we need to take care of this.
    # FIXME    


    ## Electrons
    el_v        = Collections(ev, "Electron", "vetoTTH", year=year).get()
    el_t        = Collections(ev, "Electron", "tightSSTTH", year=year).get()
    el_f        = Collections(ev, "Electron", "fakeableSSTTH", year=year).get()
    electron    = ak.concatenate([el_t, el_f], axis=1)
    electron['p4'] = get_four_vec_fromPtEtaPhiM(electron, get_pt(electron), electron.eta, electron.phi, electron.mass, copy=False)
    el_v['p4'] = get_four_vec_fromPtEtaPhiM(el_v, get_pt(el_v), el_v.eta, el_v.phi, el_v.mass, copy=False)

    sel = Selection(
        dataset = "TTW",
        events = ev,
        year = year,
        ele = electron,
        ele_veto = el_v,
        mu = muon,
        mu_veto = mu_v,
        jet_all = ev.Jet,
        jet_light = ev.Jet,
        jet_central = ev.Jet,
        jet_btag = ev.Jet,
        jet_fwd = ev.Jet,
        met = ev.MET,
    )

    dilep = sel.dilep_baseline(omit=['N_fwd>0'], tight=False)
    print ("Found %s raw events in dilep selection"%sum(dilep))
    print ("Applied the following requirements:")
    print (sel.reqs)

    trilep = sel.trilep_baseline(omit=['N_fwd>0'], tight=False)
    print ("Found %s raw events in trilep selection"%sum(trilep))
    print ("Applied the following requirements:")
    print (sel.reqs)

