import os
import awkward as ak
import gzip
import pickle
import numpy as np
from coffea import hist as chist
from coffea.lookup_tools import extractor, dense_lookup

here = os.path.dirname(os.path.abspath(__file__))

from analysis.Tools.helpers import pad_and_flatten

#### Functions needed, taken from TopEFT
StackOverUnderflow = lambda v : [sum(v[0:2])] + v[2:-2] + [sum(v[-2:])]

def GetClopperPearsonInterval(hnum, hden):
    ''' Compute Clopper-Pearson interval from numerator and denominator histograms '''
    num = list(hnum.values(overflow='all')[()])
    den = list(hden.values(overflow='all')[()])
    if isinstance(num, list) and isinstance(num[0], np.ndarray):
        for i in range(len(num)):
            num[i] = np.array(StackOverUnderflow(list(num[i])), dtype=float)
            den[i] = np.array(StackOverUnderflow(list(den[i])), dtype=float)
        den = StackOverUnderflow(den)
        num = StackOverUnderflow(num)
    else:
        num = np.array(StackOverUnderflow(num), dtype=float)
        den = np.array(StackOverUnderflow(den), dtype=float)
    num = np.array(num)
    den = np.array(den)
    num[num>den] = den[num > den]
    down, up = chist.clopper_pearson_interval(num, den)
    ratio = np.array(num, dtype=float) / den
    return [ratio, down, up]

def GetEff(num, den):
    ''' Compute efficiency values from numerator and denominator histograms '''
    ratio, down, up = GetClopperPearsonInterval(num, den)
    axis = num.axes()[0].name
    bins = num.axis(axis).edges()
    x    = num.axis(axis).centers()
    xlo  = bins[:-1]
    xhi  = bins[1:]
    return [[x, xlo-x, xhi-x],[ratio, down-ratio, up-ratio]]

def GetSFfromCountsHisto(hnumMC, hdenMC, hnumData, hdenData):
    ''' Compute scale factors from efficiency histograms for data and MC '''
    Xmc, Ymc = GetEff(hnumMC, hdenMC)
    Xda, Yda = GetEff(hnumData, hdenData)
    ratio, do, up = GetRatioAssymetricUncertainties(Yda[0], Yda[1], Yda[2], Ymc[0], Ymc[1], Ymc[2])
    return ratio, do, up

def GetRatioAssymetricUncertainties(num, numDo, numUp, den, denDo, denUp):
    ''' Compute efficiencies from numerator and denominator counts histograms and uncertainties '''
    ratio = num / den
    uncUp = ratio * np.sqrt(numUp * numUp + denUp * denUp)
    uncDo = ratio * np.sqrt(numDo * numDo + denDo * denDo)
    return ratio, -uncDo, uncUp


class triggerSF:

    def __init__(self, year=2016):
        self.year = year

        inputfiles = {
            2016: os.path.join(here, "data/trigger/triggerSF_2016.pkl.gz"),
            2017: os.path.join(here, "data/trigger/triggerSF_2017.pkl.gz"),
            2018: os.path.join(here, "data/trigger/triggerSF_2018.pkl.gz"),
        }

        with gzip.open(inputfiles[self.year]) as fin:
            hin = pickle.load(fin)

        axisY = 'l1pt'

        self.lookups = {}
        for ch in ['mm', 'em', 'ee']:
            self.lookups[ch] = {}
            h = hin['2l'][ch]
            ratio, do, up = GetSFfromCountsHisto(h['hmn'], h['hmd'], h['hdn'], h['hdd'])
            ratio = np.nan_to_num(ratio, 1.)
            do = np.nan_to_num(do, 0.)
            up = np.nan_to_num(up, 0.)
            self.lookups[ch]['central'] = dense_lookup.dense_lookup(ratio, [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
            self.lookups[ch]['up'] = dense_lookup.dense_lookup(up, [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
            self.lookups[ch]['down'] = dense_lookup.dense_lookup(do, [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])


    def get_dilep(self, ele, mu, variation='central'):
        multiplier = {'central': 0, 'up': 1, 'down': -1}
        var_fixed = {'central': 'central', 'up': 'down', 'down': 'down'}
        var = var_fixed[variation]

        # get a lepton collection
        lep = ak.concatenate([mu, ele], axis=1)
        # now sort them
        lep = lep[ak.argsort(lep.pt, ascending=False)]
        l0_is_ele = (abs(pad_and_flatten(lep[:,0:1].pdgId))==11)
        l1_is_ele = (abs(pad_and_flatten(lep[:,1:2].pdgId))==11)

        ## Di-electron SFs
        ee_sf = self.lookups['ee']['central'](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        ee_sf_var = self.lookups['ee'][var](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        ee_sf = ee_sf + multiplier[variation]*np.sqrt(np.where(ee_sf_var==1.0, 0.0, ee_sf_var)**2 + (0.02 * ee_sf)**2)

        em_sf = self.lookups['em']['central'](
            np.where(pad_and_flatten(ele[:,0:1].pt)>pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(mu[:,0:1].pt)),
            np.where(pad_and_flatten(ele[:,0:1].pt)<pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,0:1].pt)),
        )
        em_sf_var = self.lookups['em'][var](
            np.where(pad_and_flatten(ele[:,0:1].pt)>pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(mu[:,0:1].pt)),
            np.where(pad_and_flatten(ele[:,0:1].pt)<pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(ele[:,0:1].pt)),
        )
        em_sf = em_sf + multiplier[variation]*np.sqrt(np.where(em_sf_var==1.0, 0.0, em_sf_var)**2 + (0.02 * em_sf)**2)

        mm_sf = self.lookups['mm']['central'](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
        mm_sf_var = self.lookups['mm'][var](pad_and_flatten(mu[:,0:1].pt), pad_and_flatten(mu[:,1:2].pt))
        mm_sf = mm_sf + multiplier[variation]*np.sqrt(np.where(mm_sf_var==1.0, 0.0, mm_sf_var)**2 + (0.02 * mm_sf)**2)

        sf = (ee_sf*(l0_is_ele&l1_is_ele)) + (em_sf*(l0_is_ele^l1_is_ele)) + (mm_sf*(~l0_is_ele&~l1_is_ele))
        #ee_mm_em = ak.concatenate([choose(ele), choose(mu), cross(mu, ele)], axis=1)
        #sf = (pad_and_flatten(sf*(ee_mm_em.pt/ee_mm_em.pt)))
        #sf = (sf/sf)*sf
        #sf = (np.where(np.isnan(sf), 1, sf))
        
        return sf

    def get_trilep(self, ele, mu, variation="central"):
        multiplier = {'central': 0, 'up': 1, 'down': -1}
        ee_sf = self.lookups['ee']['central'](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        sf = np.ones_like(ee_sf) + multiplier[variation]*0.02
        return sf

    def values(self):

        return 0



if __name__ == '__main__':
    sf16 = triggerSF(year=2016)
    sf17 = triggerSF(year=2017)
    sf18 = triggerSF(year=2018)


    ## Load a single file here, get leptons, eval SFs just to be sure everything works

    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    import hist

    from analysis.Tools.objects import Collections

    from analysis.Tools.samples import Samples
    from analysis.Tools.config_helpers import load_yaml, data_path

    samples = Samples.from_yaml(f'analysis/Tools/data/samples_v0_8_0_SS.yaml')  # NOTE this could be era etc dependent
    fileset = samples.get_fileset(year='UL18', groups=['TTW'])

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        fileset[list(fileset.keys())[0]][0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()


    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
    mu  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()

    sel = ((ak.num(el)+ak.num(mu))==2)
    el_sel = ((ak.num(el)==2)&(ak.num(mu)==0))
    mu_sel = ((ak.num(el)==0)&(ak.num(mu)==2))

    sf_central  = sf18.get_dilep(el[sel], mu[sel])
    sf_up       = sf18.get_dilep(el[sel], mu[sel], variation='up')
    sf_down     = sf18.get_dilep(el[sel], mu[sel], variation='down')
    sf_central_3l  = sf18.get_trilep(el[sel], mu[sel])
    sf_up_3l  = sf18.get_trilep(el[sel], mu[sel], variation='up')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    print ("Mean value of SF (up): %.3f"%ak.mean(sf_up))
    print ("Mean value of SF (down): %.3f"%ak.mean(sf_down))
    print ("Mean value of trilep SF (central): %.3f"%ak.mean(sf_central_3l))
    print ("Mean value of trilep SF (up): %.3f"%ak.mean(sf_up_3l))
