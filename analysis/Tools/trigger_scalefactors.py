import os
import awkward as ak
import gzip
import pickle
import numpy as np
from coffea import hist as chist
from coffea.lookup_tools import extractor, dense_lookup

here = os.path.dirname(os.path.abspath(__file__))

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

        from analysis.Tools.helpers import pad_and_flatten
        from analysis.Tools.objects import cross, choose
        import numpy as np
        # get a lepton collection
        lep = ak.concatenate([mu, ele], axis=1)
        # now sort them
        lep = lep[ak.argsort(lep.pt, ascending=False)]
        l0_is_ele = (abs(pad_and_flatten(lep[:,0:1].pdgId))==11)
        l1_is_ele = (abs(pad_and_flatten(lep[:,1:2].pdgId))==11)

        ee_sf = self.lookups['ee']['central'](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        ee_sf = 1 + multiplier[variation]*self.lookups['ee'][multiplier](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

        em_sf = self.lookups['em']['central'](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        em_sf = 1 + multiplier[variation]*self.lookups['em'][multiplier](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))

        mm_sf = self.lookups['mm']['central'](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))
        mm_sf = 1 + multiplier[variation]*self.lookups['mm'][multiplier](pad_and_flatten(ele[:,0:1].pt), pad_and_flatten(ele[:,1:2].pt))


        sf = (ee_sf*(l0_is_ele&l1_is_ele)) + (emu_sf*(l0_is_ele^l1_is_ele)) + (mumu_sf*(~l0_is_ele&~l1_is_ele))
        #ee_mm_em = ak.concatenate([choose(ele), choose(mu), cross(mu, ele)], axis=1)
        #sf = (pad_and_flatten(sf*(ee_mm_em.pt/ee_mm_em.pt)))
        #sf = (sf/sf)*sf
        #sf = (np.where(np.isnan(sf), 1, sf))
        
        return sf

    def values(self):

        return 0



if __name__ == '__main__':
    sf16 = triggerSF(year=2016)
    sf17 = triggerSF(year=2017)
    sf18 = triggerSF(year=2018)

    #print("Evaluators found for 2016:")
    #for key in sf16.evaluator.keys():
    #    print("%s:"%key, sf16.evaluator[key])

    #print("Evaluators found for 2017:")
    #for key in sf17.evaluator.keys():
    #    print("%s:"%key, sf17.evaluator[key])

    #print("Evaluators found for 2018:")
    #for key in sf18.evaluator.keys():
    #    print("%s:"%key, sf18.evaluator[key])
        
    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from analysis.Tools.objects import Collections
    from analysis.Tools.config_helpers import loadConfig, make_small, load_yaml, data_path
    from analysis.Tools.helpers import get_samples
    from analysis.Tools.nano_mapping import make_fileset
    
    import awkward as ak

    samples = get_samples("samples_UL18.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    fileset = make_fileset(['TTW'], samples, year='UL18', skim=True, small=True, n_max=1)
    filelist = fileset[list(fileset.keys())[0]]

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        filelist[0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()
    mu  = Collections(events, 'Muon', 'tightSSTTH', verbose=1).get()

    sel = ((ak.num(el)+ak.num(mu))>1)

    sf_central  = sf18.get(el[sel], mu[sel])
    sf_up       = sf18.get(el[sel], mu[sel], variation='up')
    sf_down     = sf18.get(el[sel], mu[sel], variation='down')
    print ("Mean value of SF (central): %.3f"%ak.mean(sf_central))
    print ("Mean value of SF (up): %.3f"%ak.mean(sf_up))
    print ("Mean value of SF (down): %.3f"%ak.mean(sf_down))
