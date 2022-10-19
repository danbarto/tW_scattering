import os
import gzip
import pickle

try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import dense_lookup

class charge_flip:

    def __init__(self, year=2016, era=''):

        # hardcoding FTW
        chargeflip_sf_dict = {
                "UL16APV" : 0.79,
                "UL16"    : 0.81,
                "UL17"    : 1.22,
                "UL18"    : 1.12
            }

        self.year = year

        year = str(year)[2:]
        fr_in = os.path.expandvars(f"$TWHOME/data/chargeflip/flip_probs_topcoffea_UL{year}{era}.pkl.gz")

        with gzip.open(fr_in) as f_in:
            fr_hist = pickle.load(f_in)

        # NOTE lookup as function of pt and abs(eta)
        self.fr_lookup = dense_lookup.dense_lookup(fr_hist.values()[()], [fr_hist.axis("pt").edges(), fr_hist.axis("eta").edges()])
        self.sf = chargeflip_sf_dict[f'UL{year}{era}']

    def get(self, electron):

        f_1 = self.fr_lookup(electron.pt[:,0:1], abs(electron.eta[:,0:1])) * self.sf
        f_2 = self.fr_lookup(electron.pt[:,1:2], abs(electron.eta[:,1:2])) * self.sf

        # I'm using ak.prod and ak.sum to replace empty arrays by 1 and 0, respectively
        # FIXME check that this is actually necessary?
        weight = ak.sum(f_1/(1-f_1), axis=1)*ak.prod(1-f_2/(1-f_2), axis=1) + ak.sum(f_2/(1-f_2), axis=1)*ak.prod(1-f_1/(1-f_1), axis=1)

        return weight



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    sf16    = charge_flip(year=2016)
    sf16APV = charge_flip(year=2016, era='APV')
    sf17    = charge_flip(year=2017)
    sf18    = charge_flip(year=2018)

    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    from Tools.objects import Collections

    from Tools.helpers import get_samples
    from Tools.basic_objects import getJets, getBTagsDeepFlavB
    from Tools.config_helpers import loadConfig, make_small, load_yaml, data_path
    from Tools.nano_mapping import make_fileset

    samples = get_samples("samples_UL18.yaml")
    mapping = load_yaml(data_path+"nano_mapping.yaml")

    fileset = make_fileset(
        ['TTW'],
        samples,
        year='UL18',
        skim='topW_v0.7.0_dilep',
        small=True,
        n_max=1,
        buaf='local',
        merged=True,
    )
    filelist = fileset[list(fileset.keys())[0]]

    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        filelist[0],
        schemaclass = NanoAODSchema,
        entry_stop = n_max).events()

    el  = Collections(events, 'Electron', 'tightSSTTH', verbose=1).get()

    sel = (ak.num(el)==2)

    sf_central  = sf16.get(el[sel])

    print ("Found SFs:")
    print (sf_central)
