try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

import numpy as np


class meta_processor(processor.ProcessorABC):
    def __init__(self, accumulator={}):
        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']

        try:
            sumw = sum(events['genEventSumw'])
            sumw2 = sum(events['genEventSumw2'])
            nevents = sum(events['genEventCount'])
        except ValueError:
            # this happens for data
            sumw = 0
            sumw2 = 0
            nevents = 0


        output[events.metadata['filename']]['sumWeight'] += sumw  # naming for consistency...
        output[events.metadata['filename']]['sumWeight2'] += sumw2  # naming for consistency...
        output[events.metadata['filename']]['nevents'] += nevents
        output[events.metadata['filename']]['nChunk'] += 1

        output[dataset]['sumWeight'] += sumw
        output[dataset]['sumWeight2'] += sumw2
        output[dataset]['nevents'] += nevents
        output[dataset]['nChunk'] += 1
        
        return output

    def postprocess(self, accumulator):
        return accumulator


def get_sample_meta(fileset, samples, workers=10, skipbadfiles=True):
    
    from processor.default_accumulators import add_processes_to_output, add_files_to_output

    meta_accumulator = {}
    add_files_to_output(fileset, meta_accumulator)
    add_processes_to_output(fileset, meta_accumulator)

    meta_output = processor.run_uproot_job(
        fileset,
        "Runs",
        meta_processor(accumulator=meta_accumulator),
        processor.futures_executor,
        {'workers': workers, "skipbadfiles":skipbadfiles ,},
        chunksize=100000,
    )

    # now clean up the output, get skipped files per data set
    meta = {}

    for sample in fileset:
        meta[sample] = meta_output[sample]
        good_files = []
        skipped_files = []
        for rootfile in fileset[sample]:
            if meta_output[rootfile]:
                good_files.append(rootfile)
            else:
                skipped_files.append(rootfile)
        meta[sample]['good_files'] = good_files
        meta[sample]['n_good'] = len(good_files)
        meta[sample]['bad_files'] = skipped_files
        meta[sample]['n_bad'] = len(skipped_files)
        meta[sample]['xsec'] = samples[sample]['xsec']

    return meta


if __name__ == '__main__':

    from Tools.helpers import get_samples
    from Tools.config_helpers import redirector_ucsd, redirector_fnal
    from Tools.nano_mapping import make_fileset, nano_mapping

    
    samples = get_samples()

    fileset = make_fileset(['QCD'], samples, redirector=redirector_ucsd, small=False)

    meta = get_sample_meta(fileset)

    import pandas as pd
    df = pd.DataFrame(meta)
    print (df.transpose())
