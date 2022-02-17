#!/usr/bin/env python3

from Tools.helpers import dasWrapper

new_campaign = "RunIISummer20UL18NanoAODv9*"

with open('../data/samples_UL18.txt', 'r') as f:
    lines = f.readlines()



    for line in lines:

        old_sample, xsec = line.split()

        print (f"Old sample: {old_sample}")

        if old_sample.count('hadoop'):
            continue
        else:
            _, sample, campaign, tier = old_sample.split('/')

        das = f"/{sample}/{new_campaign}/{tier}"
        query_result = dasWrapper(das, query='')
        print ("New sample:")
        print (query_result[0], xsec)
