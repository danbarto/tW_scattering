#!/usr/bin/env python3

from Tools.helpers import dasWrapper

new_campaign = "RunIISummer20UL16NanoAODAPVv9*"
#new_campaign = "Run2018A-UL2018_MiniAODv*_NanoAODv*"

with open('../data/samples_UL18.txt', 'r') as f:
    lines = f.readlines()



    for line in lines:

        old_sample, xsec, reweight = line.split()

        print (f"Old sample: {old_sample}")

        if old_sample.count('hadoop'):
            continue
        else:
            _, sample, campaign, tier = old_sample.split('/')

        das = f"/{sample}/{new_campaign}/{tier}"
        query_result = dasWrapper(das, query='')
        print (f"Found {len(query_result)} new samples.")
        for new_sample in query_result:
            print (new_sample, xsec, reweight)
        #print ("New sample:")
        #print (query_result[0], xsec)
