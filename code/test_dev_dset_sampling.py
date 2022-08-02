#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:22:18 2022

@author: thar011
"""

data = {}
for dataset, datasetlen in [('DSET_500', 500), ('DSET_999', 999), 
                            ('DSET_1000', 1000), ('DSET_1499', 1499),
                            ('DSET_1500', 1500), ('DSET_1999', 1999),
                            ('DSET_2100', 2100), ('DSET_2520', 2520),
                            ('DSET_4123', 4123), ('DSET_10900', 10900),
                            ('DSET_19123', 19123), ('DSET_51234', 51234)]:
    data[dataset] = {"id": ['x']*datasetlen, "question": ['q']*datasetlen, "answer": ['a']*datasetlen}
    

def sample_datasets(data, approx_dev_samples):
    """ Restrict size of each dev dataset to ~approx_dev_samples
    """
    sampled_data = {}
    for dataset in data.keys():
        datasetlen = len(data[dataset]["question"])
        n_skip = round(datasetlen / approx_dev_samples)
        if n_skip <= 1:   
            sampled_data[dataset] = data[dataset]
        else:
            sampled_data[dataset] = {"id": [], "question": [], "answer": []}
            for i in range(0, datasetlen, n_skip):
                 sampled_data[dataset]["id"].append( data[dataset]["id"][i] )    
                 sampled_data[dataset]["question"].append( data[dataset]["question"][i] )    
                 sampled_data[dataset]["answer"].append( data[dataset]["answer"][i] )  
    for dataset in sampled_data.keys():
        print(f"{dataset}: New count:{len(sampled_data[dataset]['question'])} Orig:{len(data[dataset]['question'])}")
    return sampled_data
                


sd = sample_datasets(data, approx_dev_samples=1000)
sd = sample_datasets(data, approx_dev_samples=1250)
sd = sample_datasets(data, approx_dev_samples=1500)

data = sample_datasets(data, approx_dev_samples=1000)
