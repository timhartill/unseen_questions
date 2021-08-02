#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:19:07 2021

@author: tim hartill

Convert wikipedia dump into text file of form para1 \\n \npara2 \\n n...

"""

import json
import numpy as np
import os

wikifile = '/data/thar011/data/strategyqa/enwiki-20200511-cirrussearch-parasv2.jsonl'
outdir = '/data/thar011/data/unifiedqa/enwiki-20200511_selfsvised'

def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    num_jsons = len(all_json_list)
    if verbose:
        print('JSON as text successfully loaded. Number of json messages in file is ', num_jsons)
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print('Text successfully converted to JSON.')
    return all_json_list

all_json_list2 = load_jsonl(wikifile)

print('Loaded, structure = ', all_json_list2[0].keys())  # dict_keys(['title', 'section', 'headers', 'para', 'docid', 'secid', 'headerid', 'para_id'])
num_q = len(all_json_list2)   # 36617357
os.makedirs(outdir, exist_ok=True)
np.random.seed(42)
dev_indices = np.random.choice(num_q, 500, replace=False)
outlist_dev = []
outlist_train = []
print('Splitting into train and 500 dev samples..')
for i in range(num_q):
    if i in dev_indices:
        outlist_dev.append(all_json_list2[i]['para'])
    else:
        outlist_train.append(all_json_list2[i]['para'])
outfile = os.path.join(outdir, 'train.tsv')
print(f'Saving to {outfile}')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(outlist_train))
outfile = os.path.join(outdir, 'dev.tsv')
print(f'Saving to {outfile}')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(outlist_dev))
print('Finished!')


