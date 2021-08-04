#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:23:12 2021

@author: Tim Hartill

Combine strategy_qa_facts_selfsvised dev.tsv into train.tsv in a new dset
for the purposes of testing the impact on strategy_qa dev.tsv questions.

Edit files and dirs before running..

After running, manually copy dev.tsv over unchanged into the new directory...

"""
import os

in_dset = 'strategy_qa_facts_selfsvised'
out_dset = 'strategy_qa_facts_dev_in_train_selfsvised'

uqa_dir = '/data/thar011/data/unifiedqa/'

in_dir = os.path.join(uqa_dir, in_dset)
agg_file_list = ['dev.tsv', 
                 'train.tsv' ] 

agg_dset = ''
for i, file in enumerate(agg_file_list):
    in_file = os.path.join(in_dir, file)
    print(f'Opening {in_file}')
    with open(in_file, 'r') as f:
        in_dset = f.read()
    if i > 0:
        agg_dset += '\n'    
    agg_dset += in_dset

os.makedirs(os.path.join(uqa_dir, out_dset), exist_ok=True)
out_file = os.path.join(uqa_dir, out_dset, 'train.tsv')
with open(out_file, 'w') as f:
    f.write(agg_dset)



