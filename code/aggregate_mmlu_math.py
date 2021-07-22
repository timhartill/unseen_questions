#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:23:12 2021

@author: Tim Hartill

Aggregate mmlu math orientated datasets into one of reasonable size.

Edit in_dir before running..

"""
import os

out_dset = 'mmlu_elementary_to_college_math_test'

in_dir = '/data/thar011/data/unifiedqa/'

agg_dset_list = ['mmlu_elementary_mathematics_test_dedup', 
                 'mmlu_high_school_mathematics_test',
                 'mmlu_high_school_statistics_test',
                 'mmlu_college_mathematics_test' ] 

agg_dset = ''
for dset in agg_dset_list:
    in_file = os.path.join(in_dir, dset, 'test.tsv')
    print(f'Opening {in_file}')
    with open(in_file, 'r') as f:
        in_dset = f.read()
    agg_dset += in_dset

os.makedirs(os.path.join(in_dir, out_dset), exist_ok=True)
out_file = os.path.join(in_dir, out_dset, 'test.tsv')
with open(out_file, 'w') as f:
    f.write(agg_dset)





