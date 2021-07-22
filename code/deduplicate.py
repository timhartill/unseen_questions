#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:00:35 2021

@author: thar011

Some eval datasets, notably DROP (802 dups) and CONTRAST_SETS_DROP (just 2 DUPs) contain a number of duplicates
Note: A few train datasets also contain dups but these are not removed.

Note 2 : This utility also erroneously dedups some train_meta.tsv files. These are not used but could be manually deleted from the _dedup directories to tidy up

Edit DATASET_DIR before running.. 

"""
import os

DATASET_DIR = '/data/thar011/data/unifiedqa/'

def dedup(dataset):
    indir = os.path.join(DATASET_DIR,dataset)
    files = os.listdir(indir)
    files = [f for f in files if f.endswith('.tsv')]
    outdir = os.path.join(DATASET_DIR,dataset+'_dedup')
    for file in files:
        original = []
        with open(os.path.join(indir, file), 'r') as f:
            for line in f:
                original.append(line)
        unique = set(original)
        print(f'{dataset} {file}: original count:{len(original)} de-duplicated count:{len(unique)}')
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, file), 'w') as f:
            f.write(''.join(unique))
        print(f'Deduplicated file written to: {os.path.join(outdir, file)}')

            
dedup('drop')                   # over 800 dev dup
dedup('contrast_sets_drop')      # 2 dev dup
dedup('contrast_sets_boolq')    # 2 dev dup
dedup('boolq_np')               #48 dev dup
dedup('ambigqa')                #277 dev dup
dedup('social_iqa')             #19 dev dup
dedup('quoref')                     # just 1 dev dup
dedup('contrast_sets_quoref')       # just 2 dev dup
dedup('contrast_sets_ropes')       # just 1 dev dup
dedup('mmlu_us_foreign_policy_test')       # just 1 test dup
dedup('mmlu_college_physics_test')       # 11 test dup
dedup('mmlu_public_relations_test')       # 2 test dup
dedup('mmlu_high_school_psychology_test')       # 11 test dup
dedup('mmlu_professional_psychology_test')       # 2 test dup
dedup('mmlu_elementary_mathematics_test')       # 2 test dup





