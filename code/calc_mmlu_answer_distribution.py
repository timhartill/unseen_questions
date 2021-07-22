#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:30:15 2021

@author: Tim Hartill

Calculate distribution of answer choices across mmlu datasets 

"""

import os
import pandas as pd
import json

def count_dset(df, dsname):
    dsname = os.path.splitext(dsname)[0]
    dsstats = {'ds_name':dsname, 
               'ds_num': df.shape[0], 
               'option_counts':{'A':0, 'B':0, 'C':0, 'D':0},
               'option_percents': {'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0}}
    for j, row in enumerate(df.itertuples(index=False)):
        numcols = len(row)
        answer_char = row[numcols-1]
        #answer_idx = int(chr(ord(answer_char) - 17))
        dsstats['option_counts'][answer_char] += 1
    dsstats['option_percents']['A'] = dsstats['option_counts']['A'] / dsstats['ds_num']
    dsstats['option_percents']['B'] = dsstats['option_counts']['B'] / dsstats['ds_num']
    dsstats['option_percents']['C'] = dsstats['option_counts']['C'] / dsstats['ds_num']
    dsstats['option_percents']['D'] = dsstats['option_counts']['D'] / dsstats['ds_num']
    return dsstats

indir = '/data/thar011/data/mmlu/data/test/'
dsets = os.listdir(indir)
outdir = '/data/thar011/data/mmlu/data/'

#df = pd.read_csv(indir+dsets[0], header=None)  # test
#convert_dset(df, dsets[0], outdir)

dsets_stats = []
dsets_totals = {'ds_name':'TOTALS', 
               'ds_num': 0, 
               'option_counts':{'A':0, 'B':0, 'C':0, 'D':0},
               'option_percents': {'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0}}
for dset in dsets:
    print(f'Processing: {dset} ...')
    df = pd.read_csv(indir+dset, header=None)  
    dsstats = count_dset(df, dset)
    dsets_stats.append(dsstats)
    dsets_totals['ds_num'] += dsstats['ds_num']
    dsets_totals['option_counts']['A'] += dsstats['option_counts']['A']
    dsets_totals['option_counts']['B'] += dsstats['option_counts']['B']
    dsets_totals['option_counts']['C'] += dsstats['option_counts']['C']
    dsets_totals['option_counts']['D'] += dsstats['option_counts']['D']
        
dsets_totals['option_percents']['A'] = dsets_totals['option_counts']['A'] / dsets_totals['ds_num']
dsets_totals['option_percents']['B'] = dsets_totals['option_counts']['B'] / dsets_totals['ds_num']
dsets_totals['option_percents']['C'] = dsets_totals['option_counts']['C'] / dsets_totals['ds_num']
dsets_totals['option_percents']['D'] = dsets_totals['option_counts']['D'] / dsets_totals['ds_num']
    
print('Finished Processing!') 

with open(outdir+'test_ds_counts.json', 'w') as f:
    json.dump([dsets_stats, dsets_totals], f)
print(f"Saved stats to {outdir+'test_ds_counts.json'}")    

print("MMLU Dataset Statistics")
print(f"{dsets_totals['ds_name']}: Count: {dsets_totals['ds_num']}  A: {dsets_totals['option_percents']['A']*100:.2f}%  B: {dsets_totals['option_percents']['B']*100:.2f}%  C: {dsets_totals['option_percents']['C']*100:.2f}%  D: {dsets_totals['option_percents']['D']*100:.2f}%")
for i, dset in enumerate(dsets_stats):
    print(f"{dset['ds_name']}: Count: {dset['ds_num']}  A: {dset['option_percents']['A']*100:.2f}%  B: {dset['option_percents']['B']*100:.2f}%  C: {dset['option_percents']['C']*100:.2f}%  D: {dset['option_percents']['D']*100:.2f}%")






