#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:20:11 2021

@author: thar011

Convert Geva et al Numerical Literacy datasets into UnifiedQa format

From paper: Injecting numerical reasoning skills into language models

Edit indir and outdir before running...

"""

import os
import json
import shutil

indir = '/data/thar011/data/injecting_numeracy/data/'
dsets = os.listdir(indir)
outdir = '/data/thar011/data/injecting_numeracy/unifiedqa_format/'
os.makedirs(outdir, exist_ok=True)
unifiedqadir = '/data/thar011/data/unifiedqa/'  # NOT USED. Manually copy the resulting files after checking them...


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



numdata = os.path.join(indir, 'synthetic_numeric.jsonl')
num_outdir = outdir + 'synthetic_numeric'
os.makedirs(num_outdir, exist_ok=True)
txtdata_dev = os.path.join(indir, 'synthetic_textual_mixed_min3_max6_up0.7_dev.json')
txtdata_train = os.path.join(indir, 'synthetic_textual_mixed_min3_max6_up0.7_train.json')
txt_outdir = outdir + 'synthetic_textual'
os.makedirs(txt_outdir, exist_ok=True)

num = load_jsonl(numdata)
num[0].keys()  # ['id', 'expr', 'val', 'args', 'type', 'check_domain', 'split']
# split: {'train': 990000, 'dev': 9996}

fout_train = open(f"{num_outdir}/train.tsv", "w")
fout_dev = open(f"{num_outdir}/dev.tsv", "w")

for s in num:
    split = s['split'].strip()
    answer_text = str(s['val'])    
    if s['type'] == 'percent':
        expr = s['expr'].replace('::', 'of')
        question = 'What is ' + expr + '?'           
    else:    
        question = 'What is ' + s['expr'] + '?'   
    full_sample = f"{question.strip()} \\n \t{answer_text.strip()}\n"
    if split == 'train':
        fout_train.write(full_sample)  
    else:
        fout_dev.write(full_sample)  
fout_train.close()
fout_dev.close()


txtdrop = json.load(open(txtdata_dev))
fout_dev = open(f"{txt_outdir}/dev.tsv", "w")
for i, key in enumerate(txtdrop):
    context = txtdrop[key]['passage']
    for j, qa in enumerate(txtdrop[key]['qa_pairs']):  #all answers either an integer or textual
        question = qa['question']
        answer_text = str(qa['answer'])
        full_sample = f"{question.strip()} \\n {context.strip()}\t{answer_text.strip()}\n"
        fout_dev.write(full_sample)
fout_dev.close()

txtdrop = json.load(open(txtdata_train))
fout_train = open(f"{txt_outdir}/train.tsv", "w")
for i, key in enumerate(txtdrop):
    context = txtdrop[key]['passage']
    for j, qa in enumerate(txtdrop[key]['qa_pairs']):  #all answers either an integer or textual
        question = qa['question']
        answer_text = str(qa['answer'])
        full_sample = f"{question.strip()} \\n {context.strip()}\t{answer_text.strip()}\n"
        fout_train.write(full_sample)
fout_train.close()

# Manually copy the above files into the UnifiedQA data dir after checking them ...


#quick check on number of textual training samples:
k = 0
for i, key in enumerate(txtdrop):
    for j, qa in enumerate(txtdrop[key]['qa_pairs']):  #all answers either an integer or textual
        k += 1
print('Num train samples=',k)    # 2523192



