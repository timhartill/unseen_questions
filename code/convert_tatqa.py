#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:14:00 2022

Convert TAT-QA into standard format
TAT-QA repo: https://github.com/NExTplusplus/TAT-QA


@author: tim hartill


"""
import os
import json
import random
import itertools

from tqdm import tqdm
import numpy as np
from html import unescape
#import pandas as pd

import utils

OUT_DIR = '/data/thar011/data/unifiedqa/'

TAT_DEV = '/home/thar011/data/tat_qa/tatqa_dataset_dev.json'
TAT_TRAIN = '/home/thar011/data/tat_qa/tatqa_dataset_train.json'

dev = json.load(open(TAT_DEV))  #278
train = json.load(open(TAT_TRAIN))  #2201

def build_table(table):
    """ Linearize table
    """
    table_str = ''
    for i in range(len(table)):
        table_str += 'row '
        if i == 0:
            table_width = len(table[0])
        if len(table[i]) != table_width:
            print(f"Warning: differing table widths! {table}")  # Never happens
        for j in range(len(table[i])):
            if table[i][j].strip() == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                table[i][j] = "none"
            if j+1 == len(table[i]):    
                table_str += table[i][j].strip() + ' '
            else:
                table_str += table[i][j].strip() + ' | '
    return table_str

            
def build_context(split):
    """ Add 'context' key to each sample
    """                
    for sample in tqdm(split):
        table_str = build_table(sample['table']['table'])
        table_str += 'Notes: '
        for para in sample['paragraphs']:
            txt = para['text'].strip()
            if txt[-1] not in ['.', '?','!', ':', ';']:
                txt += '.'
            table_str += txt + ' '
        sample['context'] = table_str.strip()
    return


def write_std(split, out_dir, out_file):
    """ output multiple samples per context
    Note: where there are multiple answers generally it needs to predict all of them not any of them 
          so outputing as list of comma-delimited strings for all permutations
    """
    out_list = []
    for sample in split:
        for qa in sample['questions']:
            q = qa['question']
            if type(qa['answer']) == list:
                if len(qa['answer']) == 1:
                    ans = str(qa['answer'][0]).strip()
                else:
                    ans = [str(a).strip() for a in qa['answer']]
                    all_permutations = list(itertools.permutations(ans))
                    a_list = []
                    for spanlist in all_permutations:
                        a_list.append( ', '.join(spanlist) )
                    ans = a_list
                #ans = ', '.join([str(a).strip() for a in qa['answer']])
            else:
                ans = str(qa['answer']).strip()
            out_list.append( utils.create_uqa_example(q, sample['context'], ans, append_q_char='?')   )
    out_dir = os.path.join(out_dir, "tatqa")
    print(f'Outputting to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, out_file)
    print(f"Outputting: {outfile}")
    with open(outfile, 'w') as f:
        f.write(''.join(out_list))
    print('Finished!')
    return

        
build_context(dev)
build_context(train)

write_std(dev, OUT_DIR, 'dev.tsv')
write_std(train, OUT_DIR, 'train.tsv')

