#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:41:11 2022

Convert POET-SQL to standard format

@author: tim hartill


"""
import os
import json
import random
import smart_open  # import open
from tqdm import tqdm
import numpy as np
from html import unescape

import utils

OUT_DIR = '/data/thar011/data/unifiedqa/'

POET_DEV = '/home/thar011/data/poet/POET-SQL-Corpus/POET-SQL/dev.json'
POET_TRAIN = '/home/thar011/data/poet/POET-SQL-Corpus/POET-SQL/train.json'


dev = utils.load_jsonl(POET_DEV)        # 21668
train = utils.load_jsonl(POET_TRAIN)    # 3937715

def add_question_context(split):
    """ Add keys for question, context, answer
    """
    top_dict = {}
    multi = {'multi':0, 'single':0}
    cat_dict = {}
    out_dict = {}
    for i, s in enumerate(split):
        context_start = s['input'].find(' col :')
        if context_start == -1:
            print(f"{i} col : not found")
            break
#        row_start = s['input'].find(' row ')
#        if row_start < context_start:
#            print(f"{i} col : {context_start} after row : {row_start}")
        s['question'] = s['input'][:context_start]
        s['context'] = s['input'][context_start:]
        s['sel_type'] = ' '.join(s['question'].split()[:2])
        if top_dict.get(s['sel_type']) is None:
            top_dict[s['sel_type']] = 0
        top_dict[s['sel_type']] += 1    
            
        s['ans_type'] = 'multi' if ', ' in s['output'] else 'single'
        multi[s['ans_type']] += 1
        
        if s['sel_type'] in ['select count', 'select sum', 'select max', 'select min', 'select abs']:
            s['category'] = s['sel_type']
        elif s['sel_type'] == 'select (':
            s['category'] = 'select arith'
        else:
            s['category'] = s['ans_type']
        s['category'] = s['category'].replace(' ', '_')
        if cat_dict.get(s['category']) is None:
            cat_dict[s['category']] = 0
        cat_dict[s['category']] += 1
        
        if out_dict.get(s['category']) is None:
            out_dict[s['category']] = []
        out_dict[s['category']].append( utils.create_uqa_example(s['question'], s['context'], s['output'], append_q_char='?')   )
            
        if i % 500000 == 0:
            print(f"Processed: {i}")
    print(f'Multi/single answers: {multi}')
    top_dict_list = list(top_dict.items())
    top_dict_list.sort(key=lambda s : s[1], reverse=True)
    print(f'Top 20 select types: {top_dict_list[:20]}')
    print(f'Processed categories: {cat_dict}')
    return out_dict


def write_std(out_dev, out_train):
    """ Output in std format
    """
    datasets = list(out_train.keys())
    for dataset in datasets:
        out_dir = os.path.join(OUT_DIR, "poetsql_" + dataset)
        print(f'Outputting to {out_dir}')
        os.makedirs(out_dir, exist_ok=True)
        outfile = os.path.join(out_dir, 'train.tsv')
        print(f"Outputting train: {outfile}")
        with open(outfile, 'w') as f:
            f.write(''.join(out_train[dataset]))
        outfile = os.path.join(out_dir, 'dev.tsv')
        print(f"Outputting dev: {outfile}")
        with open(outfile, 'w') as f:
            f.write(''.join(out_dev[dataset]))
    print('Finished!')
    return



out_dev = add_question_context(dev)
out_train = add_question_context(train)

out_dev.keys() == out_train.keys()  #TRUE

write_std(out_dev, out_train)


