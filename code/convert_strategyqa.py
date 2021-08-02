#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:45:33 2021

@author: tim hartill

Convert StrategyQA into UnifiedQA-like format
Outputs two datasets:
    strategy_qa_facts_selfsvised: Individual paragraphs potentially used as evidence in reasoning for strategyQa questions
    strategy_qa: The actual strategyQA questions in MC format without any other context

StrategyQa from the paper:
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. 
Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. 
Transactions of the Association for Computational Linguistics, 9:346361.

Edit the directory and file names below before running...

UQA_SQA_FACTS_DIR must end with _selfsvised - this signals the train/inference programs to treat this as a self supervised task

"""

import os
import json
import numpy as np

UQA_DIR = '/data/thar011/data/unifiedqa/'
UQA_SQA_FACTS_DIR = 'strategy_qa_facts_selfsvised'
UQA_SQA_Q_DIR = 'strategy_qa' 
SQA_DIR_IN = '/data/thar011/data/strategyqa/'
SQA_TRAIN_FILE = 'strategyqa_train.json'
SQA_PARA_FILE = 'strategyqa_train_paragraphs.json'

with open(os.path.join(SQA_DIR_IN, SQA_TRAIN_FILE),'r') as f:
    sqa_train = json.load(f)  #2290 questions

with open(os.path.join(SQA_DIR_IN, SQA_PARA_FILE),'r') as f:
    sqa_para = json.load(f)  #9251 paragraphs


def flatten(alist):
    """ flatten a list of nested lists
    """
    t = []
    for i in alist:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t

def replace_chars(instr): 
    outstr = instr.replace("’", "'")
    return outstr.replace('“', '"').replace('”','"').replace("\t", " ").replace("\n", "")

    
# strategyQA has no dev split so create one as 10% of train
# Update the paragraph dict with which qids use each paragraph
num_q = len(sqa_train)
dev_size = int(num_q*0.1)
for p in sqa_para:
    sqa_para[p]['splits_used'] = set()
    sqa_para[p]['qids_used'] = []
np.random.seed(42)
dev_indices = np.random.choice(num_q, dev_size, replace=False)
for i in range(num_q):
    if i in dev_indices:
        sqa_train[i]['split'] = 'dev'
    else:    
        sqa_train[i]['split'] = 'train'
    sqa_train[i]['evidence_flattened'] = set(flatten(sqa_train[i]['evidence'])) #flatten evidence while we are at it       
    for e in sqa_train[i]['evidence_flattened']:
        if sqa_para.get(e) is not None:
            sqa_para[e]['splits_used'].add(sqa_train[i]['split'])
            sqa_para[e]['qids_used'].append(sqa_train[i]['qid'])

traincount = 0
devcount = 0
bothcount = 0
nonecount = 0
qidcounts = []
for p in sqa_para:
    if sqa_para[p]['splits_used'] == set():
        nonecount += 1
    elif sqa_para[p]['splits_used'] == {'dev', 'train'}:
        traincount += 1
        devcount += 1
        bothcount += 1
    elif sqa_para[p]['splits_used'] == {'dev'}:
        devcount += 1
    else:
        traincount += 1
    num_ref = len(sqa_para[p]['qids_used'])
    qidcounts.append(num_ref)
qids_np = np.array(qidcounts)
print(f"qids: num:{qids_np.shape[0]}  mean:{qids_np.mean():.2f}  max:{qids_np.max():.2f}  min:{qids_np.min():.2f}")
# qids: num:9251  mean:1.11  max:11.00  min:1.00
print(f"Counts: Unused:{nonecount}  Both train+Dev:{bothcount}  Train Only:{traincount}  Dev Only:{devcount}")
# Counts: Unused:0  Both train+Dev:161  Train Only:8402  Dev Only incl dev+train:1010  - dev only 849, train only 8241


# Create MLM task:
paras_dev = []    # paras needed by dev only
paras_train = []  # include paras needed by both train and dev
for p in sqa_para:
    if sqa_para[p]['splits_used'] == {'dev'}:
        paras_dev.append( replace_chars(sqa_para[p]['content'] ) )
    else:
        paras_train.append( replace_chars(sqa_para[p]['content'] ) )
        

outdir = os.path.join(UQA_DIR, UQA_SQA_FACTS_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_train))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_dev))

# create strategyQA question task:
qa_dev = []
qa_train = []
for qa in sqa_train:
    question = f"{replace_chars(qa['question'])} \\n (A) yes (B) no"
    if qa['answer']:
        answer = 'yes'
    else:
        answer = 'no'
    sample = f"{question}\t{answer}"
    if qa['split'] == 'train':
        qa_train.append(sample)
    else:
        qa_dev.append(sample)
print(f"train count: {len(qa_train)}  dev count: {len(qa_dev)}")    # train count: 2061  dev count: 229

outdir = os.path.join(UQA_DIR, UQA_SQA_Q_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write('\n'.join(qa_train))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write('\n'.join(qa_dev))






