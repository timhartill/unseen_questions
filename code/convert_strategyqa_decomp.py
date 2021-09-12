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

Note: UQA_SQA_FACTS_DIR must end with _selfsvised - this signals the train/inference programs to treat this as a self supervised task

"""

import os
import json
import numpy as np
from utils import flatten, load_jsonl

#UQA_DIR = '/data/thar011/data/unifiedqa/'
#UQA_SQA_FACTS_DIR = 'strategy_qa_facts_selfsvised'
#UQA_SQA_Q_DIR = 'strategy_qa' 
SQA_DIR_IN = '/data/thar011/data/strategyqa/'
SQA_TRAIN_FILE = 'strategyqa_train.json'
SQA_PARA_FILE = 'strategyqa_train_paragraphs.json'

SQA_REPO_GENERATED_DIR = '/data/thar011/gitrepos/strategyqa/data/strategyqa/generated/'
SQA_REPO_GENERATED_DEV_FILE = 'transformer_qa_ORA-P_dev_no_placeholders.json'
SQA_REPO_GENERATED_TRAIN_FILE = 'transformer_qa_ORA-P_train_no_placeholders.json'
SQA_REPO_GENERATED_DEV_DECOMP_FILE = 'bart_decomp_dev_predictions.jsonl'

with open(os.path.join(SQA_DIR_IN, SQA_TRAIN_FILE),'r') as f:
    sqa_train = json.load(f)  #2290 questions

with open(os.path.join(SQA_DIR_IN, SQA_PARA_FILE),'r') as f:
    sqa_para = json.load(f)  #9251 paragraphs
    
with open(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_DEV_FILE),'r') as f:
    sqa_repo_dev = json.load(f)    #229
    
with open(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_TRAIN_FILE),'r') as f:
    sqa_repo_train = json.load(f)    #2061 + 229 = 2290
    
sqa_repo_dev_decomp = load_jsonl(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_DEV_DECOMP_FILE))  #229


def replace_chars(instr): 
    outstr = instr.replace("’", "'")
    return outstr.replace('“', '"').replace('”','"').replace("\t", " ").replace("\n", "")


def retrieve_paras(para_keys):
    """ Return para corresponding to keys """
    para_list = []
    for pk in para_keys:
        para = sqa_para.get(pk)
        if para is not None:
            para_list.append(para["content"])
    return para_list


tst = sqa_train[0]
tst['decomposition']
tst['evidence']
len(tst['evidence']) == 3  # num annotators
flatten(tst['evidence'])
ann1 = tst['evidence'][0]
len(ann1) == len(tst['decomposition'])  # 1 entry per decomp
decomp_evidence_para_keys = [[] for d in range(len(tst['decomposition']))] # [[], [], []]
for j, ann in enumerate(tst['evidence']):
    for i, e in enumerate(ann):
        para_keys = flatten(e)
        decomp_evidence_para_keys[i].extend(para_keys)
        print(f"Annotator:{j} Decomp:{i}: {tst['decomposition'][i]} {e} gold paras flattened:{para_keys}")
decomp_evidence_para_keys = [list(set(pk)) for pk in decomp_evidence_para_keys]
decomp_evidence_para_keys[0]
decomp_evidence_para_keys[0][::-1]

para_list = retrieve_paras(decomp_evidence_para_keys[0])
' '.join(para_list)
' '.join(para_list[::-1])
    
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






