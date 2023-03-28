#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:58:13 2023

@author: tim hartill

Combine pos + neg rationales from:

pos:    
- hpqa gold sentences (90K)
- r4c rationales (2.3K)

neg:
- r4c negs from llm neg prompt (T43,44,45,46)
- hpqa negs from 10K llm samples many of which are totally wrong hence useful as negs. (T28,29)



"""

import os
import json
import copy
import random
from tqdm import tqdm
from html import unescape

import eval_metrics
import utils
import text_processing


UQA_DIR = eval_metrics.UQA_DIR


# Base files: all hpqa samples each with 1 pos made of gold sents:
file_hpqa_dev = '/home/thar011/data/hpqa/hotpot_dev_fullwiki_v1_rr_pos.jsonl'
file_hpqa_train = '/home/thar011/data/hpqa/hotpot_train_fullwiki_v1_rr_pos.jsonl'

# r4c subset of hpqa: each with 3 pos
file_r4c_dev = '/home/thar011/data/hpqa_r4c/r4c_pos_paras_dev_csf.jsonl'
file_r4c_train = '/home/thar011/data/hpqa_r4c/r4c_pos_paras_train.jsonl'

dev_rr_format = utils.load_jsonl(file_hpqa_dev) # 7405
train_rr_format = utils.load_jsonl(file_hpqa_train) # 90447

dev_rr_format_r4c = utils.load_jsonl(file_r4c_dev) # 2209
train_rr_format_r4c = utils.load_jsonl(file_r4c_train) # 2379

dev_rr_format = utils.merge_pos_into_rr(dev_rr_format, dev_rr_format_r4c)
train_rr_format = utils.merge_pos_into_rr(train_rr_format, train_rr_format_r4c)

# TODO add r4c negs incl 'good' neg elimination
#TODO add hpqa llm negs


#TODO for other datasets - eliminate negs which contain F1 > thresh over the stemmed answer string - do this PER PROMPT since ans in prompt is ok for negation prompts
#TODO for MC datasets - augment pos paras with "the answer must be" and 'thus of the choices'? QA model llm_expl datsets have these forms so do for pos_paras and neg_paras. Do sampled pos & negs give these - NOT enough?
#TODO for MC datasets - input with and without MC options into rr reranker - but no other variations identified so do dynamically in the rr model dataloader








