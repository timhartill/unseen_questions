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

# LLM neg rationales
file_rr_dev_negs = ['/large_data/thar011/out/mdr/logs/LLM_TEST28_SPANYN_hpqa_dev_using_muv2_1krandord-02-02-2023-LLM-bigscience-bloom-maxsmpls1000-randTrue/llm_samples_with_context.json', 
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T43_HPQA_R4C_DEV_onv6_sample-03-02-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T44_HPQA_R4C_DEV_onv6mod2_sample-03-03-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
               ]
file_rr_train_negs = ['/large_data/thar011/out/mdr/logs/LLM_TEST29_SPANYN_hpqa_train_using_muv2_10krandord-02-03-2023-LLM-bigscience-bloom-maxsmpls10000-randTrue/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T45_HPQA_R4C_TRAIN_onv6_sample-03-04-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T46_HPQA_R4C_TRAIN_onv6mod2_sample-03-05-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 ]

# the full final sample files with potentially multiple pos and negs including samples without negs:
rr_dev = '/home/thar011/data/hpqa/hotpot_dev_rr_all_pos_neg.jsonl'
rr_train = '/home/thar011/data/hpqa/hotpot_train_rr_all_pos_neg.jsonl'
# the full final sample files with potentially multiple pos and negs excluding samples without negs:
rr_dev_exclposonly = '/home/thar011/data/hpqa/hotpot_dev_rr_all_pos_neg_exclposonly.jsonl'
rr_train_exclposonly = '/home/thar011/data/hpqa/hotpot_train_rr_all_pos_neg_exclposonly.jsonl'


dev_rr_format = utils.load_jsonl(file_hpqa_dev) # 7405
train_rr_format = utils.load_jsonl(file_hpqa_train) # 90447

dev_rr_format_r4c = utils.load_jsonl(file_r4c_dev) # 2209
train_rr_format_r4c = utils.load_jsonl(file_r4c_train) # 2379

dev_rr_format = utils.merge_pos_into_rr(dev_rr_format, dev_rr_format_r4c)
train_rr_format = utils.merge_pos_into_rr(train_rr_format, train_rr_format_r4c)

#merge negs - eliminate stemmed negs which contain F1 > thresh or EM over the stemmed answer string - do this PER PROMPT since ans in prompt is ok for negation prompts
# merge routine to align pos and negs
dev_rr_format = utils.load_merge_negs(dev_rr_format, file_rr_dev_negs, overlap_method='f1')
utils.saveas_jsonl(dev_rr_format, rr_dev)

train_rr_format = utils.load_merge_negs(train_rr_format, file_rr_train_negs, overlap_method='f1')
utils.saveas_jsonl(train_rr_format, rr_train)

utils.output_neg_tsv(dev_rr_format, os.path.join(UQA_DIR, 'hpqa_neg_expl_ans'), 'dev.tsv')
utils.output_neg_tsv(train_rr_format, os.path.join(UQA_DIR, 'hpqa_neg_expl_ans'), 'train.tsv')

# save final rr model creak training dataset - only output where negs exist which is all of them in this case but for consistency and debug..
utils.output_rr_where_negs_exist(dev_rr_format, outfile=rr_dev_exclposonly)  #2399
utils.output_rr_where_negs_exist(train_rr_format, outfile=rr_train_exclposonly) #8025

#TODO - evaluate..
# how many yes/nos have VALID neg rationales?
# could I get more if look at exact match?

#TODO for MC datasets - augment pos paras with "the answer must be" and 'thus of the choices'? QA model llm_expl datsets have these forms so do for pos_paras and neg_paras. Do sampled pos & negs give these - NOT enough?
#TODO for MC datasets - input with and without MC options into rr reranker - but no other variations identified so do dynamically in the rr model dataloader








