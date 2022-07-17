#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:16:07 2022

@author: tim hartill

Combine hpqa and hover etc train and dev splits for retriever model training

Notes
- hpqa comparison samples have no 'bridge' key
- hpqa samples have no 'src' key
- nq examples have 'id' key instead of '_id'
- nq & musique examples pos_paras have no sentence_labels or sentence_spans so can only use retriever query 'para' encoding 
- The mhop_dataset_var get_item process handles these differences where needed so not standardized here

"""

import os
import json
import random
import numpy as np
from html import unescape

import utils


# dict_keys(['question', 'answers', 'src', 'type', '_id', 'bridge', 'num_hops', 'pos_paras', 'neg_paras'])
hover_dev = utils.load_jsonl('/home/thar011/data/baleen_downloads/hover/hover_dev_with_neg_and_sent_annots.jsonl')
hover_train = utils.load_jsonl('/home/thar011/data/baleen_downloads/hover/hover_train_with_neg_and_sent_annots.jsonl')

# comparison q: dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id']) optional 'bridge', no 'src'
hpqa_dev = utils.load_jsonl('/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl')
hpqa_train = utils.load_jsonl('/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl')

# dict_keys(['question', 'answers', 'id', 'type', 'src', 'para_agg_map', 'bridge', 'pos_paras', 'neg_paras']) 'id' not '_id'. no sentence labels so must use query para enc in retriever training, para_agg_map unused
nq_dev = utils.load_jsonl('/home/thar011/data/DPR/nq_dev_v1.0_with_neg_v0.jsonl')
nq_train = utils.load_jsonl('/home/thar011/data/DPR/nq_train_v1.0_with_neg_v0.jsonl')

# dict_keys(['question', 'answers', 'src', 'type', '_id', 'pos_paras', 'neg_paras', 'bridge'])
mu_dev = utils.load_jsonl('/home/thar011/data/musique/musique_v1.0/data/musique_ans_v1.0_dev_retriever_new_with_hl_negs_v0.jsonl')
mu_train = utils.load_jsonl('/home/thar011/data/musique/musique_v1.0/data/musique_ans_v1.0_train_retriever_full_with_hl_negs_v0.jsonl')




hpqa_hover_dev = hover_dev + hpqa_dev
hpqa_hover_train = hover_train + hpqa_train

utils.saveas_jsonl(hpqa_hover_dev, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_dev_with_neg_v0_sentannots.jsonl')
utils.saveas_jsonl(hpqa_hover_train, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_train_with_neg_v0_sentannots.jsonl')

hpqa_hover_nq_dev = hpqa_hover_dev + nq_dev
hpqa_hover_nq_train = hpqa_hover_train + nq_train

utils.saveas_jsonl(hpqa_hover_nq_dev, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_dev_with_neg_v0.jsonl')
utils.saveas_jsonl(hpqa_hover_nq_train, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_train_with_neg_v0.jsonl')

hpqa_hover_nq_mu_dev = hpqa_hover_nq_dev + mu_dev
hpqa_hover_nq_mu_train = hpqa_hover_nq_train + mu_train

utils.saveas_jsonl(hpqa_hover_nq_mu_dev, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_dev_with_neg_v0.jsonl')
utils.saveas_jsonl(hpqa_hover_nq_mu_train, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_train_with_neg_v0.jsonl')

hpqa_hover_mu_dev = hpqa_hover_dev + mu_dev
hpqa_hover_mu_train = hpqa_hover_train + mu_train

utils.saveas_jsonl(hpqa_hover_mu_dev, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_mu_dev_with_neg_v0.jsonl')
utils.saveas_jsonl(hpqa_hover_mu_train, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_mu_train_with_neg_v0.jsonl')


