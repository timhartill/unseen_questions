#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:43:19 2022

@author: tim hartill

Aggregate datasets with sentence annotations into single train / dev ie fever + hover + hpqa

"""
import os
import utils


FEVER_TRAIN = '/home/thar011/data/fever/fever_train_with_sent_annots.jsonl' #NOTE: Original with some mis-labelled gold sents
FEVER_DEV = '/home/thar011/data/fever/fever_dev_with_sent_annots.jsonl'     #NOTE: Original with some mis-labelled gold sents

# For cleaner training set use these instead for fever:
FEVER_TRAIN_V3 = '/home/thar011/data/fever/fever_train_with_sent_annots_labelfixes_singleonly_v3.jsonl'
FEVER_DEV_V3 = '/home/thar011/data/fever/fever_dev_with_sent_annots_labelfixes_singleonly_v3.jsonl'    

HOVER_TRAIN = '/home/thar011/data/baleen_downloads/hover/hover_train_with_neg_and_sent_annots.jsonl'
HOVER_DEV = '/home/thar011/data/baleen_downloads/hover/hover_dev_with_neg_and_sent_annots.jsonl'

HPQA_TRAIN = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl'
HPQA_DEV = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl'

SENT_DIR = '/home/thar011/data/sentences/'
SENT_TRAIN = os.path.join(SENT_DIR, 'sent_train.jsonl')
SENT_DEV = os.path.join(SENT_DIR, 'sent_dev.jsonl')
os.makedirs(SENT_DIR, exist_ok=True)


fever_train = utils.load_jsonl(FEVER_TRAIN)
fever_dev = utils.load_jsonl(FEVER_DEV)

fever_train_v3 = utils.load_jsonl(FEVER_TRAIN_V3)
fever_dev_v3 = utils.load_jsonl(FEVER_DEV_V3)

hover_train = utils.load_jsonl(HOVER_TRAIN)
hover_dev = utils.load_jsonl(HOVER_DEV)

hpqa_train = utils.load_jsonl(HPQA_TRAIN)
hpqa_dev = utils.load_jsonl(HPQA_DEV)


sent_train = hover_train + hpqa_train + fever_train  # 239276
sent_dev = hover_dev + hpqa_dev + fever_dev          # 26587

utils.saveas_jsonl(sent_train, SENT_TRAIN)
utils.saveas_jsonl(sent_dev, SENT_DEV)

sent_train = hover_train + hpqa_train + fever_train_v3  # 205320
sent_dev = hover_dev + hpqa_dev + fever_dev_v3          # 23495

SENT_TRAIN = os.path.join(SENT_DIR, 'sent_train_feversingleonlyv3.jsonl')
SENT_DEV = os.path.join(SENT_DIR, 'sent_dev_feversingleonlyv3.jsonl')


utils.saveas_jsonl(sent_train, SENT_TRAIN)
utils.saveas_jsonl(sent_dev, SENT_DEV)

