#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:16:07 2022

@author: tim hartill

Combine hpqa and hover train and dev splits

Notes
- hpqa comparison samples have no 'bridge' key
- hpqa samples have no 'src' key
- The mhop_dataset_var get_item process handles this so no need to 'fix' here

"""

import os
import json
import random
import numpy as np
from html import unescape

import utils


hover_dev = utils.load_jsonl('/home/thar011/data/baleen_downloads/hover/hover_dev_with_neg_and_sent_annots.jsonl')
hover_train = utils.load_jsonl('/home/thar011/data/baleen_downloads/hover/hover_train_with_neg_and_sent_annots.jsonl')

hpqa_dev = utils.load_jsonl('/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl')
hpqa_train = utils.load_jsonl('/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl')


hpqa_hover_dev = hover_dev + hpqa_dev
hpqa_hover_train = hover_train + hpqa_train

utils.saveas_jsonl(hpqa_hover_dev, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_dev_with_neg_v0_sentannots.jsonl')
utils.saveas_jsonl(hpqa_hover_train, '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_train_with_neg_v0_sentannots.jsonl')


