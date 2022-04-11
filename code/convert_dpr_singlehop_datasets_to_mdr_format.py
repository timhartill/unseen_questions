#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:11:34 2022

@author: tim hartill

Convert DPR versions of single hop datasets to "MDR" format to faciliate adding to training to bolster SQUAD-open retrieval performance

First download copies of the relevant files using the links found in https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py
Then extract to their json form.

"""

import os
import json


DPR_NQ_DEV = '/home/thar011/data/DPR/biencoder-nq-dev.json'
DPR_NQ_TRAIN = '/home/thar011/data/DPR/biencoder-nq-train.json'
DPR_NQ_TRAIN_ADV_NEGS = '/home/thar011/data/DPR/nq-train-dense-results_as_input_with_gold.json'

DPR_TREC_DEV = '/home/thar011/data/DPR/curatedtrec-dev.json'
DPR_TREC_TRAIN = '/home/thar011/data/DPR/curatedtrec-train.json'

DPR_TQA_DEV = '/home/thar011/data/DPR/triviaqa-dev_new.json'
DPR_TQA_TRAIN = '/home/thar011/data/DPR/triviaqa-train_new.json'

DPR_WQ_DEV = '/home/thar011/data/DPR/webquestions-dev.json'
DPR_WQ_TRAIN = '/home/thar011/data/DPR/webquestions-train.json'


nq_dev = json.load(open(DPR_NQ_DEV)) #6515 [ dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']) ]
nq_train = json.load(open(DPR_NQ_TRAIN)) #58880 [dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])]
nq_train_adv = json.load(open(DPR_NQ_TRAIN_ADV_NEGS))  # 69639  dict_keys(['question', 'answers', 'negative_ctxs', 'hard_negative_ctxs', 'positive_ctxs'])

score_1000_test = [n for n in nq_dev if n['positive_ctxs'][0]['score'] != 1000] # all 1st entires have score 1000
score_1000_test = [n for n in nq_dev if len( n['positive_ctxs'])>1 and n['positive_ctxs'][1]['score'] == 1000] # None of the 2nd paras score 1000
score_1000_test = [n for n in nq_dev if n['answers'][0] not in n['positive_ctxs'][0]['text']]  #894 but appear to be differences in comma placement

#TODO Use score 1000 entry as the positive for NQ
#  nq_train_adv is newer train set, gives perf boost. But why difft # training samples?

dev_questions = set([n['question'] for n in nq_dev])
train_adv_questions = set([n['question'] for n in nq_train_adv])

overlap = train_adv_questions.intersection(dev_questions)  # set() so use nq_train_adv

multi_answer_test = [n for n in nq_dev if len(n['answers']) > 1]  # 696 have >1 answer
multi_answer_test = [n for n in nq_train_adv if len(n['answers']) > 1] # 7190 have >1 answer

len(nq_train_adv[0]['positive_ctxs']) #6
len(nq_train_adv[0]['negative_ctxs']) #0
len(nq_train_adv[0]['hard_negative_ctxs']) #30

any_neg_ctxs = [n for n in nq_train_adv if len(n['negative_ctxs']) > 0]  # None so use hard_negative_ctxs!

no_hard_negs = [n for n in nq_train_adv if len(n['hard_negative_ctxs']) == 0]  # 3  maybe just exclude these or fill with randoms
few_hard_negs = [n for n in nq_train_adv if len(n['hard_negative_ctxs']) < 10]  # 28 maybe just exclude these

no_hard_negs = [n for n in nq_dev if len(n['hard_negative_ctxs']) == 0]  # 7
few_hard_negs = [n for n in nq_dev if len(n['hard_negative_ctxs']) < 10]  # 30 - top up with 'negative ctxs' Note: 'negative_ctxs are quite randon, hard_negative_ctxs are much closer

no_hard_negs = [n for n in nq_dev if len(n['negative_ctxs']) == 0]  # 0
few_hard_negs = [n for n in nq_dev if len(n['negative_ctxs']) < 10]  # 0

### TREC - SKIP #####
trec_dev = json.load(open(DPR_TREC_DEV))  # 116  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
trec_train = json.load(open(DPR_TREC_TRAIN))  # 1125  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

# note strange trec answer format for 1st dev sample - ['Long Island|New\\s?York|Roosevelt Field'] and passage format: 'Charles Lindbergh Charles Augustus Lindbergh (February 4, 1902 – August 26, 1974) was an American aviator, military officer, author, inventor, explorer, and environmental activist. At age 25 in 1927, he went from obscurity as a [[U.S. Air Mail]] pilot to instantaneous world fame by winning the [[Orteig Prize]]: making a nonstop flight from [[Roosevelt Field (airport)|Roosevelt Field]], [[Long Island]], [[New York (state)|New York]], to [[Paris]], France. Lindbergh covered the -hour, flight alone in a single-engine purpose-built [[Ryan Airline Company|Ryan]] [[monoplane]], the "[[Spirit of St. Louis]]". This was not the [[Transatlantic flight of Alcock and Brown|first flight between North America and'
# several train egs have invalid highest scoring positive para
# SKIP TREC!!

tqa_dev = json.load(open(DPR_TQA_DEV))  # 8837  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
tqa_train = json.load(open(DPR_TQA_TRAIN))  #78785  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

#tqa_dev[1] has 0 positive_ctxs.. so has tqa_train[0]  NEED TO SKIP THESE

wq_dev = json.load(open(DPR_WQ_DEV))  # 278  dict_keys(['question', 'answers', 'dataset', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
wq_train = json.load(open(DPR_WQ_TRAIN))  #2474 dict_keys(['question', 'answers', 'dataset', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

# 1st one had invalid highest scoring para - skip WQ!



