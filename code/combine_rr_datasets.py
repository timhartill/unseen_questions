#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:42:46 2023

@author: tim hartill

Combine rr datasets into singe training file for rr model

rationale reranker 'rr' training format:

    Output format:
    [ {'question': 'question text EXCLUDING MC options and preceding initial ctxt if any',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'hpqa',
       'pos_paras': [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]]}, ...],
       'neg_paras': [], #Same format as pos_paras but filled in later
       'mc_options':  '(A) banana (B) ...'  # "" if multichoice options don't exist...
       'context': 'An initial para or other necessary context if exists'  # "" if initial para doesnt exist which is always in training...
       }, {...}, ..
     
    ]


"""
import os
import json
import utils
import random

RR_OUT_BASE = '/home/thar011/data/rationale_reranker/'

RR_IN_DEV = ['/home/thar011/data/creak/creak_dev_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/hpqa/hotpot_dev_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/eraser/fever/fever_dev_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/qasc/QASC_Dataset/qasc_dev_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/worldtree/WorldtreeExplanationCorpusV2.1_Feb2020/worldtree_dev_rr_all_pos_neg_exclposonly.jsonl',]

RR_IN_TRAIN = ['/home/thar011/data/creak/creak_train_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/hpqa/hotpot_train_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/eraser/fever/fever_train_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/qasc/QASC_Dataset/qasc_train_rr_all_pos_neg_exclposonly.jsonl',
             '/home/thar011/data/worldtree/WorldtreeExplanationCorpusV2.1_Feb2020/worldtree_train_rr_all_pos_neg_exclposonly.jsonl',]


def load_print_stats_combine(split_list, outfile):
    """ Load each file, print stats for each then combine into one file for output
    """
    outstatsfile = os.path.join(RR_OUT_BASE, outfile + '_counts.json')
    outfile = os.path.join(RR_OUT_BASE, outfile + '.jsonl')
    outlist = []
    stats = {}
    stats['TOTAL'] = {'raw': 0, 'pos':0, 'neg': 0, 'extpos': 0, 'extneg': 0}
    print("######################################")
    print(f"Combining for: {outfile}")
    
    for file in split_list:
        print("######################################")
        print(f"Loading: {file}")
        currlist = utils.load_jsonl(file)
        numsamples = len(currlist)
        numpos = sum([len(c['pos_paras']) for c in currlist])   # number of q[+mc]+pos samples
        numneg = sum([len(c['neg_paras']) for c in currlist])   # number of q[+mc]+neg samples
        hasmc = currlist[0].get('mc_options') is not None
        if not hasmc:
            for c in currlist:
                c['mc_options'] = ''  # for downstream consistency
        hascontext = currlist[0].get('context') is not None
        if not hascontext:
            for c in currlist:
                c['context'] = ''  # for downstream consistency. Note no training samples have context key
        extpos = numpos*2 if hasmc else numpos
        extneg = numneg*2 if hasmc else numneg
        stats[currlist[0]['src']] = {'raw': numsamples, 'pos':numpos, 'neg': numneg, 'mc': hasmc, 'ctxt': hascontext, 'extpos': extpos, 'extneg': extneg}
        print(stats[currlist[0]['src']])
        print("######################################")
        stats['TOTAL']['raw'] += numsamples
        stats['TOTAL']['pos'] += numpos
        stats['TOTAL']['neg'] += numneg
        stats['TOTAL']['extpos'] += extpos
        stats['TOTAL']['extneg'] += extneg
        outlist.extend(currlist)
    print("########### TOTALS ###########################")
    print(stats['TOTAL'])
    for s in outlist:
        random.shuffle(s['pos_paras'])
        random.shuffle(s['neg_paras'])
    utils.saveas_jsonl(outlist, outfile)
    json.dump(stats, open(outstatsfile, 'w'))
    print(f"Counts saved to {outstatsfile}")
    return outlist

random.seed(42)
dev_out = load_print_stats_combine(RR_IN_DEV, 'rr_dev')
train_out = load_print_stats_combine(RR_IN_TRAIN, 'rr_train')




