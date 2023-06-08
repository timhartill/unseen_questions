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
import copy

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

ITER_DEV = '/home/thar011/data/sentences/sent_dev_feversingleonlyv3.jsonl'      # s1, s2 model dev : hover, fever, hpqa
ITER_TRAIN = '/home/thar011/data/sentences/sent_train_feversingleonlyv3.jsonl'  # s1, s2 model train : hover, fever, hpqa


RR_DEV_FEVER_NEWNEG = '/home/thar011/data/eraser/fever/fever_dev_rr_all_pos_neg_extrasinglenegs.jsonl'
RR_TRAIN_FEVER_NEWNEG = '/home/thar011/data/eraser/fever/fever_train_rr_all_pos_neg_extrasinglenegs.jsonl'



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
dev_out = load_print_stats_combine(RR_IN_DEV, 'rr_dev')         # load each file, combine and save
train_out = load_print_stats_combine(RR_IN_TRAIN, 'rr_train')

####
# Create rr training samples for iter-like contexts fromev set scorer train samples to supplement the short rationale-form samples
####

tokenizer = utils.load_model(loadwhat='tokenizer_only')
iter_dev = utils.load_jsonl(ITER_DEV)  # s1, s2 sent annotated dev split: 23495
iter_train = utils.load_jsonl(ITER_TRAIN)  # s1, s2 sent annotated train split: 205320

#test = [copy.deepcopy(iter_dev[0]), copy.deepcopy(iter_dev[5000]), copy.deepcopy(iter_dev[-1])]
#test_out = utils.make_rr_from_mdr_format(test, tokenizer, max_toks=507, include_title_prob=1.0, include_all_sent_prob=0.1)

rr_dev_all = utils.make_rr_from_mdr_format(iter_dev, tokenizer, max_toks=507, include_title_prob=1.0, include_all_sent_prob=0.1)
utils.saveas_jsonl(rr_dev_all, os.path.join(RR_OUT_BASE, 'rr_dev_iterctxtsv3_all.jsonl'))
rr_train_all = utils.make_rr_from_mdr_format(iter_train, tokenizer, max_toks=507, include_title_prob=1.0, include_all_sent_prob=0.1)
utils.saveas_jsonl(rr_train_all, os.path.join(RR_OUT_BASE, 'rr_train_iterctxtsv3_all.jsonl'))

# load rr_dev, rr_train
rr_dev_combo = utils.load_jsonl(os.path.join(RR_OUT_BASE, 'rr_dev.jsonl'))  #11240
rr_train_combo = utils.load_jsonl(os.path.join(RR_OUT_BASE, 'rr_train.jsonl'))  #48554

# merge hpqa, hover iter pos/negs into rr combo that has rationales
rr_dev_combo_iter =  utils.merge_pos_into_rr(rr_dev_combo, rr_dev_all, include_negs=True, add_src_to_key=True, strip_from_src='_iter')
rr_train_combo_iter =  utils.merge_pos_into_rr(rr_train_combo, rr_train_all, include_negs=True, add_src_to_key=True, strip_from_src='_iter')

# add hover separately since doesnt exist in current rr rationale based combo:
# note: below throws ignorable error msgs since many samples in rr_dev_all don't appear in rr_dev_combo
rr_dev_combo_iter += [s for s in rr_dev_all if s['src']=='hover_iter']  # + 4000 = 15240
rr_train_combo_iter += [s for s in rr_train_all if s['src']=='hover_iter']  # + 18171 = 66725
# save combined file
utils.saveas_jsonl(rr_dev_combo_iter, os.path.join(RR_OUT_BASE, 'rr_dev_rat_iterctxtsv3_merged.jsonl'))
utils.saveas_jsonl(rr_train_combo_iter, os.path.join(RR_OUT_BASE, 'rr_train_rat_iterctxtsv3_merged.jsonl'))

###################################
# Create version with extra negs
####################################

rr_dev_combo_iter = utils.load_jsonl(os.path.join(RR_OUT_BASE, 'rr_dev_rat_iterctxtsv3_merged.jsonl'))
rr_train_combo_iter = utils.load_jsonl(os.path.join(RR_OUT_BASE, 'rr_train_rat_iterctxtsv3_merged.jsonl'))

# add extra fever rationale negs
fever_dev_newneg = utils.load_jsonl(RR_DEV_FEVER_NEWNEG)    #6000
fever_train_newneg = utils.load_jsonl(RR_TRAIN_FEVER_NEWNEG)  #71716

for s in fever_dev_newneg: s['src'] = 'fever_newnegrat'
for i,s in enumerate(fever_train_newneg):
    if i == 10000:  #just take 10K extra train samples from this source. Note: this means to this point we've taken the first ~30K fever train samples
        break
    s['src'] = 'fever_newnegrat'

rr_dev_combo_iter += [s for s in fever_dev_newneg if s['src']=='fever_newnegrat']  # 15240 + 6000 = 21240
rr_train_combo_iter += [s for s in fever_train_newneg if s['src']=='fever_newnegrat']  # 66725 + 10000 = 76725

# create extra hpqa, hover, fever rationale-like samples from s1,s2 train set
iter_dev = utils.load_jsonl(ITER_DEV)  # s1, s2 sent annotated dev split: 23495
iter_train = utils.load_jsonl(ITER_TRAIN)  # s1, s2 sent annotated train split: 205320

fever_train = [s for s in iter_train if s.get('src') is not None and s['src']=='fever']  # 96702 take 10k samples not already in train
fever_train = fever_train[30000:40000]
rr_train_fever_newratlike = utils.make_rr_rat_from_mdr_format(fever_train, include_title_prob=0.75)

random.seed(42)

hover_train = [s for s in iter_train if s.get('src') is not None and s['src']=='hover']  # 18171 take 10k samples will be dup since all hover in train
idxs = [i for i in range(len(hover_train))]
random.shuffle(idxs)
idxs = idxs[:10000]
hover_train = [hover_train[i] for i in idxs]
rr_train_hover_newratlike = utils.make_rr_rat_from_mdr_format(hover_train, include_title_prob=0.75)


hpqa_train = [s for s in iter_train if s.get('src') is None]  # 90447 take 10k samples will be dup since all hover in train
idxs = [i for i in range(len(hpqa_train))]
random.shuffle(idxs)
idxs = idxs[:10000]
hpqa_train = [hpqa_train[i] for i in idxs]
rr_train_hpqa_newratlike = utils.make_rr_rat_from_mdr_format(hpqa_train, include_title_prob=0.75)


rr_train_combo_iter += rr_train_fever_newratlike + rr_train_hover_newratlike + rr_train_hpqa_newratlike
for s in rr_train_combo_iter:  # 172 blank negs..
    if len(s['neg_paras']) == 0: s['neg_paras'] = [{'text': 'A dummy irrelevant sentence.', 'sentence_spans': [[0, 28]]}]
    
utils.saveas_jsonl(rr_train_combo_iter, os.path.join(RR_OUT_BASE, 'rr_train_rat_iterctxtsv3_merged_extrarelevancerats.jsonl'))


rr_dev_newratlike = utils.make_rr_rat_from_mdr_format(iter_dev, include_title_prob=0.75)
rr_dev_combo_iter += rr_dev_newratlike
for s in rr_dev_combo_iter:  # 236 blank negs..
    if len(s['neg_paras']) == 0: s['neg_paras'] = [{'text': 'A dummy irrelevant sentence.', 'sentence_spans': [[0, 28]]}]

utils.saveas_jsonl(rr_dev_combo_iter, os.path.join(RR_OUT_BASE, 'rr_dev_rat_iterctxtsv3_merged_extrarelevancerats.jsonl'))

