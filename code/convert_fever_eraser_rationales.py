#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:03:21 2023

@author: tim hartill

Convert fever rationales from eraser version of fever...

Input format:
    
    {"annotation_id": "10", "classification": "REFUTES", "docids": ["Ireland"], 
     "evidences": [[{"docid": "Ireland", "end_sentence": 7, "end_token": 177, "start_sentence": 6, "start_token": 157, 
                     "text": "The island 's geography comprises relatively low-lying mountains surrounding a central plain , with several navigable rivers extending inland ."}]], 
     "query": "Ireland does not have relatively low-lying mountains.", "query_type": null}


Output 'rr' format:
    [ {'question': 'full question text incl MC options and preceding initial ctxt',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'fever',
       pos_paras: [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]]}, ...],
       neg_paras: [], #filled in later
      },
     
    ]

"""

import os
import random
import copy
import eval_metrics
import utils
from text_processing import normalize_unicode, convert_brc, replace_chars, format_sentence, create_sentence_spans, strip_accents, split_into_sentences

UQA_DIR = eval_metrics.UQA_DIR
BASE_DIR = '/home/thar011/data/eraser/fever/'

rr_dev = os.path.join(BASE_DIR, 'fever_dev_rationales.jsonl')
rr_train = os.path.join(BASE_DIR, 'fever_train_rationales.jsonl')

rr_dev_exclposonly = BASE_DIR + 'fever_dev_rr_all_pos_neg_exclposonly.jsonl'
rr_train_exclposonly = BASE_DIR + 'fever_train_rr_all_pos_neg_exclposonly.jsonl'

rr_dev_newneg = BASE_DIR + 'fever_dev_rr_all_pos_neg_extrasinglenegs.jsonl'
rr_train_newneg = BASE_DIR + 'fever_train_rr_all_pos_neg_extrasinglenegs.jsonl'


file_rr_dev_negs = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T48_FEVER_DEV_onv6_sample-03-07-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json', 
               ]
file_rr_train_negs = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T49_FEVER_TRAIN_onv6_sample_1st20K-03-10-2023-LLM-bigscience-bloom-maxsmpls20000-randFalse/llm_samples_with_context.json',
                 ]



dev_in = utils.load_jsonl( os.path.join(BASE_DIR, 'dev.jsonl') )
train_in = utils.load_jsonl( os.path.join(BASE_DIR, 'train.jsonl') )


def normalise_text(text):
    """ normalize_unicode(convert_brc(train_in[1]['evidences'][1][0]['text']))
    s = {"annotation_id": "101042__0", "classification": "SUPPORTS", "docids": ["Augustus"], "evidences": [[{"docid": "Augustus", "end_sentence": 28, "end_token": 698, "start_sentence": 27, "start_token": 687, "text": "Augustus died in AD 14 at the age of 75 ."}, {"docid": "Augustus", "end_sentence": 29, "end_token": 717, "start_sentence": 28, "start_token": 698, "text": "He probably died from natural causes , although there were unconfirmed rumors that his wife Livia poisoned him ."}], [{"docid": "Augustus", "end_sentence": 7, "end_token": 128, "start_sentence": 6, "start_token": 90, "text": "23 September 63 BC -- 19 August 14 AD -RRB- was the founder of the Roman Principate and considered the first Roman emperor , controlling the Roman Empire from 27 BC until his death in AD 14 ."}, {"docid": "Augustus", "end_sentence": 29, "end_token": 717, "start_sentence": 28, "start_token": 698, "text": "He probably died from natural causes , although there were unconfirmed rumors that his wife Livia poisoned him ."}]], "query": "Augustus died in 14 AD from natural causes.", "query_type": None}   
    evidences = [[{"docid": "Augustus", "end_sentence": 28, "end_token": 698, "start_sentence": 27, "start_token": 687, "text": "Augustus died in AD 14 at the age of 75 ."}, 
                  {"docid": "Augustus", "end_sentence": 29, "end_token": 717, "start_sentence": 28, "start_token": 698, "text": "He probably died from natural causes , although there were unconfirmed rumors that his wife Livia poisoned him ."}], 
                 [{"docid": "Augustus", "end_sentence": 7, "end_token": 128, "start_sentence": 6, "start_token": 90, "text": "23 September 63 BC -- 19 August 14 AD -RRB- was the founder of the Roman Principate and considered the first Roman emperor , controlling the Roman Empire from 27 BC until his death in AD 14 ."}, 
                  {"docid": "Augustus", "end_sentence": 29, "end_token": 717, "start_sentence": 28, "start_token": 698, "text": "He probably died from natural causes , although there were unconfirmed rumors that his wife Livia poisoned him ."}]]
    """
    text = format_sentence(normalize_unicode(convert_brc(text)))
    return text


def process(split):
    out_list = []
    for i, s in enumerate(split):
        out = {}
        q = s['query'].strip()
        if q[-1] in ['.', '!', '?', ':', ';']:
            q = q[:-1]
        out['question'] = q + '?'
        if s['classification'] == 'SUPPORTS':
            out['answers'] = ['yes']
        else:
            out['answers'] = ['no']
        out['_id'] = str(s['annotation_id'])
        out['src'] = 'fever'
        
        evidence_list = utils.flatten(s['evidences'])
        evidence_list.sort(key=lambda x: x['start_sentence'])
        evidence_list = [normalise_text(e['text']).replace("( ", "(").replace(" )", ")") for e in evidence_list]
        evidence_list = utils.unique_preserve_order(evidence_list)
        
        sentence_spans = []
        new_start, new_end = 0, 0
        text = ''
        for sent in evidence_list:
            if sent != '':
                if len(sentence_spans) > 0:
                    sent = ' ' + sent
                new_end = new_start + len(sent)
                sentence_spans.append( [new_start, new_end] )
                new_start = new_end
                text += sent                               
        out['pos_paras'] = [ {'text': text, 'sentence_spans': sentence_spans} ]
        out['neg_paras'] = []
        if len(sentence_spans) > 0:
            out_list.append(out)
    return out_list
    
dev_out = process(dev_in)
train_out = process(train_in)

utils.saveas_jsonl(dev_out, rr_dev)
utils.saveas_jsonl(train_out, rr_train)

dev_expl_ans = [utils.create_uqa_example(s['question'], s['pos_paras'][0]['text'], s['answers'][0]) for s in dev_out]
utils.save_uqa(dev_expl_ans, os.path.join(eval_metrics.UQA_DIR, 'fever_expl_ans'), 'dev.tsv')
train_expl_ans = [utils.create_uqa_example(s['question'], s['pos_paras'][0]['text'], s['answers'][0]) for s in train_out]
utils.save_uqa(train_expl_ans, os.path.join(eval_metrics.UQA_DIR, 'fever_expl_ans'), 'train.tsv')

#############################
# Create RR model training data below
#############################

dev_rr_format = utils.load_jsonl(rr_dev)      #6122 -> 6000
train_rr_format = utils.load_jsonl(rr_train)  #97957 -> 91274 after de-dup questions in load_merge_negs(..)

# merge routine to align pos and negs
dev_rr_format = utils.load_merge_negs(dev_rr_format, file_rr_dev_negs)  # dev_rr_format = all q+pos with negs added where there is a match ie the superset
utils.saveas_jsonl(dev_rr_format, rr_dev)

train_rr_format = utils.load_merge_negs(train_rr_format, file_rr_train_negs)
utils.saveas_jsonl(train_rr_format, rr_train)

utils.output_neg_tsv(dev_rr_format, os.path.join(UQA_DIR, 'fever_neg_expl_ans'), 'dev.tsv')
utils.output_neg_tsv(train_rr_format, os.path.join(UQA_DIR, 'fever_neg_expl_ans'), 'train.tsv')

#dev_rr_format = utils.load_jsonl(rr_dev)
#train_rr_format = utils.load_jsonl(rr_train)

# save  rr model fever training dataset - only output where negs exist 
utils.output_rr_where_negs_exist(dev_rr_format, outfile=rr_dev_exclposonly)   
utils.output_rr_where_negs_exist(train_rr_format, outfile=rr_train_exclposonly) 

dev_rr_format_exclposonly = utils.load_jsonl(rr_dev_exclposonly) # 2472
train_rr_format_exclposonly = utils.load_jsonl(rr_train_exclposonly)  #8896

# create single random easy neg for samples that don't have negs. These are added to rr training in combine_rr_datasets.py
dev_rr_format = utils.load_jsonl(rr_dev)      #6000 NOTE: loading here means llm negs from  utils.load_merge_negs above have been added...
train_rr_format = utils.load_jsonl(rr_train)  #91274 after de-dup questions in load_merge_negs(..)


def add_single_easy_neg(samples, force_overwrite=False):
    """ Add a single random neg to samples without negs and return just those samples
    random neg is just a pos from a different sample..
    """
    orig_idxs = [i for i,s in enumerate(samples) if len(s['neg_paras']) == 0 or force_overwrite ]
    outlist = []
    for i, s in enumerate(samples):
        if len(s['neg_paras']) > 0 and not force_overwrite:
            continue
        idx = random.choice(orig_idxs)
        while idx == i:
            idx = random.choice(orig_idxs)
        out = copy.deepcopy(s)
        out['neg_paras'] = [ samples[idx]['pos_paras'][0] ]
        outlist.append(out)
    print(f"Returning {len(outlist)} of {len(samples)}")
    return outlist

random.seed(42)
dev_rr_format_newneg = add_single_easy_neg(dev_rr_format, force_overwrite=True)  #6000
utils.saveas_jsonl(dev_rr_format_newneg, rr_dev_newneg)

train_rr_format_newneg = add_single_easy_neg(train_rr_format, force_overwrite=False)  #71716
utils.saveas_jsonl(train_rr_format_newneg, rr_train_newneg)


