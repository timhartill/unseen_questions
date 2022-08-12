#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:29:23 2022

Build <No Answer> dataset from combined HPQA/Hover/FEVER examples.

Uses stategy built for  encode_context_stage2(..) in reader/reader_dataset.py

Input samples dataset created using combine_sentence_annot_datasets.py which in turn aggregates samples created
in convert_hotpot_sentence_annotations.py, convert_hover.py and convert_fever_sentence_annotations.py respectively.

@author: tim hartill

"""
import os
import random
from tqdm import tqdm

import eval_metrics
import utils
from text_processing import get_sentence_list, split_into_sentences, create_sentence_spans


SENT_DIR = '/home/thar011/data/sentences/'
SENT_TRAIN = os.path.join(SENT_DIR, 'sent_train.jsonl')
SENT_DEV = os.path.join(SENT_DIR, 'sent_dev.jsonl')

sent_dev = utils.load_jsonl(SENT_DEV)     #26587
sent_train = utils.load_jsonl(SENT_TRAIN) #239276

def standardize(split):
    for sample in tqdm(split):
        utils.consistent_bridge_format(sample)
        sample['para_idxs'] = utils.get_para_idxs(sample["pos_paras"])
        if sample['answers'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
            sample['answers'][0] = 'yes'
        elif sample['answers'][0] in ["REFUTES", "NOT_SUPPORTED"]:
            sample['answers'][0] = 'no'
        elif sample['answers'][0] == 'NOT ENOUGH INFO':  #Unused
            sample['answers'][0] = '<No Answer>'
    return

standardize(sent_dev)
standardize(sent_train)


def create_context(sample, make_noans, train):
    """
    encode context for stage 2: add non-extractive answer choices, add sentence start markers [unused1] for sentence identification
    encode as: " [SEP] yes no [unused0] [SEP] [unused1] title2 | sent0 [unused1] title2 | sent2 [unused1] title0 | sent2 ..."
    if make_noans == -1 then substitute some/all pos sents with neg sents -> label = <No Answer>
    else pos sample -> all pos sents present -> label = fully evidential
    In both pos/neg samples, additional random neg sents are added
    """
    para_titles = utils.flatten(sample['bridge'])
    all_pos_sents = []
    all_neg_sents = []
    for t in para_titles:
        para = sample['pos_paras'][ sample['para_idxs'][t][0] ]
        pos_sents = utils.encode_title_sents(para['text'], t.strip(), para['sentence_spans'], para['sentence_labels'])
        all_pos_sents.extend( pos_sents )
        neg_sent_idxs = []
        for i in range(len(para['sentence_spans'])): # Add neg sents from pos paras
            if i not in para['sentence_labels']:
                neg_sent_idxs.append(i)
        neg_sents = utils.encode_title_sents(para['text'], t, para['sentence_spans'], neg_sent_idxs)
        all_neg_sents.extend( neg_sents )
        
    num_pos_initial = len(all_pos_sents) # annotation errors mean occasionally there are 0 positive sents
    all_pos_labels = [1] * num_pos_initial
    
    first_time = True
    while first_time or len(all_neg_sents) < num_pos_initial:
        first_time = False
        for i in range(2):  # add neg sents from neg paras
            para = random.choice(sample["neg_paras"])
            t = para["title"].strip()
            sent_spans = create_sentence_spans( split_into_sentences(para["text"]) )
            neg_sent_idxs = list(range(len(sent_spans)))
            neg_sents = utils.encode_title_sents(para['text'], t, sent_spans, neg_sent_idxs)
            all_neg_sents.extend( neg_sents )
    if train:
        random.shuffle(all_neg_sents)
    
    if make_noans == -1: # neg sample - replace some pos sents with neg sents, label will be insuff evidence ie <No Answer>
        curr_pos_idxs = list(range(num_pos_initial))
        if train:
            divisor = random.choice([2,3])
            random.shuffle(curr_pos_idxs)
        else:
            divisor = 2
        if sample['src'] != 'fever':  # fever sent evidentiality is "or" not "and" so set all neg sample sents to neg_sents since partially replacing pos sents doesnt work wrt label
            firstnegidx = num_pos_initial // divisor
        else:
            firstnegidx = 0  
        neg_idx = -1
        for i in range(firstnegidx, num_pos_initial):
            neg_idx += 1
            all_pos_sents[i] = all_neg_sents[neg_idx]
            all_pos_labels[i] = 0
        all_neg_sents = all_neg_sents[neg_idx+1:]
    
    max_sents = random.choice([7,8,9]) if train else 9
    num_to_add = max_sents - num_pos_initial
    if num_to_add > 0:
        all_pos_sents.extend( all_neg_sents[:num_to_add] )  # additional negs
        num_to_add = len(all_pos_sents) - len(all_pos_labels)
        all_pos_labels += [0] * num_to_add
        
    #shuffle sents preserving label mapping
    all_pair = list(zip(all_pos_sents, all_pos_labels))
    if train:
        random.shuffle(all_pair)
    context = ' '.join([s[0] for s in all_pair])
    s_labels = [s[1] for s in all_pair]
    pos_sent_idxs = [i for i,s in enumerate(s_labels) if s == 1]

    #context = " [SEP] yes no [unused0] [SEP] " + context  # ELECTRA tokenises yes, no to single tokens
    #(doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, 
    # all_doc_tokens, sent_starts) = utils.context_toks_to_ids(context, tokenizer, sent_marker='[unused1]', special_toks=special_toks)
    
    #sample["context_processed"] = {
    #        "doc_tokens": doc_tokens,                     # [whole words]
    #        "char_to_word_offset": char_to_word_offset,   # [char idx -> whole word idx]
    #        "orig_to_tok_index": orig_to_tok_index,       # [whole word idx -> subword idx]
    #        "tok_to_orig_index": tok_to_orig_index,       # [ subword token idx -> whole word token idx]
    #        "all_doc_tokens": all_doc_tokens,             # [ sub word tokens ]
    #        "context": context,                           # full context string ie everything after question
    #        "sent_starts": sent_starts,                   # [sentence start idx -> subword token idx]
    #        "sent_labels": s_labels,                      # [multihot sentence labels]
    #        "passage": {'pos_sent_idxs': pos_sent_idxs},  # dict for stage1 format compatability: was the pos or neg para {'title':.. 'text':..., pos/neg specific keys}
    #}
    return all_pair

create_context(sent_dev[0], make_noans=100, train=False)


