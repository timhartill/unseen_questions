#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:14:05 2022

@author: tim hartill

Read sentence-level annotations from original HPQA train/dev files and add to mdr-formatted HPQA train/dev files


"""

import os
import json
from html import unescape

import utils


HPQA_DEV = '/home/thar011/data/hpqa/hotpot_dev_fullwiki_v1.json'
HPQA_TRAIN = '/home/thar011/data/hpqa/hotpot_train_v1.1.json'

MDR_DEV = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0.json'
MDR_TRAIN = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0.json'
MDR_PROCESSED_CORPUS = '/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json'

MDR_UPDATED_DEV = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl'
MDR_UPDATED_TRAIN = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl'


hpqa_dev = json.load(open(HPQA_DEV))        # 7405 dict_keys(['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']) 
hpqa_train = json.load(open(HPQA_TRAIN))  # 90447 #'supporting_facts' = [ [title, sentence_idx], ..]

mdr_dev = utils.load_jsonl(MDR_DEV)        # 7405  dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id'])
mdr_train = utils.load_jsonl(MDR_TRAIN)     # 90447
mdr_corpus = json.load(open(MDR_PROCESSED_CORPUS)) # 5233329  MDR processed file has the sentence mappings, easier than doing raw processing on hpqa wiki dump..


def create_sentence_spans(sent_list):
    """ Convert list of sentences into [ [s0start, s0end], [s1start, s1end], ...] ( where ''.join(sent_list) should = the complete para text..)
    """
    sentence_spans = []
    curr_para_len = 0
    for s in sent_list:
        slen = len(s)
        if slen > 0:
            sentence_spans.append( [curr_para_len, curr_para_len + slen] )
            curr_para_len += slen
    return sentence_spans
    
    

def build_title_dict(mdr_corpus):
    """ Convert MDR processed corpus file from:
        {'idx':{'title': 'title', 'text':'abstract text', 'sents': ['Sentence 1.', ' Sentence 2.', ...]} }
    """
    corpus_sents = {}
    dup_titles = []
    for idx in mdr_corpus:
        spans = create_sentence_spans(mdr_corpus[idx]['sents'])
        title = mdr_corpus[idx]['title']
        if corpus_sents.get(unescape(title)) is not None:
            print(f"Duplicate title: {title} (unescaped: {unescape(title)})")
            dup_titles.append(title)
        corpus_sents[unescape(title)] = {'sentence_spans':spans, 'text': mdr_corpus[idx]['text']}
    print(f"Duplicate titles: {len(dup_titles)}")  #0
    return corpus_sents, dup_titles

hpqa_sentence_spans, dup_titles = build_title_dict(mdr_corpus)

# [] and true for mdr_train para 1 and mdr_dev both paras also...so training para text lines up with corpus text..
#[m for m in mdr_train if m['pos_paras'][0]['text'] != hpqa_sentence_spans[unescape(m['pos_paras'][0]['title'])]['text'] ]


def aggregate_sent_annots(supporting_facts):
    """ Aggregate supporting fars from eg [['Allie Goertz', 0], ['Allie Goertz', 1], ['Allie Goertz', 2], ['Milhouse Van Houten', 0]]
    to {'Allie Goertz': [0,1,2], 'Milhouse Van Houten': [0]}
    """
    label_dict = {}
    for t, s in supporting_facts: 
        title_unescaped = unescape(t)
        if label_dict.get(title_unescaped) is None:
            label_dict[title_unescaped] = []
        label_dict[title_unescaped].append(s)
    for t in label_dict:
        label_dict[t].sort()    
    return label_dict
        


def add_span_and_sent_annot(mdr_split, hpqa_split, sentence_spans):
    """ Add keys for sentence_spans and sentence annotations
    Note: mdr and hpqa files are in the same order...
    """
    for i, s in enumerate(mdr_split):
        sent_labels = aggregate_sent_annots(hpqa_split[i]['supporting_facts'])
        for para in s['pos_paras']:
            title_unescaped = unescape(para['title'])  # a few titles are actually escaped..
            spans = sentence_spans[title_unescaped]['sentence_spans']
            para['sentence_spans'] = spans
            para['sentence_labels'] = sent_labels[title_unescaped]
        if i % 10000 == 0:
            print(f"Processed: {i}")
    return


def check_sentence_annots(split):
    """ check sentence annotations are all valid
    """
    errlist = []
    for i, s in enumerate(split):
        for j, para in enumerate(s['pos_paras']):
            if len(para['sentence_labels']) == 0:
                print(f"sample:{i} pospara: {j} Zero length sentence label!")
                errlist.append([i,j,"zero len label"])
            for sent_idx in para['sentence_labels']:
                if sent_idx >= len(para['sentence_spans']):
                    print(f"sample:{i} pospara: {j} sent idx > # sentences!")
                    errlist.append([i,j,"idx > # sents"])
                else:
                    start, end = para['sentence_spans'][sent_idx]
                    if start < 0 or len(para['text']) == 0:
                        print(f"sample:{i} pospara: {j} invalid start")
                        errlist.append([i,j,"invalid start or zero len text"])
                    if end > len(para['text']):    
                        print(f"sample:{i} pospara: {j} invalid end")
                        errlist.append([i,j,"invalid end"])
    return errlist
    

add_span_and_sent_annot(mdr_dev, hpqa_dev, hpqa_sentence_spans)
add_span_and_sent_annot(mdr_train, hpqa_train, hpqa_sentence_spans)

utils.saveas_jsonl(mdr_dev, MDR_UPDATED_DEV)
utils.saveas_jsonl(mdr_train, MDR_UPDATED_TRAIN)

errlist = check_sentence_annots(mdr_dev) # sample:5059 pospara: 0 sent idx > # sentences! Error is in hpqa sentence annot
errlist = check_sentence_annots(mdr_train)  #Errors all seem to be in the hpqa sentence annots
#sample:514 pospara: 1 sent idx > # sentences!
#sample:8332 pospara: 0 sent idx > # sentences!
#sample:9548 pospara: 1 sent idx > # sentences!
#sample:13415 pospara: 0 sent idx > # sentences!
#sample:20594 pospara: 0 sent idx > # sentences!
#sample:22896 pospara: 0 sent idx > # sentences!
#sample:27436 pospara: 0 sent idx > # sentences!
#sample:37004 pospara: 1 sent idx > # sentences!
#sample:38579 pospara: 0 sent idx > # sentences!
#sample:41267 pospara: 0 sent idx > # sentences!
#sample:45705 pospara: 1 sent idx > # sentences!
#sample:49355 pospara: 0 sent idx > # sentences!
#sample:50651 pospara: 1 sent idx > # sentences!
#sample:52080 pospara: 0 sent idx > # sentences!
#sample:60885 pospara: 1 sent idx > # sentences!
#sample:67475 pospara: 0 sent idx > # sentences!
#sample:77109 pospara: 0 sent idx > # sentences!
#sample:85934 pospara: 0 sent idx > # sentences!
#sample:86118 pospara: 1 sent idx > # sentences!
#sample:86193 pospara: 1 sent idx > # sentences!
#sample:88641 pospara: 0 sent idx > # sentences!
#sample:89961 pospara: 0 sent idx > # sentences!




