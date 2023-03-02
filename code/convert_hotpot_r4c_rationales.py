#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:13:33 2023

@author: tim hartill

Create gold rationales from the r4c (Inoue et al 2020) derived subset of HPQA.

Output format:
    [ {'question': 'full question text incl MC options and preceding initial ctxt',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'r4c',
       pos_paras: [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]]}, ...],
       neg_paras: [], #filled in later
      },
     
    ]

Note:
    random.shuffle(sentence_spans)
    text_shuffled = ' '.join([tststr[s:e].strip() for s,e in sentence_spans])

"""

import os
import json

import eval_metrics
import utils
from text_processing import format_sentence, create_sentence_spans, split_into_sentences


HPQA_DEV = '/home/thar011/data/hpqa/hotpot_dev_fullwiki_v1.json'
HPQA_TRAIN = '/home/thar011/data/hpqa/hotpot_train_v1.1.json'

BASE_DIR = '/home/thar011/data/hpqa_r4c'
R4C_DEV = os.path.join(BASE_DIR, 'dev_csf.json')
R4C_TRAIN = os.path.join(BASE_DIR, 'train.json')

R4C_DEV_OUT = os.path.join(BASE_DIR, 'r4c_pos_paras_dev_csf.jsonl')
R4C_TRAIN_OUT = os.path.join(BASE_DIR, 'r4c_pos_paras_train.jsonl')

hpqa_dev = json.load(open(HPQA_DEV))        # 7405 dict_keys(['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']) 
hpqa_train = json.load(open(HPQA_TRAIN))    # 90447 #'supporting_facts' = [ [title, sentence_idx], ..]

hpqa_dev = {h['_id']:h for h in hpqa_dev}
hpqa_train = {h['_id']:h for h in hpqa_train}

r4c_dev = json.load(open(R4C_DEV))  #2290
r4c_train = json.load(open(R4C_TRAIN)) #2379


def process(split, lookup):
    """ look up question and answer, merge with pos rationales
    """
    outlist = []
    for i, k in enumerate(split):
        out = {}
        orig = lookup[k]
        out['question'] = orig['question']
        out['answers'] = [orig['answer']]
        out['_id'] = str(k)
        out['src'] = 'r4c'
        pos_paras = []
        for ratlist in split[k]: # 1 per annotator = 3 total
            rationale = ''
            for slist in ratlist: # each sent ['It Takes a Family', 1, ['It Takes a Family book', 'is a response to', 'the 1996 book']]
                sent = format_sentence( ' '.join(slist[2]) ) + ' '
                rationale += sent
            rationale = rationale.strip()
            sentence_spans = create_sentence_spans(split_into_sentences(rationale))
            #print(f"RATIONALE: #{rationale}# {sentence_spans} ##{['!'+rationale[s:e]+'!' for s,e in sentence_spans]}##")
            pos_paras.append( {'text': rationale, 'sentence_spans': sentence_spans} )
        out['pos_paras'] = pos_paras
        out['neg_paras'] = []
        outlist.append(out)
        if i % 250 == 0:
            print(f"Processed: {i}")
            
    return outlist


def output_expl_ans(split, outfile):
    """ Output 1 tsv train/dev per annotator as r4c_expl_ans_n 
    """ 
    for i in range(3):
        outlist = []
        for s in split:
            outlist.append( utils.create_uqa_example(s['question'], s['pos_paras'][i]['text'], s['answers'][0]) )
        outdir = os.path.join(eval_metrics.UQA_DIR, 'hpqa_r4c_expl_ans_' + str(i))
        utils.save_uqa(outlist, outdir, outfile)
    return
    

dev_out = process(r4c_dev, hpqa_dev)
train_out = process(r4c_train, hpqa_train)

utils.saveas_jsonl(dev_out, R4C_DEV_OUT)
utils.saveas_jsonl(train_out, R4C_TRAIN_OUT)

output_expl_ans(dev_out, 'dev.tsv')
output_expl_ans(train_out, 'train.tsv')

