#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:04:18 2022

Convert Twentyquestions dataset to uqa format

https://github.com/allenai/twentyquestions


@author: tim hartill


"""

import os

import eval_metrics
import utils

UQA_DIR = eval_metrics.UQA_DIR
OUT_DS_DIR = os.path.join(UQA_DIR, 'twentyquestions')

devfile = '/home/thar011/data/twentyquestions/twentyquestions/twentyquestions-dev.jsonl'
trainfile = '/home/thar011/data/twentyquestions/twentyquestions/twentyquestions-train.jsonl'
testfile = '/home/thar011/data/twentyquestions/twentyquestions/twentyquestions-test.jsonl'


dev = utils.load_jsonl(devfile)  # 15403     dict_keys(['subject', 'question', 'answer', 'quality_labels', 'score', 'high_quality', 'labels', 'is_bad', 'true_votes', 'majority', 'subject_split_index', 'question_split_index'])
train = utils.load_jsonl(trainfile)  # 46566
test = utils.load_jsonl(testfile)  # 16921

#tst = len([t for t in train if t['score'] < 2])  # 0

def make_samples(split, out_dir, out_file):
    out_list = []
    for s in split:
        s['subj_question'] = s['subject'].strip() + ' - ' + s['question'].strip()
        s['answer_yn'] = 'yes' if s['majority'] else 'no'
        out_list.append( utils.create_uqa_example(s['subj_question'], ' ', s['answer_yn'], append_q_char='?') )
    utils.save_uqa(out_list, out_dir, out_file)
    return 

make_samples(dev, OUT_DS_DIR, 'dev.tsv')
make_samples(train, OUT_DS_DIR, 'train.tsv')
make_samples(test, OUT_DS_DIR, 'test.tsv')



