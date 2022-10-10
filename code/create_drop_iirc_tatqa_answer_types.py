#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:37:07 2022

Extract answer types from original DROP, TATQA and IIRC files 
and save to UQA_DIR/answer_types/anstypes_datasetname__train|dev|test.jsonl

@author: tim hartill


"""

import os
import json

import eval_metrics
import utils

OUT_DIR = os.path.join(eval_metrics.UQA_DIR, 'answer_types')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Outputting files to {OUT_DIR}")

dropdevfile = os.path.join(eval_metrics.HDATA, 'data/drop/drop_dataset/drop_dataset_dev.json')
droptrainfile = os.path.join(eval_metrics.HDATA, 'data/drop/drop_dataset/drop_dataset_train.json')

iircdevfile = os.path.join(eval_metrics.HDATA, 'data/iirc/iirc_train_dev/dev.json')
iirctrainfile = os.path.join(eval_metrics.HDATA, 'data/iirc/iirc_train_dev/train.json')
iirctestfile = os.path.join(eval_metrics.HDATA, 'data/iirc/iirc_train_dev/iirc_test.json')

tatdevfile = os.path.join(eval_metrics.HDATA, 'data/tat_qa/tatqa_dataset_dev.json')
tattrainfile = os.path.join(eval_metrics.HDATA, 'data/tat_qa/tatqa_dataset_train.json')


def parse_and_assign_type_drop(split, out_file):
    """ Parse Drop samples and output answer types
    Note: Adapted from https://github.com/allenai/unifiedqa/encode_datasets.py to ensure 
          resulting order matches corresponding uqa tsv-formatted file
    outputs jsonl list of form {'ans_type': ans_type, 'pkey': key, 'qid': qpair['query_id'], 
                                 'tsv_question': question, 'tsv_answer': ans_text}
    where pkey = passage key and qid = question id in source file 
    and tsv_question and tsv_answer match what is in the tsv-formatted version
    out_list list idx = idx in tsv-formatted version and bad_list has items that couldnt be matched
    """
    out_list = []
    bad_list = []
    for key in split.keys():
        for qpair in split[key]['qa_pairs']:
            ans_text = ''
            question = qpair['question'].replace("\t", " ").replace("\n", " ")
            answer = qpair['answer']
            # print(answer)
            ans_type = ''
            number = answer['number']
            spans = answer['spans']
            if len(spans) > 0:
                ans_text = ", ".join(spans)
                if len(spans) == 1:
                    ans_type = 'SPAN'
                else:
                    ans_type = 'SPANS'
            elif len(number) > 0:
                ans_text = number
                ans_type = 'NUM'
            else:
                day = answer['date']['day']
                month = answer['date']['month']
                year = answer['date']['year']
                if len(month) > 0:
                    ans_text += month
                if len(day) > 0:
                    ans_text += f" {day}"
                if len(year) > 0:
                    ans_text += f" {year}"
                if ans_text != '':
                    ans_type = 'DATE'
            
            if ans_text == "":  # bad questions: 0 in dev, 9 in train
                print(f" >>>> no answer for the question . . . key:{key} question:{qpair['question']}")
                bad_list.append({'ans_type': ans_type, 'pkey': key, 'qid': qpair['query_id'], 
                                 'tsv_question': question, 'tsv_answer': ans_text})
                continue
            ans_text = ans_text.replace("\t", " ").replace("\n", " ")
            out_list.append({'ans_type': ans_type, 'pkey': key, 'qid': qpair['query_id'], 
                             'tsv_question': question, 'tsv_answer': ans_text})
    print(f"Number of bad questions: {len(bad_list)}")
    out_file = os.path.join(OUT_DIR, out_file)
    print(f"Outputting DROP answer types to {out_file}..")
    utils.saveas_jsonl(out_list, out_file)
    return 


def parse_and_assign_type_iirc(split, out_file):
    """ Parse IIRC samples and output answer types:
    SPAN (includes multiple spans), VALUE (Number), BINARY (Yes/No), NONE (<No Answer>)
    """
    out_list = []
    for i, sample in enumerate(split):
        for j, question in enumerate(sample['questions']):
            answer_info = question["answer"]
            a_type = answer_info["type"]
            if a_type == "span":
                answer_spans = [a["text"].strip() for a in answer_info["answer_spans"]]
                if len(answer_spans) == 1:
                    answer_spans = answer_spans[0]  # single span can be a string
                else:  # multiple spans: answer = list of all span permutations. In training only the first is used (EM) but in validation max EM over all permuations is used
                    answer_spans = ', '.join(answer_spans)
            elif a_type == "value":
                answer_spans = answer_info["answer_value"].strip()
            elif a_type == "binary":
                answer_spans = answer_info["answer_value"].strip()
            elif a_type == "none":
                answer_spans = '<No Answer>'
            if sample.get('pid') is None:
                sample['pid'] = str(i)
            if question.get('qid') is None:
                question['qid'] = str(j)
            out_list.append({'ans_type': a_type.upper(), 'pkey': sample['pid'], 'qid': question['qid'], 
                             'tsv_question': question['question'].strip().replace('\n', '').replace('\t', ''), 
                             'tsv_answer': answer_spans.strip().replace('\n', '').replace('\t', '')})
    out_file = os.path.join(OUT_DIR, out_file)
    print(f"Outputting IIRC answer types to {out_file}..")
    utils.saveas_jsonl(out_list, out_file)
    return


def parse_and_assign_type_tat(split, out_file):
    """ Parse TAT-Qa samples and output answer types:
    SPAN, MULTI-SPAN, COUNT (Number), ARITHMETIC (NUMBER)
    Adds 'ans_from' key with value 'table-text' or 'text'
    """
    out_list = []
    for i, sample in enumerate(split):
        for qa in sample['questions']:
            q = qa['question'].strip().replace('\n', '').replace('\t', '')
            if type(qa['answer']) == list:
                if len(qa['answer']) == 1:
                    ans = str(qa['answer'][0]).strip()
                else:
                    ans = [str(a).strip() for a in qa['answer']]
                    ans = ', '.join(ans)
            else:
                ans = str(qa['answer']).strip()
            ans = ans.strip().replace('\n', '').replace('\t', '')
            a_type = qa['answer_type'].upper()
            out_list.append({'ans_type': a_type.upper(), 'pkey': str(i), 'qid': qa['uid'], 
                             'tsv_question': q, 
                             'tsv_answer': ans,
                             'ans_from': qa['answer_from']})
    out_file = os.path.join(OUT_DIR, out_file)
    print(f"Outputting TAT-QA answer types to {out_file}..")
    utils.saveas_jsonl(out_list, out_file)
    return




print("Loading DROP...")
dev = json.load(open(dropdevfile))   # dict with keys like 'nfl_1184' (1st) and 'history_3552' (last)
train = json.load(open(droptrainfile))

parse_and_assign_type_drop(dev, out_file='anstypes_drop_dev.jsonl')
parse_and_assign_type_drop(train, out_file='anstypes_drop_train.jsonl')


print('Loading IIRC...')
dev = json.load(open(iircdevfile))   
train = json.load(open(iirctrainfile))
test = json.load(open(iirctestfile))

parse_and_assign_type_iirc(dev, out_file='anstypes_iirc_dev.jsonl')
parse_and_assign_type_iirc(train, out_file='anstypes_iirc_train.jsonl')
parse_and_assign_type_iirc(test, out_file='anstypes_iirc_test.jsonl')


print('Loading TATQA...')
dev = json.load(open(tatdevfile))   
train = json.load(open(tattrainfile))

parse_and_assign_type_tat(dev, out_file='anstypes_tatqa_dev.jsonl')
parse_and_assign_type_tat(train, out_file='anstypes_tatqa_train.jsonl')


