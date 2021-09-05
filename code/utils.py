#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:40:49 2021

@author: Tim Hartill

Misc utility fns

"""
import json
import os


def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    num_jsons = len(all_json_list)
    if verbose:
        print('JSON as text successfully loaded. Number of json messages in file is ', num_jsons)
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print('Text successfully converted to JSON.')
    return all_json_list


def convert_pararules(all_json_list):
    """ Convert ParaRules-formatted jsonl into uqa-like format
    """
    questions = []
    answers = []
    for sample in all_json_list:
        context = sample['context']
        for q in sample['questions']:
            question = q['text'].strip() + ' \\n ' + context.strip()
            questions.append( question )
            if q['label']:
                answer = 'yes'
            else: 
                answer = 'no'
            answers.append( answer )
    return questions, answers


def load_uqa_supervised(file, ans_lower=True, verbose=True):
    """ Load a unifiedqa formatted .tsv file and return question+context as list of str amd answers as list of str
    """
    if verbose:
        print(f"Reading {file}...")
    questions = []
    answers = []
    with open(file, "r") as f:
        for line in f:
            question, answer = line.split("\t")
            if ans_lower:
                answer = answer.lower()
            questions.append( question.strip() )
            answers.append ( answer.strip() )
    if verbose:
        print(f"Successfully loaded {len(questions)} rows.")
    return questions, answers


def load_uqa_selfsupervised(file, verbose=True):
    """ Load a unifiedqa formatted .tsv file and return question+context as list of str amd answers as list of str
    """
    if verbose:
        print(f"Reading {file}...")
    questions = []
    answers = []
    with open(file, "r") as f:
        for line in f:
            question = line.strip()
            answer = ""
            questions.append( question )
            answers.append ( answer )
    if verbose:
        print(f"Successfully loaded {len(questions)} rows.")
    return questions, answers




def flatten(alist):
    """ flatten a list of nested lists
    """
    t = []
    for i in alist:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t





