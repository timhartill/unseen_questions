#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:40:49 2021

@author: Tim Hartill

Misc utility fns - general on top and Huggingface related below

"""
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining
from bart import MyBart
from data import normalize_num, split_digits_special


#####################################
# General utils
#####################################

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


def create_uqa_example(question, context=None, answer=None, append_nl=True):
    """ Returns an example in uqa format
    """
    sample = question.strip()
    if sample[-1] == '.':
        sample = sample[:-1]
    if sample[-1] != '?':
        sample += '?'
    sample += ' \\n'
    if context is not None and context != '':
        sample += ' ' + context.strip()
    if answer is not None and answer != '':
        sample += '\t' + answer.strip()
        if append_nl:
            sample += '\n'
    return sample


#######################
# HF Utils
#######################

def load_model(model_name, checkpoint=None):
    """ Load a model and set for prediction
    Usage: tokenizer, model = load_model(model_name, checkpoint)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "facebook/bart-large":   
        my_model = MyBart
    else:
        my_model = AutoModelForPreTraining  # HF documentation indicates this gives right models for T5 and gpt2 as well as vanilla bart
    
    print(f"Loading model: {model_name}")
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")       
        model = my_model.from_pretrained(model_name, state_dict=torch.load(checkpoint))  
    else:
        print("No checkpoint loaded. Training from base pretrained model.")
        model = my_model.from_pretrained(model_name) 
    
    model.to(torch.device("cuda"))
    model.eval()
    print("Model loaded!")
    return tokenizer, model


def run_model(input_string, model, tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True,
              indiv_digits=False, norm_numbers=True, norm='', special='Ġ', verbose=False,
              truncation=True, max_input_length=512, lower=True, **generator_args):
    """ Run cut-down version of tokenisation and generation pipeline for single input string
        See https://huggingface.co/blog/how-to-generate for generation options
    Returns: Full result depending on which **generator params are set (e.g.including any scores, attns etc)
             if one of the params in **generator is return_dict_in_generate=True
             otherwise returns the decoded prediction.
    Usage:
    res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers,
                    num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                    output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
    pred, score = get_single_result(res)        
    """
    if verbose:
        print(f"Approx word count: {len(input_string.split())}")
    if lower:
        input_string = input_string.lower()
    if norm_numbers:
        input_string = normalize_num(input_string, norm=norm)    
    toks = tokenizer.tokenize(input_string)
    if indiv_digits:
        toks = split_digits_special(toks, special=special)
    ids = tokenizer.convert_tokens_to_ids(toks)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    numtoks = len(ids)
    if verbose:
        print(f"Number of tokens: {numtoks}")
    if truncation and numtoks > max_input_length-1:
        ids = ids[:max_input_length-1]
        if verbose:
            print(f"Truncated to {max_input_length} tokens.")
    ids = ids + [tokenizer.eos_token_id]    
    ids = [ids]
    #input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = torch.LongTensor(ids)
    input_ids = input_ids.to(torch.device("cuda"))
    res = model.generate(input_ids, **generator_args)
    if isinstance(res, dict):
        res['preds'] = tokenizer.batch_decode(res['sequences'], skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        return res
    return tokenizer.batch_decode(res, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


def get_single_result(res):
    """ Process single result from a res object
    Usage: pred, score = get_single_result(res)
    """
    return res.preds[0].strip(), float(res.sequences_scores.detach().cpu().numpy()[0])







