#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:40:49 2021

@author: Tim Hartill

Misc utility fns - general on top and Huggingface related below

"""
import json
import pickle
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import GPTJForCausalLM, AutoModelForCausalLM
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


def loadas_json(file):
    """ Load python obj from json file
    """
    obj = json.load(open(file))
    return obj


def saveas_json(obj, file, mode="w", indent=5, add_nl=False):
    """ Save python object as json to file
    default mode w = overwrite file
            mode a = append to file
    indent = None: all json on one line
                0: pretty print with newlines between keys
                1+: pretty print with that indent level
    add_nl = True: Add a newline before outputting json. ie if mode=a typically indent=None and add_nl=True   
    Example For outputting .jsonl (note first line doesn't add a newline before):
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=False)
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=True)
          
    """
    with open(file, mode) as fp:
        if add_nl:
            fp.write('\n')
        json.dump(obj, fp, indent=indent)
    return True    


def saveas_jsonl(obj_list, file, initial_mode = 'w', verbose=True, update=5000):
    """ Save a list of json msgs as a .jsonl file of form:
        {json msg 1}
        {json msg 2}
        ...
        To overwrite exiting file use default initial_mode = 'w'. 
        To append to existing file set initial_mode = 'a'
    """
    if initial_mode == 'w':
        if verbose:
            print('Creating new file:', file)
        add_nl = False
    else:
        if verbose:
            print('Appending to file:', file)
        add_nl = True
    mode = initial_mode
    for i, json_obj in enumerate(obj_list):
            saveas_json(json_obj, file, mode=mode, indent=None, add_nl=add_nl)
            add_nl = True
            mode = 'a'
            if verbose:
                if i % update == 0:
                    print('Processed:', i)
    if verbose:
        print('Finished adding to:', file)        
    return True


def saveas_pickle(obj, file):
    """ Save python obj to file
    """
    with open(file,"wb") as fp:
        pickle.dump(obj, fp)
    return True


def loadas_pickle(file):
    """ Load python obj from pickle file
    """
    with open(file, 'rb') as fp:
        obj = pickle.load(fp)
    return obj     


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
    """ Load a unifiedqa formatted .tsv file and return question+context as list of str and answers as list of str
    """
    if verbose:
        print(f"Reading {file}...")
    questions = []
    answers = []
    ctr = 1
    with open(file, "r") as f:
        for line in f:
            try:
                question, answer = line.split("\t")
            except:
                print(f"ERROR loading line: {ctr} ##{line}##")
            if ans_lower:
                answer = answer.lower()
            questions.append( question.strip() )
            answers.append ( answer.strip() )
            ctr += 1
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
    if sample[-1] in ['.','!']:
        sample = sample[:-1]
    if sample[-1] != '?':
        sample += '?'
    sample += ' \\n'
    if context is not None and context != '':
        sample += ' ' + context.strip()
    if answer is not None and answer != '':
        sample += ' \t' + answer.strip()
        if append_nl:
            sample += '\n'
    return sample


def find_mc_answer(context, correct_option='(A)'):
    """ Parse a uqa formatted MC sample and return the answer matching an option key """
    if correct_option[0] != '(':
        correct_option = '(' + correct_option + ')'
    start_idx = context.find(correct_option)
    if start_idx == -1:
        return ''
    start_idx += 3
    end_idx = context[start_idx:].find('(')
    if end_idx == -1:
        ans = context[start_idx:].strip()
    else:
        ans = context[start_idx:start_idx + end_idx].strip()
    return ans   


def format_decomp_ans(decomp_q, decomp_a, decomp_id, prior_answers):
    """ Return formatted decomp question plus answer after substituting prior answers into vars of form #1, #2 etc
        where #1 refers to the answer of the first decomp in the list (i.e. decomp[0])
        in form: ' ## decomp_q? decomp_a'
    """
    this_decomp = decomp_q.strip()
    if this_decomp[-1] in ['.','!']:
        this_decomp = this_decomp[:-1].strip()
    if this_decomp[-1] != '?':
        this_decomp += '?'
    var_idx = this_decomp.find('#')
    while var_idx != -1 and decomp_id > 0: # first decomp never contains a variable but one example starts with #9..
        prior_step = this_decomp[var_idx+1]  # max 5 decomp steps so this works.
        if prior_step not in ['1','2','3','4','5','6','7','8','9']:
            prior_idx = 0    # at least one entry has #!
        else:
            prior_idx = int(prior_step)-1
        try:
            subst = prior_answers[prior_idx]
        except Exception as e:
            print(e)
            print(f"Q:{decomp_q} j:{decomp_id} prior_answers:{prior_answers} prior_idx:{prior_idx}")
        this_decomp = this_decomp.replace('#'+prior_step, subst)
        var_idx = this_decomp.find('#')
    decomp_answer = decomp_a.strip()  #' #' + str(decomp_id+1) + ':' + decomp_a.strip()
    decomp_ans_str = ' ## ' + this_decomp + ' ' + decomp_answer
    prior_answers.append(decomp_answer)
    return decomp_ans_str, prior_answers, this_decomp


def parse_decomp_str(decomp_str):
    """ Parse a gt or predicted decomp string of form "overall answer ## decomp1 q? decomp1 ans ## decomp2 q? decomp2 ans"
    """
    decomp_dict = {'ans':'', 'dalist':[], 'dlist':[], 'alist':[]}
    decomps = decomp_str.split('##')
    decomp_dict['ans'] = decomps[0].strip()
    for decomp in decomps[1:]:
        da = decomp.strip()
        if da != '':
            decomp_dict['dalist'].append(da)
            da_split = da.split('?')
            if len(da_split) == 1:
                if da[-1] == '?': # assume this is question
                    decomp_dict['dlist'].append( da_split[0].strip()+'?' )
                    decomp_dict['alist'].append( '' )
                else:    
                    decomp_dict['dlist'].append( '' )
                    decomp_dict['alist'].append( da_split[0].strip() )
            elif len(da_split) == 2:
                decomp_dict['dlist'].append( da_split[0].strip()+'?' )
                decomp_dict['alist'].append( da_split[1].strip() )
            elif da_split[-1].strip() == '': # if last ? is at the end, assume it's part of decomp answer
                decomp_dict['dlist'].append('?'.join(da_split[:-2])+'?' )
                decomp_dict['alist'].append( da_split[-2].strip()+'?' )
            else:
                decomp_dict['dlist'].append('?'.join(da_split[:-1])+'?' )
                decomp_dict['alist'].append( da_split[-1].strip() )
    return decomp_dict


def get_parsed_decomp_str(in_list):
    """ Return a list of parsed decomp strings """
    decomp_list = []
    for decomp_str in in_list:
        decomp_list.append( parse_decomp_str(decomp_str) )
    return decomp_list            


def get_parsed_decomp_by_key(decomp_list, key, join_list=True):
    """ Return a list of just the values in decomp_list[i]['key']
    """
    key_list = []
    for decomp_dict in decomp_list:
        val = decomp_dict[key]
        if join_list and type(val) == list:
            val = ' '.join(val).strip()
        key_list.append( val )
    return key_list
    

#######################
# HF Utils
#######################

def load_model(model_name="facebook/bart-large", checkpoint=None, loadwhat='both', 
               to_cuda=True, use_fp16=False):
    """ Load a model and set for prediction
    Usage: tokenizer, model = load_model(model_name, checkpoint)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if loadwhat == 'tokenizer_only': 
        return tokenizer
    else:
        if model_name == "facebook/bart-large":   
            my_model = MyBart
        elif model_name in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
            my_model = AutoModelForCausalLM
        elif model_name == "EleutherAI/gpt-j-6B":
            my_model = AutoModelForCausalLM
        else:
            my_model = AutoModelForPreTraining  # HF documentation indicates this gives right models for T5 and gpt2 as well as vanilla bart
        
        print(f"Loading model: {model_name}")
        if checkpoint is not None:
            print(f"Loading checkpoint from {checkpoint}")       
            model = my_model.from_pretrained(model_name, state_dict=torch.load(checkpoint))  
        else:
            print("No checkpoint loaded. Training from base pretrained model.")
            if use_fp16 and model_name == "EleutherAI/gpt-j-6B":
                model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
            else:
                model = my_model.from_pretrained(model_name) 
        
        if to_cuda:
            model.to(torch.device("cuda"))
        model.eval()
        print("Model loaded!")
    return tokenizer, model


def string_to_ids(input_string, tokenizer, indiv_digits=False, norm_numbers=True, norm='', special='Ġ', 
                  verbose=False, truncation=True, max_input_length=512, lower=True, 
                  append_eos=True, prepend_bos=True):
    """ Convert single input string to model-ready ids using cut-down version 
        of tokenisation pipeline
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
    if prepend_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    numtoks = len(ids)
    if verbose:
        print(f"Number of tokens: {numtoks+1}")
    if truncation and numtoks > max_input_length-1:
        ids = ids[:max_input_length-1]
        if verbose:
            print(f"Truncated to {max_input_length} tokens.")
    if append_eos:        
        ids = ids + [tokenizer.eos_token_id]    
    return ids


def decode_ids(res, tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    """         See https://huggingface.co/blog/how-to-generate for generation options
    Returns: Full result depending on which **generator params are set (e.g.including any scores, attns etc)
             if one of the params in **generator is return_dict_in_generate=True
             otherwise returns the decoded prediction.
    Note: if res of shape [id1, id4, id2, ..] result will be ['txt1', 'txt4', 'txt2', ..]
          if res of shape [[id1, id4, id2, ..]] result will be ['txt1 txt4 txt2 ..']
    """
    return tokenizer.batch_decode(res, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


def run_model(input_string, model, tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True,
              indiv_digits=False, norm_numbers=True, norm='', special='Ġ', verbose=False,
              truncation=True, max_input_length=512, lower=True, to_cuda=True,
              append_eos=True, prepend_bos=True, **generator_args):
    """ Run cut-down version of tokenisation and generation pipeline for single input string
    Usage:
    # input_string is either a sinle string of a list of strings
    # ** generator_args:
    #    greedy: model.generate(input_ids, max_length=50) 
    #    beam: model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True,num_return_sequences=2,no_repeat_ngram_size=2)
    #    sample: model.generate(input_ids, do_sample=True, max_length=50, top_k=0, temperature=0.7) # the lower the temp the greater the chance of picking high prob words
    #    topk: model.generate(input_ids, do_sample=True, max_length=50, top_k=50) # only sample from the top 50 words by prob each time
    #    nucleus: model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0) # (also called topP) choose from the top words whose collective prob exceeds p
    #    combo: model.generate(input_ids,do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3)
    
    # for gpt2 add: append_eos=False, prepend_bos=False,
    res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers,
                    num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                    output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
    # returns num_return_sequences outputs * num input samples = #samples
    #res.preds has ["pred1 sentence", "pred2 sentence"] for #samples = 2
    #res.sequences has output tokens as ids shape [#samples, max output len]
    #res.sequences_scores returns overall score (final beam score) of each returned seq [#samples]
    #res.scores is tuple of output num_toks entries of [#beams*#samples, vocab size] if input_string is a list of #samples
    
    pred, score = get_single_result(res, idx=0)        
    """
    ids = []
    if type(input_string) == list:
        for istr in input_string:
            idsingle = string_to_ids(input_string, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, 
                                norm=norm, special=special, verbose=verbose, truncation=truncation, 
                                max_input_length=max_input_length, lower=lower, append_eos=append_eos,
                                prepend_bos=prepend_bos)
            ids.append(idsingle)
    else:    
        ids = string_to_ids(input_string, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, 
                            norm=norm, special=special, verbose=verbose, truncation=truncation, 
                            max_input_length=max_input_length, lower=lower, append_eos=append_eos,
                            prepend_bos=prepend_bos)
        ids = [ids]
    #input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = torch.LongTensor(ids)
    if to_cuda:
        input_ids = input_ids.to(torch.device("cuda"))
    res = model.generate(input_ids, **generator_args)
    if isinstance(res, dict):        
        res['preds'] = decode_ids(res['sequences'], tokenizer, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        return res
    return decode_ids(res, tokenizer, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


def get_single_result(res, idx=0):
    """ Process single result from a res object
    Usage: pred, score = get_single_result(res, idx=the index into res)
    """
    return res.preds[idx].strip(), float(res.sequences_scores.detach().cpu().numpy()[idx])







