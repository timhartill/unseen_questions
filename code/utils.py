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
import random
import copy
import time
from html import unescape
import fnmatch
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import GPTJForCausalLM, AutoModelForCausalLM
from bart import MyBart
from text_processing import normalize_num, split_digits_special

MULTI_ANS_SEP = '#!#'

#####################################
# General utils
#####################################

def get_timestamp():
    """ Return a timestamp of the current local time in filename-friendly format
    """
    t = time.localtime()
    return str(t.tm_year) + '_' + str(t.tm_mon) + '_' + str(t.tm_mday) + '_' + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + '_' + str(t.tm_sec)


def list_files_pattern(dirtolist, pattern='*'):
    """ Returns a list of files in a dictionary matching a pattern
    """
    return [file for file in os.listdir(dirtolist) if fnmatch.fnmatch(file, pattern)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

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
    if verbose:
        print('JSON as text successfully loaded. Loading...')
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print(f'Text successfully converted to JSON. Number of json messages: {len(all_json_list)}')
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
                if i > 0 and i % update == 0:
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


def load_uqa_supervised(file, ans_lower=True, verbose=True, return_parsed=False):
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

            if answer.lstrip().startswith(MULTI_ANS_SEP): # #!# answer 1#!# answer 2 #!# -> ['answer 1','answer 2']
                answer = answer.strip().strip(MULTI_ANS_SEP).split(MULTI_ANS_SEP)
                    
            if type(answer) == list:
                answer = [a.strip() for a in answer]
            else:
                answer = answer.strip()
                
            questions.append( question.strip() )
            answers.append ( answer )
            ctr += 1
    if verbose:
        print(f"Successfully loaded {len(questions)} rows.")
    if return_parsed:
        return parse_uqa_supervised(questions, answers)
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


def parse_uqa_supervised(questions, answers):
    """ Convert [questions], [answers] into jsonl style format:
        [{'question': 'q txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
    """
    uqa_parsed = []
    for q, a in zip(questions, answers):
        curr = {'question': q, 'answer': a}
        q_list = q.split('\\n')
        curr['q_only'] = q_list[0].strip()
        if len(q_list) >= 2:
            q2 = q_list[1].strip()
            if q2.startswith('(A)'):
                curr['mc_options'] = q2
                if len(q_list) >= 3:
                    curr['context'] = q_list[2].strip()
                else:
                    curr['context'] = ''
            else:
                curr['mc_options'] = ''
                curr['context'] = q2
        uqa_parsed.append(curr)
    return uqa_parsed


def return_sublist(sample_list, indices, key=None):
    """ Return sublist of list entries matching the list of indices. 
    If key is specified assumes sample_list in jsonl format and returns a simple list of values from that key
    """
    sublist = []
    for idx in indices:
        if key is None:
            sublist.append(sample_list[idx])
        else:
            sublist.append(sample_list[idx][key])           
    return sublist


def add_key(sample_list, new_data, key, make_copy=True):
    """ Add [new data] into new key of [sample_list] (which is assumed 'jsonl' format).
    |sample_list| must equal |new_data|...
    new_data can be a simple list or a jsonl style list
    """
    if make_copy:
        sample_list = copy.deepcopy(sample_list)
    for s,n in zip(sample_list, new_data):
        s[key] = n
    if make_copy:
        return sample_list
    return


def add_constant(sample_list, const, key):
    """ Add a specific value to a new key of each element of a jsonl-formatted list """
    for s in sample_list:
        s[key] = const
    return


def build_dict_from_jsonl(jsonl, key):
    """ Convert jsonl list into a dict with key 'key'.
    """
    out_dict = {q[key]:q for q in jsonl}
    return out_dict


# from https://github.com/castorini/transformers-arithmetic/blob/main/main.py
def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz") -> str:
    """ convert base 10 integer into another base """
    return ((num == 0) and numerals[0]) or (
        convert_to_base(num // base, base, numerals).lstrip(numerals[0]) + numerals[num % base])


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


def unique_preserve_order(seq):
    """ From https://stackoverflow.com/a/480227/1493011
    Remove dups from list while preserving order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def create_uqa_example(question, context=None, answer=None, append_nl=True, append_q_char='?'):
    """ Returns an example in uqa format
    Note: to create a self supervised example put the input para in question and set context = ' ' and append_q_char='.'
            to create a closed book example likewise set context = ' ' but append_q_char='' or '?'
    """
    sample = question.strip().replace('\n', '').replace('\t', '')
    if sample[-1] in ['.','!','?'] and append_q_char != '':
        sample = sample[:-1]
    if sample[-1] != append_q_char:
        sample += append_q_char
    sample += ' \\n'
    if context is not None and context != '':
        sample += ' ' + context.strip().replace('\n','').replace('\t', '')
        
    if answer is not None:
        if type(answer) == str and answer != '':
            sample += ' \t' + answer.strip().replace('\n', '').replace('\t', '')
        elif type(answer) == list and len(answer) != 0:
            sample += ' \t' + MULTI_ANS_SEP + MULTI_ANS_SEP.join(answer).replace('\n', '').replace('\t', '') + MULTI_ANS_SEP
        
    if append_nl:
        sample += '\n'
    return sample


def create_uqa_context(mc_options=None, para_context=None):
    """ Create a UQA context depending on whether there are multi-choice options and/or a paragraph context as either:
        mc options
        mc options\\npara_context
        para_context
        or None
    """
    context = ''
    if para_context is not None and para_context.strip() != '':
        context = para_context.strip()
    if mc_options is not None and mc_options.strip() != '':
        if context != '':
            context = mc_options.strip() + '\\n' + context
        else:
            context = mc_options.strip()
    if context == '':
        context = None
    return context


def save_uqa(out, out_dset_dir, file):
    """ Save uqa datafile from list of uqa formatted examples that end in \n
    """
    if os.path.splitext(file)[1] != '.tsv':
       file += '.tsv' 
    os.makedirs(out_dset_dir, exist_ok=True)
    outfile = os.path.join(out_dset_dir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))
    return


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
    

def get_checkpoint(args, logger):
    """ Return checkpoint if there is one
    """
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        logger.info(f"Error running Predict: Specified checkpoint doesnt exist: Checkpoint={args.checkpoint}") 
        assert os.path.exists(args.checkpoint), "Exiting. Please remediate and restart."
    checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
    if not os.path.exists(checkpoint):
        checkpoint = None
    return checkpoint


#######################
# Pytorch utils
#######################

def num_gpus():
    """ return number of visible gpus 
        Note: visible gpus will be numbered 0, 1,.. num_visible_gpus-1
    """
    return torch.cuda.device_count()

def empty_cache():
    """ empty pytorch cache, sometimes solves issue of OOM not "resetting" and continuing to cause OOM errors """
    torch.cuda.empty_cache()

def check_mem():
    """ return list of free memory per visible gpu """
    os.environ['TOKENIZERS_PARALLELISM']='true'
    mem = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read()
    mem = mem.strip().split('\n')
    mem = [float(m.split(', ')[0])-float(m.split(', ')[1]) for m in mem]
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpus is not None:
        gpus = [int(g) for g in gpus.split(',')]
        vis_mem = []
        for gpu in gpus:
            for i, m in enumerate(mem):
                if gpu == i:
                    vis_mem.append(mem[i])
                    break
        return vis_mem
    return mem

def get_model_device(model):
    """ returns device that model is on i.e. the device that the first parameter is on"""
    return next(model.parameters()).device


# Below from and adapted from https://github.com/facebookresearch/multihop_dense_retrieval:

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def load_saved(model, path, exact=True, strict=False):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    def filter(x): return x[7:] if x.startswith('module.') else x
    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict, strict=strict) #TJH Added , strict=False: when strict=True can't load orig mdr models/..pt due to imcompatible roberta versions
    return model

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def convert_to_half(sample):
    if len(sample) == 0:
        return {}

    def _convert_to_half(maybe_floatTensor):
        if torch.is_tensor(maybe_floatTensor) and maybe_floatTensor.type() == "torch.FloatTensor":
            return maybe_floatTensor.half()
        elif isinstance(maybe_floatTensor, dict):
            return {
                key: _convert_to_half(value)
                for key, value in maybe_floatTensor.items()
            }
        elif isinstance(maybe_floatTensor, list):
            return [_convert_to_half(x) for x in maybe_floatTensor]
        else:
            return maybe_floatTensor

    return _convert_to_half(sample)


#######################
# HF Utils
#######################

def load_model(model_name="facebook/bart-large", checkpoint=None, loadwhat='both', 
               to_cuda=True, use_fp16=False, cuda_device=None):
    """ Load a model and set for eval
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
            if cuda_device is None:  # else typically cuda_device = 0-based int id of particular device
                cuda_device = "cuda"
            model.to(torch.device(cuda_device))
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
              indiv_digits=False, norm_numbers=False, norm='', special='Ġ', verbose=False,
              truncation=True, max_input_length=512, lower=True, #to_cuda=True, cuda_device=None,
              append_eos=True, prepend_bos=True, only_decode_new=False, cut_at_nl=False, **generator_args):
    """ Run cut-down version of tokenisation and generation pipeline for single input string
    Usage:
    # input_string is either a single string of a list of strings
    # for LM typically: append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True (non-LM typically use defaults)
    # ** generator_args:
    #    greedy: model.generate(input_ids, max_length=50) 
    #    beam: model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True,num_return_sequences=2,no_repeat_ngram_size=2)
    #    sample: model.generate(input_ids, do_sample=True, max_length=50, top_k=0, temperature=0.7) # the lower the temp the greater the chance of picking high prob words
    #    topk: model.generate(input_ids, do_sample=True, max_length=50, top_k=50) # only sample from the top 50 words by prob each time
    #    nucleus: model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0) # (also called topP) choose from the top words whose collective prob exceeds p so lower p = fewer but higher prob words to choose from
    #    combo: model.generate(input_ids,do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3)
    
    # for gpt2 add: append_eos=False, prepend_bos=False,
    res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers,
                    num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                    output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
    # returns num_return_sequences outputs * num input samples = #samples
    #res.preds has ["pred1 sentence", "pred2 sentence"] for #samples = 2
    #res.sequences has output tokens as ids shape [#samples, max output len]
    #res.sequences_scores returns overall score (final beam score) of each returned seq [#samples]  NOTE: don't get sequences_scores for nucleus sampling'
    #res.scores is tuple of output num_toks entries of [#beams*#samples, vocab size] if input_string is a list of #samples
    #   see https://huggingface.co/docs/transformers/internal/generation_utils
            https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    pred, score = get_single_result(res, idx=0)        
    """
    ids = []
    start_decode = 99999999
    if type(input_string) == list:
        for istr in input_string:
            idsingle = string_to_ids(istr, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, 
                                norm=norm, special=special, verbose=verbose, truncation=truncation, 
                                max_input_length=max_input_length, lower=lower, append_eos=append_eos,
                                prepend_bos=prepend_bos)
            if len(idsingle) < start_decode:
                start_decode = len(idsingle)
            ids.append(idsingle)
    else:    
        ids = string_to_ids(input_string, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, 
                            norm=norm, special=special, verbose=verbose, truncation=truncation, 
                            max_input_length=max_input_length, lower=lower, append_eos=append_eos,
                            prepend_bos=prepend_bos)
        start_decode = len(ids)
        ids = [ids]
    #input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = torch.LongTensor(ids)
    if model.device != input_ids.device:
        input_ids = input_ids.to(model.device)
    #if to_cuda:
    #    if cuda_device is None:  # else typically cuda_device = 0-based int id of particular device
    #        cuda_device = "cuda"
    #    input_ids = input_ids.to(torch.device(cuda_device))
    res = model.generate(input_ids, **generator_args)
    if isinstance(res, dict):     
        if only_decode_new:  #LMs output orig input + preds, skip orig input decode
            res['preds'] = decode_ids(res['sequences'][:, start_decode:], tokenizer, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            res['start_decode'] = start_decode
            if cut_at_nl:  #truncate to first \n to prevent additional babble 
                res['preds'] = [p.strip().split('\n')[0] for p in res['preds']]
        else:
            res['preds'] = decode_ids(res['sequences'], tokenizer, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            res['start_decode'] = 0
            
        res['sequences'] = res.sequences.cpu().numpy()
        if res.get('scores') is not None:
            scores = []
            for score in res['scores']:
                scores.append(score.cpu().numpy())
            res['scores'] = scores
        if res.get('sequences_scores') is not None:
            res['sequences_scores'] = res.sequences_scores.cpu().numpy()
        return res
    return decode_ids(res, tokenizer, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


def get_single_result(res, idx=0):
    """ Process single result from a res object
    Usage: pred, score = get_single_result(res, idx=the index into res)
    """
    return res.preds[idx].strip(), float(res.sequences_scores[idx]) # float(res.sequences_scores.detach().cpu().numpy()[idx])


# Utils related to retrieval below here

def encode_text(tokenizer, text, text_pair=None, max_input_length=512, 
                truncation=True, return_tensors="pt", padding=False):
    """ Encode text in standard way using various options
    padding='max_length' will pad to max_input_length
    truncation = True with text_pair will use 'longest first' strategy ie iteratively remove token from current longest of text or text_pair
    With text_pair can also set "only_second" to just truncate text_pair or 'only_first' to just truncate text
    tokenizer.encode_plus('a') : {'input_ids': [0, 102, 2], 'attention_mask': [1, 1, 1]}
    tokenizer.encode_plus('a', text_pair='b') : {'input_ids': [0, 102, 2, 2, 428, 2], 'attention_mask': [1, 1, 1, 1, 1, 1]}
    tokenizer.batch_encode_plus(['a', 'b', 'c']) : {'input_ids': [[0, 102, 2], [0, 428, 2], [0, 438, 2]], 'attention_mask': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}
    tokenizer.batch_encode_plus([['a','b'], ['b','c'], ['c','d']]) : {'input_ids': [[0, 102, 2, 2, 428, 2], [0, 428, 2, 2, 438, 2], [0, 438, 2, 2, 417, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}
    
    """
    if type(text) == str:
        encode_dict = tokenizer.encode_plus(text, text_pair=text_pair, max_length=max_input_length, 
                                            truncation=truncation, padding=padding, return_tensors=return_tensors)
    else:
        encode_dict = tokenizer.batch_encode_plus(text, max_length=max_input_length, 
                                                  truncation=truncation, padding=padding, return_tensors=return_tensors)
    return encode_dict


def encode_query_paras(text, title=None, sentence_spans=None, selected_sentences=None, 
                       use_sentences=False, prepend_title=False, title_sep = ':'):
    """ Encode the query as either just the paragraph or as selected sentences from the paragraph prepended by the title
    sentence_spans = [ [s1startidx, s1endidx], [s2startidx, s2endidx], ...]
    selected sentences = [sentenceidx1, sentenceidx2, ...]    
    """
    if not use_sentences or len(selected_sentences)==0:
        txt = text.strip()
        if txt[-1] not in ['.', '?', '!']:
            txt += '.'            
        return txt
    newtext = ''
    for sent_idx in selected_sentences:
        if sent_idx < 0 or sent_idx >= len(sentence_spans): #hpqa, fever have a few annotation errors where sent_idx > num sentences
            continue
        start, end = sentence_spans[sent_idx]
        sent = text[start:end].strip()
        if sent[-1] not in ['.','?','!']:
            sent += '.'
        newtext = newtext + ' ' + sent
    if newtext.strip() == '':  # If no sents found due to annotation errors, use truncated para
        newtext = text[:600].strip() + '...'
    if prepend_title:
        newtext = unescape(title.strip() + title_sep) + ' ' + newtext
    return newtext.strip()


########################
#  Utils for wikipedia docs ######
########################

def build_title_idx(docs, verbose=False):
    """ The HPQA corpus files contain some hrefs that have casing different to the actual title casing.
    Simply making everything lowercase wont work as this creates a smallish number of duplicates.
    So we build a dict with key title.lower():
        {'title_lower': [{'title': the title, 'id': id of this title}]}
        Also title_lower is UNESCAPED meaning " ' & < > are as here not escaped like the orig titles which have eg &amp; for &
    """
    titledict = {}
    dupdict = {}
    for i, doc in enumerate(docs):
        tlower = unescape(doc['title'].lower()) # unescaped
        title_entry = {'title': doc['title'], 'id':doc['id'], 'idx': i} # not escaped so matches title in corpus
        if titledict.get(tlower) is None:
            titledict[tlower] = [title_entry]
        else:
            titledict[tlower].append(title_entry)
            if verbose:
                print(f"Dup lowercase: {tlower}  New Entry:{title_entry}")
            dupdict[tlower] = titledict[tlower]
        if i % 1000000 == 0:
            print(f"Processed: {i} Dups so far:{len(dupdict)}")
    print(f"Total dups: {len(dupdict)}")
    return titledict, dupdict


def map_title_case(hlink, titledict, id_type='idx', verbose=False):
    """ Some titles in HPQA abstracts have incorrect casing. Attempt to map casing.
    hlink is a wiki doc title not necessarily from and hlink..
    titledict has key 'title' with entry(s) like: [{'title': 'Chinnar Wildlife Sanctuary', 'id': '9642568', 'idx': 0}]
    id_type = 'id' will return wiki doc id, id_type='idx' will return idx of this title in docs
    Note unescape will map eg &amp; to & but will have no effect on already unescaped text so can pass either escaped or unescaped version
    
    """
    tlower = unescape(hlink.lower())
    tmap = titledict.get(tlower)
    status = 'nf'
    idx = -1
    if tmap is not None:                # if not found at all just return existing hlink
        if len(tmap) == 1:
            if hlink != tmap[0]['title']:
                if verbose:
                    print(f"Hlink case different. Orig:{hlink} actual: {tmap[0]['title']}")
                status = 'sc'
            else:
                status = 'sok'
            hlink = tmap[0]['title']    # only one candidate, use that
            idx = tmap[0][id_type]
        else:
            for t in tmap:
                if hlink == t['title']: # exact match amongst candidates found, use that
                    return hlink, 'mok', t[id_type]
            hlink = tmap[0]['title']    # otherwise just return first
            idx = tmap[0][id_type]
            status = 'mc'
            if verbose:
                print(f"Hlink lower:{tlower} No exact match found so assigning first: {hlink}")
    return hlink, status, idx


def get_hyperlinked_docs(docs, titledict, curr_d_idx, curr_p_idx, exclude=set()):
    """ Return indices of docs that are linked to by curr_d_idx, curr_p_idx
    """
    docs_linked_to_idx = set()
    docs_linked_to_id = set()
    for title in docs[curr_d_idx]['paras'][curr_p_idx]['hyperlinks_cased'].keys():
        new_title, status, d_idx = map_title_case(title, titledict)
        if d_idx != -1 and d_idx not in exclude:
            docs_linked_to_idx.add(d_idx)
            docs_linked_to_id.add(docs[d_idx]['id'])
    return docs_linked_to_idx, docs_linked_to_id


def get_paras(docs, d_idxs, p_idxs=[0]):
    """ Return paras for a set of doc idxs
    """
    paras = []
    for d_idx in d_idxs:
        para = ''
        for p_idx in p_idxs:
            if p_idx < len(docs[d_idx]['paras']) and p_idx > -1:
                para += ' ' + docs[d_idx]['paras'][p_idx]['text']
        paras.append( {'doc_id': docs[d_idx]['id'], 'title': docs[d_idx]['title'], 'title_unescaped': unescape(docs[d_idx]['title']),
                       'text': para.strip(), 'para_idxs': p_idxs} )
    return paras


def get_para_idxs(para_list):
    """ Return dict of {title: {idx: [idx in pos_paras]}} idx is a list to allow for possibility of same title duplicated in pos paras either for same title/difft paras or in case of FEVER same title, same para but difft sentence annotation
    para_list: [[{'title': 'Kristian Zahrtmann', 'text': 'Peder Henrik Kristian Zahrtmann, known as Kristian Zahrtmann, (31 March 1843 – 22 June 1917) was a Danish painter. He was a part of the Danish artistic generation in the late 19th century, along with Peder Severin Krøyer and Theodor Esbern Philipsen, who broke away from both the strictures of traditional Academicism and the heritage of the Golden Age of Danish Painting, in favor of naturalism and realism.',
                   'sentence_spans': [[0, 114], [114, 408]],
                   'sentence_labels': [0, 1]}, ... ]
    returns: {'Kristian Zahrtmann': [0],
                 'Peder Severin Krøyer': [1],
                 'Ossian Elgström': [2]}
    """
    para_idx_dict = {}
    for idx, para in enumerate(para_list):
        if para_idx_dict.get(para['title']) is None:
            para_idx_dict[para['title']] = []
        para_idx_dict[para['title']].append( idx )
    return para_idx_dict
        
    
def get_para_docidxs(para_list, titledict):
    """ return list of doc idxs for para titles
    """
    para_doc_ids = []
    for para in para_list:
        new_title, status, d_idx = map_title_case(para['title'], titledict)    
        para_doc_ids.append(d_idx)  # d_idx = -1 if not found
    return para_doc_ids


def add_neg_paras_single(docs, titledict, s):
    """ Create negative paras for a sample using hyperlinks
    """
    pos_doc_idxs = get_para_docidxs(s['pos_paras'], titledict)
    neg_idxs = set()
    pos_doc_set = set(pos_doc_idxs)
    status = 'ok'
    for idx in pos_doc_idxs:
        if idx != -1:
            curr_negs = set()
            num_paras = len(docs[idx]['paras'])
            for i in range(num_paras):
                docs_linked_to_idx, _ = get_hyperlinked_docs(docs, titledict, idx, i, exclude=pos_doc_set)
                curr_negs = curr_negs.union(docs_linked_to_idx)
                if len(curr_negs) >= 10:
                    break
            neg_idxs = neg_idxs.union(curr_negs)
    neg_paras = get_paras(docs, neg_idxs)
    final_negs = [{'title': unescape(n['title']), 'text': n['text'], 'src':'hl'} for n in neg_paras]
    if len(final_negs) == 0:
        status = 'nf'
    elif len(final_negs) < 10:
        status = 'sf'
    while len(final_negs) < 10:
        rand_idx = random.randint(0, len(docs)-1)
        if rand_idx not in neg_idxs and rand_idx not in pos_doc_set:
            n = docs[rand_idx]['paras'][0]
            title = unescape(docs[rand_idx]['title'])
            final_negs.append({'title': title, 'text': n['text'], 'src':'rd'})
    return final_negs, status


def add_neg_paras(docs, titledict, split):
    """ Add adversarial negatives by taking hyperlinked docs to the pos paras that are not any of the pos paras
    """
    status_counts = {'ok':0, 'nf':0, 'sf':0}
    for i,s in enumerate(split):
        s['neg_paras'], status = add_neg_paras_single(docs, titledict, s)
        status_counts[status] += 1
        if i % 5000 == 0:
            print(f'Processed: {i}')
    print(f"Status counts: total:{len(split)} {status_counts}")
    return


def consistent_bridge_format(sample):
    """ translate sample bridge format into consistent 'multi' format:
         eg sample['bridge'] = [['Ossian Elgström', 'Kristian Zahrtmann', 'Peder Severin Krøyer'], ['bananarama'], ['tango']]
            means all paras from sample['bridge'][0] (but in any order) must come before sample['bridge'][1] which in turn (in any order if > 1 para) must come before sample['bridge'][2] ..

    hpqa: comparison: - no bridge key -> [ [p1, p2] ]
          bridge: bridge has final para -> [[p1], [p2]]
    squad, nq, tqa: type='' -> [ [p1] ] 
    fever:  bridge [p1, p2] -> [[p1], [p2]]  
            bridge [p1] -> [ [p1] ]
    """
    if sample.get('src') is None: #if no src key assume this is MDR-formatted HPQA data and reformat
        if sample.get('bridge') is not None and type(sample['bridge']) is not list:
            sample['bridge'] = [ sample['bridge'] ]
        sample['src'] = 'hotpotqa'
    if sample.get('_id') is None:  #bqa id key = 'id'
        if sample.get('id') is not None:
            sample['_id'] = sample['id']
        else:
            sample['_id'] = 'noid'
        
    if sample['type'] == 'comparison': #bqa hpqa sample
        bridge_list = [ [sample['pos_paras'][0]['title'], sample['pos_paras'][1]['title']] ]
    elif sample['type'] == 'bridge':  #bqa hpqa sample
            for para in sample['pos_paras']:
                if para['title'] != sample['bridge'][0]:
                    start_para = para
                else:
                    bridge_para = para
            bridge_list = [ [start_para['title']], [bridge_para['title']] ]    
    elif sample['type'] == '': # single hop eg squad, nq, tqa 
        bridge_list = [ [sample['pos_paras'][0]['title']] ]    
    elif sample['type'] == 'fever':
        if len(sample['bridge']) == 1:
            bridge_list = [ sample['bridge'] ]
        else:
            if len(set(sample['bridge'])) == len(sample['bridge']): # multiple & unique paras 
                bridge_list = [ [title] for title in sample['bridge']]  # unclear whether order matters but treat as though it does matter
            else:
                bridge_list = [ list(set(sample['bridge'])) ] # para title repeated for difft sentence labels in FEVER multi
                para_idxs = get_para_idxs(sample['pos_paras'])
                new_pos_paras = []
                for t in para_idxs:
                    idx_list = para_idxs[t]
                    new_para = copy.deepcopy(sample['pos_paras'][idx_list[0]])
                    if len(idx_list) > 1: # multiple identical paras, merge sentence labels
                        for idx in idx_list[1:]:
                            merge_labels = sample['pos_paras'][idx]['sentence_labels']
                            new_para['sentence_labels'].extend(merge_labels)
                        new_para['sentence_labels'].sort()
                        new_pos_paras.append(new_para)
                    else:
                        new_pos_paras.append( new_para )
                sample['pos_paras'] = new_pos_paras                                
    elif sample['type'] == 'multi':
        bridge_list = sample['bridge']
        
    sp_gold = []
    for para in sample['pos_paras']:
        for sent_idx in para['sentence_labels']:
            sp_gold.append( [para['title'], sent_idx] )
    sample['sp_gold'] = sp_gold           # full evidence annotation over all evidential paras 
    sample['num_hops'] = len(flatten(bridge_list))
    sample['bridge'] = bridge_list 
    return       
    
    

