#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:39:12 2021

@author: tim hartill

Generate explanation components with GPT-J


Notes: 
    Liu params: few-shot (k)=5, generate inferences (m)=20, nucleus p = 0.5, max 64 tokens or when hit \n
    West params: few-shot (k)=10, generate inferences (m)=10, nucleus p = 0.9, number examples, freq penalty=0.5
    
    
  

"""
import os
import numpy as np
import utils
import language_modelling
import text_processing

MAX_INPUT_LENGTH=1000
GEN_DEPTH=1
SUNIQUE=-1
LOWER=False
VERBOSE=True
ADD_NOUN=['general', 'where', 'when', 'size']
ADD_VERB=['general']
MAX_NEW_TOKENS=64
NUM_RETURN_SEQUENCES=10

UQA_DIR = '/data/thar011/data/unifiedqa/'
PROMPT_DIR = os.path.join(UQA_DIR, 'prompts')
TEST_TEMPLATES = language_modelling.load_templates([os.path.join(PROMPT_DIR, 'sqa_k12_raw.txt')])

cuda_device = 0
model_name = "EleutherAI/gpt-j-6B"
tokenizer, model = utils.load_model(model_name, checkpoint=None, cuda_device=cuda_device)

def save_candidates(dset='strategy_qa_expl_ans', file='train'):
    """ Create candidate explanation components and save them in the same dir as the source file
    Output jsonl format: [{'question':'full question with context', 'answer':'ans', 'q_only':'..', 
                           'mc_options': 'mc options', 'context':'non-mc context if any', 
                           'expl_components':{ expl_components format} }]
    expl_components format: {'question':q, 'expl_depth':[ ['depth 0 expl components 1', 'd0 ec 2', ..], ['depth 1 expl components', ..], .. ],
                              'noun':{'general':{'Aristotle':['Sentence 1', 'Sentence 2', ...], 'a laptop':[...]}, ...}
                              'verb':{'general':{'running':[...], 'jumping':[...]}}
                             }
    """
    infile = os.path.join(UQA_DIR, dset, file+'.tsv')  # usually the q[+mc]+e->a version but any uqa formatted file will work
    if VERBOSE:
        print(f"Loading: {infile}")
    samples = utils.load_uqa_supervised(infile, ans_lower=False, return_parsed=True)
    questions = [text_processing.format_sentence(s['q_only'], endchar='') for s in samples]    
    components = language_modelling.gen_expl(TEST_TEMPLATES, model, tokenizer, questions, verbose=VERBOSE, lower=LOWER,
                                                max_input_length=MAX_INPUT_LENGTH, gen_depth=GEN_DEPTH, su_stage_1_stop=SUNIQUE, 
                                                add_noun=ADD_NOUN, add_verb=ADD_VERB,
                                                max_new_tokens=MAX_NEW_TOKENS, do_sample=True, top_k=0, top_p=0.9, temperature=0.7,
                                                num_return_sequences=NUM_RETURN_SEQUENCES, output_scores=False, return_dict_in_generate=True, 
                                                pad_token_id=tokenizer.eos_token_id)
    samples = utils.add_key(samples, components, key='expl_components')
    outfile = os.path.join(UQA_DIR, dset, file+'_expl_components.jsonl')
    utils.saveas_jsonl(samples, outfile, verbose=VERBOSE)    
    return

save_candidates(dset='strategy_qa_expl_ans', file='dev')
save_candidates(dset='strategy_qa_expl_ans', file='train')


