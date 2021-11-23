#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:39:12 2021

@author: tim hartill

Test different prompting formats with GPT-J


Notes: 
    Liu params: few-shot (k)=5, generate inferences (m)=20, nucleus p = 0.5, max 64 tokens or when hit \n
    West params: few-shot (k)=10, generate inferences (m)=10, nucleus p = 0.9, number examples, freq penalty=0.5

"""
import os
import numpy as np
from utils import load_model, run_model, get_single_result, load_prompt_template, fill_prompt_template, load_uqa_supervised

UQA_DIR = '/data/thar011/data/unifiedqa/'
PROMPT_DIR = os.path.join(UQA_DIR, 'prompts')
QASC_DIR = os.path.join(UQA_DIR, 'qasc')
QASC_DEV = os.path.join(QASC_DIR, 'dev.tsv')

model_name = "EleutherAI/gpt-j-6B"

tokenizer, model = load_model(model_name, checkpoint=None)

qasc_dev = load_uqa_supervised(QASC_DEV, return_parsed=True)

qasc_1_fact = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_single_fact_liu_v1.txt')
qasc_2_fact = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_multi_fact_sameliuquestions_v1.txt')
qasc_2_fact_numbered = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_multi_fact_numbered_sameliuquestions_v1.txt')

p_qasc_1 = fill_prompt_template(qasc_1_fact, query=qasc_dev[0]['q_only'])
p_qasc_2 = fill_prompt_template(qasc_2_fact, query=qasc_dev[0]['q_only'])
p_qasc_2_numbered = fill_prompt_template(qasc_2_fact_numbered, query=qasc_dev[0]['q_only'])


res = run_model(p_qasc_1, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2_numbered, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2_numbered, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2_numbered, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.5, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)



#TODO Having both facts seems to work better than just one fact.
#TODO Adding example numbering seems to work better at keeping outputs centered on the query topic
#TODO 0.9 seems to work a bit better than 0.5 - 0.5 has > proportion of blank knowledge and not obviously more diversity
#TODO Need to test with having more examples in the prompt

#TODO Can we use the cosine similarity of each sentence in order to determine which ones to select?



