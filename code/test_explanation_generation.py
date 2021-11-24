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
QASC_DEV = os.path.join(UQA_DIR, 'qasc', 'dev.tsv')
QASC_EXPLANATION_FILE = os.path.join(UQA_DIR, 'qasc_od_expl', 'train.tsv')

model_name = "EleutherAI/gpt-j-6B"

tokenizer, model = load_model(model_name, checkpoint=None)

qasc_dev = load_uqa_supervised(QASC_DEV, return_parsed=True)

qasc_train_expl = load_uqa_supervised(QASC_EXPLANATION_FILE, ans_lower=False, return_parsed=True)
num_q = len(qasc_train_expl)
np.random.seed(42)
prompt_indices = np.random.choice(num_q, 100, replace=False)
# array([5914, 5425, 1430, 7324, 4028, 1009, 3172, 2892, 3985, 5023, 4074,
#       1302, 4471, 7541,  554, 6864,  483, 6908, 6159, 5057, 5170, 2199,
#       3837, 2345, 5137, 7331, 4825, 1242, 1882, 5519, 4525, 1730, 5861,
#       6091, 2406, 2302,  233,  794,  866, 3333, 1400, 1744, 7937, 6224,
#       4510, 4922,  932, 3567, 4151, 1737,  318, 2995, 2338, 5513, 7743,
#       1926, 3012, 1575, 4113,  349, 3355, 7716, 4606, 3942, 1010, 3844,
#        239, 6438, 3238, 6879,  748, 6218, 5324, 3149, 1295, 7685, 2867,
#       7977, 3217, 6642, 4270, 7165, 6968, 5815, 6594, 3018, 4394, 2663,
#       6084,  453, 3995, 7780, 6612, 6075, 4668, 5548, 2348, 8088, 4674,
#       6195])
prompt_5 = prompt_indices[:5]
prompt_10 = prompt_indices[:10]
prompt_32 = prompt_indices[:32]
prompt_rand = prompt_indices[np.random.choice(prompt_indices.shape[0], 20, replace=False)]

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



