#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:56:47 2021

@author: tim hartill

Create q[+mc]+e->a datasets using explanation components generated from a language model
eg via explanation_component_generation_gptj.py

full set = include_keys=['expl_depth', 'noun', 'verb', 'general', 'where', 'when', 'size']

"""

import utils
import language_modelling

tokenizer = utils.load_model(model_name="facebook/bart-large", loadwhat='tokenizer_only')

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=-1, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['expl_depth'])

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=-1, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['expl_depth', 'noun', 'general'])

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=-1, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['expl_depth', 'noun', 'general', 'where', 'when', 'size'])

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=-1, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['noun', 'general', 'where', 'when', 'size'])

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=-1, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['expl_depth', 'noun', 'verb', 'general', 'where', 'when', 'size'])

samples = language_modelling.create_expl(tokenizer, in_dset='strategy_qa_od_ans', 
                                         expl_dset='strategy_qa_expl_ans', 
                                         file='dev', 
                                         expl_model='', methods=['rand'], su_stop=50, 
                                         soft_max_tokens=350, verbose=True, seed=42, save=True,
                                         include_keys=['expl_depth', 'noun', 'general'])



