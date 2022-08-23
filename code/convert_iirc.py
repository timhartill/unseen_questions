#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:18:51 2022

Convert IIRC to std formats
https://allenai.org/data/iirc

"IIRC is a crowdsourced dataset of 13441 questions over 5698 paragraphs  from  English  Wikipedia,  
with most of the questions requiring information from one or more documents hyperlinked to the 
associated paragraphs, in addition to the original para- graphs themselves.""

from README:
The json has a list of passages. Each passage has [title, links, questions, text].
Overall:
[ 
  {
    title,
    links,
    questions,  (see below for structure)
    text        (Original passage for which the question was written)
  },
]

questions:
[
  {
    question,
    answer,
    question_links,  (Links labeled by question writer as relevant)
    context          (context selected by question answerer as necessary)
  }
]

Each context paragraph contains links to other Wikipedia articles contained in 'context_articles'..

@author: tim hartill

"""

import os
import json
import random
import re

import eval_metrics
import utils
import text_processing

UQA_DIR = eval_metrics.UQA_DIR

file_dev = '/home/thar011/data/iirc/iirc_train_dev/dev.json'
file_train = '/home/thar011/data/iirc/iirc_train_dev/train.json'
file_test = '/home/thar011/data/iirc/iirc_train_dev/iirc_test.json'  # test is labelled

file_context_articles = '/home/thar011/data/iirc/context_articles.json'

dev = json.load(open(file_dev))         #430  # dict_keys(['pid', 'questions', 'links', 'text', 'title'])
train = json.load(open(file_train))     #4754
test = json.load(open(file_test))       #514

context_articles = json.load(open(file_context_articles))  # 56550 dict with key lowercase title -> str of entire article incl href tags, formatting tags etc

# print(text_processing.create_paras(context_articles["angela scoular"]))

#TODO convert each answer type including concatinating multi-span answers together
#TODO create IIRC_OD_ANS = q->a
#TODO create IIRC_C = q+single gold para->a
#TODO create IIRC_G = q+single gold+all gold paras->a
#TODO create IIRC_P = q+single gold+all gold phrases->a
#TODO (eventually) create IIRC_R = q+single gold+ retrieved paras->a
#TODO (eventually) create IIRC_RS = q+single gold+ retrieved sents->a






