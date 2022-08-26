#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:18:51 2022

Convert IIRC to std formats
https://allenai.org/data/iirc
https://github.com/jferguson144/IIRC-baseline

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
import itertools
from html import unescape
from tqdm import tqdm

import eval_metrics
import utils
import text_processing

UQA_DIR = eval_metrics.UQA_DIR
addspecialtoksdict = eval_metrics.special_tokens_dict  # test tokenization length with ind. digit tokenization...
tokenizer = utils.load_model(model_name='facebook/bart-large', loadwhat='tokenizer_only', special_tokens_dict=addspecialtoksdict)        

max_toks = 512
added_bits = len(tokenizer.tokenize('<s>. \\n</s>'))
max_toks = max_toks - added_bits
print(f"Max num tokens for text after allowing for BOS, EOS etc: {max_toks}")


file_dev = '/home/thar011/data/iirc/iirc_train_dev/dev.json'
file_train = '/home/thar011/data/iirc/iirc_train_dev/train.json'
file_test = '/home/thar011/data/iirc/iirc_train_dev/iirc_test.json'  # test is labelled

file_context_articles = '/home/thar011/data/iirc/context_articles.json'





train = json.load(open(file_train))     #4754
test = json.load(open(file_test))       #514
dev = json.load(open(file_dev))         #430  # dict_keys(['pid', 'questions', 'links', 'text', 'title'])

context_articles = json.load(open(file_context_articles))  # 56550 dict with key lowercase title -> str of entire article incl href tags, formatting tags etc

def get_article(title):
    article = context_articles.get(title.strip())  # 36 not lowercased..
    if article is None:
        article = context_articles.get(title.strip().lower())
    return article

def get_random_article(exclude=[]):
    """ Get a random title whose title is not in exclude
    """
    title = random.choice(list(context_articles.keys())) 
    while title in exclude:
        title = random.choice(list(context_articles.keys()))
    return title, get_article(title)

def get_random_para(exclude=[]):
    """ Get a random paragraph from a random article
    """
    title, article = get_random_article(exclude)
    para_list = text_processing.create_paras(article)
    while para_list == []:
        title, article = get_random_article(exclude)
        para_list = text_processing.create_paras(article)        
    para = random.choice(para_list)
    return title, text_processing.format_sentence(para)
    

def get_para(article_no_tags, sent, key='\n', widen_by=0, max_chars=600):
    """ Try to get a full para demarcated by \n or a full sentence or sentences demarcated by '.'
    """
    findstart = article_no_tags.find( sent.strip() )
    if findstart == -1:
        return '#NF#'
    else:
        if findstart-widen_by <= 0:
            para_start = 0
        else:
            para_start =  max(0, article_no_tags.rfind(key, 0, findstart-widen_by)+1)  # make parastart the last \n before the start of the substring snippet
        para_end = min(len(article_no_tags), article_no_tags.find(key, findstart+len(sent)+widen_by))
        if para_end-para_start > max_chars:  # #if too big, just grab the key sentence 
            return text_processing.format_sentence(sent[:max_chars])  # in some cases the sent text itself is huge
        #print(f"findstart:{findstart}  para_start:{para_start}  para_end:{para_end}")    
    return text_processing.format_sentence(article_no_tags[para_start:para_end])


def consolidate_contexts(title_main, question):
    """ Takes'gold_contexts' key to each question: {'title': {'gold_paras': ['Para 1', 'Para 2', ..]},  #title = 'main' or actual title
                                                         {'neg_paras': ['Para 1']}  # each title has only 'gold_paras' xor 'neg_paras' 
        and consolidates paras into new 'final_gold_context' and 'final_initial_context' keys            
    """
    golds = {}
    negs = {}
    for rawtitle in question['gold_contexts']:
        title = unescape(title_main).strip() if rawtitle == 'main' else unescape(rawtitle).strip()
        if question['gold_contexts'][rawtitle].get('gold_paras') is not None:
            para_combo = ' '.join(question['gold_contexts'][rawtitle]['gold_paras']).strip()
            golds[title] = para_combo
            if rawtitle == 'main':
                question['final_initial_context'] = title + ': ' + para_combo
        else:
            para_combo = ' '.join(question['gold_contexts'][rawtitle]['neg_paras'])
            negs[title] = para_combo
            
    finalstr = ''
    titles = list(golds.keys())
    random.shuffle(titles)
    for title in titles:
        finalstr += ' ' + title + ': ' + golds[title]
    if question['final_answer'] == '<No Answer>':
        num_rands = 1 if random.random() < 0.5 else 2
        for i, title in enumerate(negs):
            if i >= num_rands:
                break
            if random.random() < 0.5:
                finalstr += ' ' + title + ': ' + negs[title]
            else:
                finalstr = title + ': ' + negs[title] + ' ' + finalstr.strip()                
    question['final_gold_context'] = finalstr.strip()
    return


def build_golds(title_main, text_main, question):
    """ context = [{'passage':'title', 'text': 'sentence', 'indices':[start, end]}]
    
    adds 'gold_contexts' key to each question: {'title': {'gold_paras': ['Para 1', 'Para 2', ..]},  #title = 'main' or actual title
                                                         {'neg_paras': ['Para 1']}  # each title has only 'gold_paras' xor 'neg_paras' 
                                                }
    """
    titles = set([c['passage'] for c in question['context']])
    titledict = {}
    for title in titles:
        #realtitle = unescape(title).strip() if title != 'main' else unescape(title_main).strip()
        titledict[title] = {}
        sents = sorted([c for c in question['context'] if c['passage'] == title], key=lambda c: c['indices'])
        #titledict[title]['gold_sents'] = ' '.join([text_processing.format_sentence(s['text']) for s in sents])
        gold_paras = []
        if len(sents) >= 3: # if more than 2 sents in context, just concat the sents instead of matching a para for each of them
            para = ' '.join([text_processing.format_sentence(s['text']) for s in sents])
            gold_paras.append(para)
        elif title != 'main':
            article = get_article(title)
            article_no_tags = text_processing.remove_html_tags(article)
            for s in sents:
                start, end = s['indices']
                if start >= 0:
                    txt = s['text']
                else:
                    txt = s['text'][start*-1:]  # some indices are negative and don't match article text..
                para = get_para(article_no_tags, txt)
                if para == '#NF#':
                    print(f"ERROR: {title_main}: Unable to match CONTEXT: {s} in ARTICLE minus tags: {title}!")
                else:
                    if para not in gold_paras:
                        gold_paras.append(para)
        else:  # main
            for s in sents:
                start, end = s['indices']
                if start >= 0:
                    txt = s['text']
                else:
                    txt = s['text'][start*-1:]  # some indices are negative and don't match article text..
                para = get_para(text_main, txt, key='.', widen_by=75)
                if para == '#NF#':
                    print(f"ERROR: {title_main}: Unable to match CONTEXT: {s} in ARTICLE minus tags: {title}!")
                else:
                    if para not in gold_paras:
                        gold_paras.append(para)
        new_gold_paras = []
        for i, curr_para in enumerate(gold_paras):  # remove paras that are substrings of other paras
            substr = False
            for j, para in enumerate(gold_paras):
                if i != j and curr_para in para:
                    substr = True
                    break
            if not substr:
                new_gold_paras.append(curr_para)    

        titledict[title]['gold_paras'] = new_gold_paras

    exclude_titles = list(titledict.keys()) + [title_main]
    for i in range(2):
        title, para = get_random_para(exclude=exclude_titles)
        titledict[title] = {}
        titledict[title]['neg_paras'] = [para]
    question['gold_contexts'] = titledict
    consolidate_contexts(title_main, question)    
    return
    

    
def process_ds_split(split):
    """ Answer preprocessing based on https://github.com/jferguson144/IIRC-baseline/make_drop_style.py
    """
    
    max_ans_spans = -1
    print(f"Processing samples for {len(split)} docs...")
    for sample in tqdm(split):
        #print(f"Started {i}")
        title_main = sample['title']
        for j, question in enumerate(sample['questions']):
            #print(f"Started {j}")
            answer_info = question["answer"]
            a_type = answer_info["type"]
            if a_type == "span":
                answer_spans = [a["text"].strip() for a in answer_info["answer_spans"]]
                if len(answer_spans) == 1:
                    answer_spans = answer_spans[0]  # single span can be a string
                else:  # multiple spans: answer = list of all span permutations. In training only the first is used (EM) but in validation max EM over all permuations is used
                    if len(answer_spans) > max_ans_spans:
                        max_ans_spans = len(answer_spans)
                        #print(f'new max answer_span length: {max_ans_spans}')
                        #print(question)
                    if len(answer_spans) < 4:   # 4 = 28 permutations, 5 = 120 permutations ...
                        all_permutations = list(itertools.permutations(answer_spans))
                    else:
                        all_permutations = [answer_spans]
                    a_list = []
                    for spanlist in all_permutations:
                        a_list.append( ', '.join(spanlist) )
                    answer_spans = a_list
            elif a_type == "value":
              answer_spans = answer_info["answer_value"].strip()
            elif a_type == "binary":
              answer_spans = answer_info["answer_value"].strip()
            elif a_type == "none":
              answer_spans = '<No Answer>'
            question['final_answer'] = answer_spans            
            build_golds(title_main, sample['text'].strip(), question)
            #print(f"Ended {i} {j}")
    print("Finished!")
    return


def save_split(split, tokenizer, max_toks, outfile):
    out_od = []  # q->a
    out_ic = []  # q+initial gold para sents->a
    out_gc = []  # q+initial gold para sents+all other gold para sents->a
    num_over_maxtoks = 0
    mean_over = 0
    num_over_not_na = 0
    for sample in tqdm(split):
        for j, question in enumerate(sample['questions']):
            q = question['question']
            ans = question['final_answer']
            initial = question['final_initial_context']
            gold = question['final_gold_context']
            numtoks = len(tokenizer.tokenize(q + ' \\n ' + gold))
            if numtoks > max_toks:
                num_over_maxtoks += 1
                if ans != '<No Answer>':
                    num_over_not_na += 1
                mean_over += numtoks
            out_od.append( utils.create_uqa_example(q, '', ans))
            out_ic.append( utils.create_uqa_example(q, initial, ans))
            out_gc.append( utils.create_uqa_example(q, gold, ans))
    print(f"Num of samples: {len(out_od)}")
    print(f"Num q+gold context over {max_toks}: {num_over_maxtoks} avg:{mean_over/num_over_maxtoks}  num not na:{num_over_not_na}")
    out_dir = os.path.join(UQA_DIR, 'iirc_od_ans')
    utils.save_uqa(out_od, out_dir, outfile)
    out_dir = os.path.join(UQA_DIR, 'iirc_initial_context')
    utils.save_uqa(out_ic, out_dir, outfile)
    out_dir = os.path.join(UQA_DIR, 'iirc_gold_context')
    utils.save_uqa(out_gc, out_dir, outfile)
    return

random.seed(42)
process_ds_split(dev)
process_ds_split(train)
process_ds_split(test)


save_split(dev, tokenizer, max_toks, 'dev.tsv')   #1301
save_split(train, tokenizer, max_toks, 'train.tsv') #10839
save_split(test, tokenizer, max_toks, 'test.tsv')  #1301


#TODO (eventually) create IIRC_R = q+single gold+ retrieved paras->a
#TODO (eventually) create IIRC_RS = q+single gold+ retrieved sents->a






