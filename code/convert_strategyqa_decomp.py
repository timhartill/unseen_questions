#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:45:33 2021

@author: tim hartill

Convert StrategyQA into UnifiedQA-like format
Outputs two datasets:
    strategy_qa_facts_selfsvised: Individual paragraphs potentially used as evidence in reasoning for strategyQa questions
    strategy_qa: The actual strategyQA questions in MC format without any other context

StrategyQa from the paper:
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. 
Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. 
Transactions of the Association for Computational Linguistics, 9:346361.

Edit the directory and file names below before running...

Note: UQA_SQA_FACTS_DIR must end with _selfsvised - this signals the train/inference programs to treat this as a self supervised task

"""

import os
import json
import numpy as np
from collections import Counter
from utils import flatten, load_jsonl, create_uqa_example, load_model, run_model, get_single_result



#UQA_DIR = '/data/thar011/data/unifiedqa/'
#UQA_SQA_FACTS_DIR = 'strategy_qa_facts_selfsvised'
#UQA_SQA_Q_DIR = 'strategy_qa' 
SQA_DIR_IN = '/data/thar011/data/strategyqa/'
SQA_TRAIN_FILE = 'strategyqa_train.json'
SQA_PARA_FILE = 'strategyqa_train_paragraphs.json'

SQA_REPO_GENERATED_DIR = '/data/thar011/gitrepos/strategyqa/data/strategyqa/generated/'
SQA_REPO_GENERATED_DEV_FILE = 'transformer_qa_ORA-P_dev_no_placeholders.json'
SQA_REPO_GENERATED_TRAIN_FILE = 'transformer_qa_ORA-P_train_no_placeholders.json'
SQA_REPO_GENERATED_DEV_DECOMP_FILE = 'bart_decomp_dev_predictions.jsonl'


SQA_JSON_OUT_FILE = 'strategyqa_train_dev_consolidated.json'

model_name = "facebook/bart-large"
checkpoint = "/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-150000.pt"
special='Ġ'
indiv_digits = True  # individual digit tokenization
norm_numbers = True # normalize numbers. 




with open(os.path.join(SQA_DIR_IN, SQA_TRAIN_FILE),'r') as f:
    sqa_train = json.load(f)  #2290 questions

with open(os.path.join(SQA_DIR_IN, SQA_PARA_FILE),'r') as f:
    sqa_para = json.load(f)  #9251 paragraphs
    
with open(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_DEV_FILE),'r') as f:
    sqa_repo_dev = json.load(f)    #229
    
with open(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_TRAIN_FILE),'r') as f:
    sqa_repo_train = json.load(f)    #2061 + 229 = 2290
    
#sqa_repo_dev_decomp = load_jsonl(os.path.join(SQA_REPO_GENERATED_DIR, SQA_REPO_GENERATED_DEV_DECOMP_FILE))  #229 NOT USED

#questions = [s['question'] for s in sqa_repo_train] # 2061 There is a single duplicated question in train with different decomposition: Can you find Bob Marley's face in most smoke shops?
#count_dup = Counter(questions)
#for q in count_dup.keys():
#    if count_dup[q] > 1:
#        print(f'Duplicate question {count_dup[q]}: {q}')  # Can you find Bob Marley's face in most smoke shops?
#        dup = q
#for s in sqa_repo_train:
#    if s['question'] == dup:
#        print(s)  #'qid': '8a5edfb7385edb776926' and 'qid': 'cb900171acc9047115b4'
#for s in sqa_train:
#    if s['question'] == dup:
#        print(s)

# use question + 1st decomp step as key since qids don't match and question by itself is not quite unique..
sqa_repo_devtrain = {s['question']+s['decomposition'][0]:s for s in sqa_repo_train} # sqa qids in repo don't match qids in file from allenai sqa download page..
for qid in sqa_repo_devtrain:
    sqa_repo_devtrain[qid]['split'] = 'train'
for s in sqa_repo_dev:
    qid = s['question']+s['decomposition'][0]
    sqa_repo_devtrain[qid] = s
    sqa_repo_devtrain[qid]['split'] = 'dev'
print(f"Number of dev+train samples: {len(sqa_repo_devtrain)}")  #2290



def replace_chars(instr): 
    outstr = instr.replace("’", "'").replace("‘", "'")
    return outstr.replace('“', '"').replace('”','"').replace("\t", " ").replace("\n", "")


def para_keys_per_decomp(sqa_sample, verbose=False):
    """ Return cleaned up list of paragraph keys per decomp step each aggregated over multiple annotators
    Note: each sublist still contains meta--keys like "no_evidence" and "operation"
    Return format: [ [unique para keys for decomp step 1], [unique para keys for decomp step 2], [...] ]
    """
    decomp_evidence_para_keys = [[] for d in range(len(sqa_sample['decomposition']))] # [[], [], []]
    for j, ann in enumerate(sqa_sample['evidence']):
        for i, e in enumerate(ann):
            para_keys = flatten(e)
            decomp_evidence_para_keys[i].extend(para_keys)
            if verbose:
                print(f"Annotator:{j} Decomp:{i}: {sqa_sample['decomposition'][i]} {e} gold paras flattened:{para_keys}")
    decomp_evidence_para_keys = [list(set(pk)) for pk in decomp_evidence_para_keys]
    return decomp_evidence_para_keys


def retrieve_paras(sqa_sample):
    """ Return para corresponding to each key in list of lists of paragraph keys """
    para_list = []
    for i, para_keys in enumerate(sqa_sample['evidence_cleaned']):
        step_paras = []
        for pk in para_keys:
            para = sqa_para.get(pk)
            if para is not None:
                step_paras.append(para["content"])
        para_list.append(step_paras)
    return para_list


# strategyQA has no dev split so create one as 10% of train
# Update the paragraph dict with which qids use each paragraph
num_q = len(sqa_train)
dev_size = int(num_q*0.1)
for p in sqa_para:
    sqa_para[p]['splits_used'] = set()
    sqa_para[p]['qids_used'] = []
np.random.seed(42)
dev_indices = np.random.choice(num_q, dev_size, replace=False)
for i in range(num_q):
    if i in dev_indices:
        sqa_train[i]['split'] = 'dev'
    else:    
        sqa_train[i]['split'] = 'train'
    sqa_train[i]['evidence_flattened'] = set(flatten(sqa_train[i]['evidence'])) #flatten evidence while we are at it       
    for e in sqa_train[i]['evidence_flattened']:
        if sqa_para.get(e) is not None:
            sqa_para[e]['splits_used'].add(sqa_train[i]['split'])
            sqa_para[e]['qids_used'].append(sqa_train[i]['qid'])
    sqa_train[i]['evidence_cleaned'] = para_keys_per_decomp(sqa_train[i])
    sqa_train[i]['paragraphs'] = retrieve_paras(sqa_train[i])
    qid = sqa_train[i]['question'] + sqa_train[i]['decomposition'][0]
    sqa_train[i]['split_sqa_repo'] = sqa_repo_devtrain[qid]['split']
    sqa_train[i]['step_answers_sqa_repo'] = sqa_repo_devtrain[qid]['step_answers']
    sqa_train[i]['qid_sqa_repo'] =  sqa_repo_devtrain[qid]['qid']

traincount = 0
devcount = 0
bothcount = 0
nonecount = 0
qidcounts = []
for p in sqa_para:
    if sqa_para[p]['splits_used'] == set():
        nonecount += 1
    elif sqa_para[p]['splits_used'] == {'dev', 'train'}:
        traincount += 1
        devcount += 1
        bothcount += 1
    elif sqa_para[p]['splits_used'] == {'dev'}:
        devcount += 1
    else:
        traincount += 1
    num_ref = len(sqa_para[p]['qids_used'])
    qidcounts.append(num_ref)
qids_np = np.array(qidcounts)
print(f"qids: num:{qids_np.shape[0]}  mean:{qids_np.mean():.2f}  max:{qids_np.max():.2f}  min:{qids_np.min():.2f}")
# qids: num:9251  mean:1.11  max:11.00  min:1.00
print(f"Counts: Unused:{nonecount}  Both train+Dev:{bothcount}  Train Only:{traincount}  Dev Only:{devcount}")
# Counts: Unused:0  Both train+Dev:161  Train Only:8402  Dev Only incl dev+train:1010  - dev only 849, train only 8241

# Iteratively get answers to decomp step questions from UQA+TDND, record them and substitute them into remaining decomp steps
tokenizer, model = load_model(model_name, checkpoint)

for i in range(num_q):
    decomp_steps = []
    decomp_answers = []
    answer_scores = []
    decomp_answers_reversed = []
    answer_scores_reversed = []
    for j, ds in enumerate(sqa_train[i]['decomposition']):
        decomp_step = ds
        var_idx = decomp_step.find('#')
        while var_idx != -1:
            prior_step = decomp_step[var_idx+1]  # max 5 decomp steps so this works.
            if prior_step not in ['1','2','3','4','5','6','7','8','9']:
                prior_idx = 0    # at least one entry has #!
            else:
                prior_idx = int(prior_step)-1
            if answer_scores[prior_idx] >= answer_scores_reversed[prior_idx]:
                subst = decomp_answers[prior_idx]
            else:
                subst = decomp_answers_reversed[prior_idx]
            decomp_step = decomp_step.replace('#'+prior_step, subst)
            var_idx = decomp_step.find('#')    
        decomp_steps.append(decomp_step)
        
        input_string = create_uqa_example(decomp_step, ' '.join(sqa_train[i]['paragraphs'][j]))
        res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, special=special,
                        num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                        output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
        pred, score = get_single_result(res)  
        if pred == '<no answer>':
            score = -100.0
        decomp_answers.append(pred)
        answer_scores.append(score)
        input_string = create_uqa_example(decomp_step, ' '.join(sqa_train[i]['paragraphs'][j][::-1]))
        res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers, special=special,
                        num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                        output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
        pred, score = get_single_result(res)  
        if pred == '<no answer>':
            score = -100.1
        decomp_answers_reversed.append(pred)
        answer_scores_reversed.append(score)
        print(f"{i} {j} {decomp_step} ANS:{decomp_answers[-1]}({answer_scores[-1]}) REV:{pred}({score})")
    sqa_train[i]['decomposition_substituted'] = decomp_steps
    sqa_train[i]['step_answers'] = decomp_answers
    sqa_train[i]['step_answers_scores'] = answer_scores
    sqa_train[i]['step_answers_reversed'] = decomp_answers_reversed
    sqa_train[i]['step_answers_scores_reversed'] = answer_scores_reversed
 
print('Finished!')
print('Convert evidenced_flattened to list for saving to json...')
for s in sqa_train:
    s['evidence_flattened'] = list(s['evidence_flattened'])
outfile = os.path.join(SQA_DIR_IN, SQA_JSON_OUT_FILE)  
with open(outfile, 'w') as fp:
    json.dump(sqa_train, fp, indent=5)   
print(f"Saved to {outfile}")

# outputs:
#1 strategy_qa questions plus gold paras - to match Roberta*_ora-p
#2 strategy_qa questions plus gold paras reversed - combine with above after determining which has highest answer score
#3 sqa FINAL decomp step with substitutions plus gt y/n answers - to match Roberta*_ora_p-d
#4 decomp step questions with substitutions plus predicted answers - my best
#5 decomp step questions with substitutions plus predicted answers - sqa paper
#6 DECOMPOSER 1: strategy_qa question only as input with label = overall Y/N answer + decomp steps
#7 DECOMPOSER 2: strategy_qa question only as input with label = overall Y/N answer + decomp steps + step answers





"""
# Create MLM task:
paras_dev = []    # paras needed by dev only
paras_train = []  # include paras needed by both train and dev
for p in sqa_para:
    if sqa_para[p]['splits_used'] == {'dev'}:
        paras_dev.append( replace_chars(sqa_para[p]['content'] ) )
    else:
        paras_train.append( replace_chars(sqa_para[p]['content'] ) )
        

outdir = os.path.join(UQA_DIR, UQA_SQA_FACTS_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_train))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_dev))

# create strategyQA question task:
qa_dev = []
qa_train = []
for qa in sqa_train:
    question = f"{replace_chars(qa['question'])} \\n (A) yes (B) no"
    if qa['answer']:
        answer = 'yes'
    else:
        answer = 'no'
    sample = f"{question}\t{answer}"
    if qa['split'] == 'train':
        qa_train.append(sample)
    else:
        qa_dev.append(sample)
print(f"train count: {len(qa_train)}  dev count: {len(qa_dev)}")    # train count: 2061  dev count: 229

outdir = os.path.join(UQA_DIR, UQA_SQA_Q_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write('\n'.join(qa_train))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write('\n'.join(qa_dev))

"""




