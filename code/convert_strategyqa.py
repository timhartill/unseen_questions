#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:45:33 2021

@author: tim hartill

Convert StrategyQA into UnifiedQA-like format
Outputs three datasets:
    strategy_qa_facts_selfsvised: Individual paragraphs potentially used as evidence in reasoning for strategyQa questions
    strategy_qa: The actual strategyQA questions in MC format without any other context
    strategy_qa_yn: strategyQA questions in yn question format (no mc options)
    strategy_qa_

StrategyQa from the paper:
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. 
Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. 
Transactions of the Association for Computational Linguistics, 9:346361.

Edit the directory and file names below before running...

Note: UQA_SQA_FACTS_DIR must end with _selfsvised - this signals the train/inference programs to treat this as a self supervised task

NOTE: Established that BIG-bench split = SQA full train 2290 examples ie uses this as unseen "eval/test" as does UQAv2/PALM/Sanh/Self-consistency
Therefore output train file as test/dev split

"""

import os
import json
import numpy as np
import eval_metrics
import utils
import text_processing

UQA_DIR = eval_metrics.UQA_DIR
UQA_SQA_FACTS_DIR = 'strategy_qa_facts_selfsvised'
UQA_SQA_Q_DIR = 'strategy_qa' 
UQA_SQA_BIGBENCH = 'strategy_qa_bigbench'
SQA_DIR_IN = '/home/thar011/data/strategyqa/'
SQA_TRAIN_FILE = 'strategyqa_train.json'
SQA_PARA_FILE = 'strategyqa_train_paragraphs.json'
SQA_BIG_BENCHMARK_FILE = '/home/thar011/data/strategyqa/big_benchmark_sqa_task.json'

SQA_BASE = os.path.join(UQA_DIR, UQA_SQA_Q_DIR)
SQA_BASE_BB = os.path.join(UQA_DIR, UQA_SQA_BIGBENCH)

Q_PREFIX = 'Add Explanation: '
Q_FORCE_YN = 'Yes or no - '
DS_FORCE_YN_SUFFIX = '_yn'
OD_EXPL = '_od_expl'
EXPL_ANS = '_expl_ans'
MC_ANS = '_mc_ans'
OD_ANS = '_od_ans'
GOLD_ANS = '_gold_context_'

with open(os.path.join(SQA_DIR_IN, SQA_TRAIN_FILE),'r') as f:
    sqa_train = json.load(f)  #2290 questions

with open(os.path.join(SQA_DIR_IN, SQA_PARA_FILE),'r') as f:
    sqa_para = json.load(f)  #9251 paragraphs

#with open(os.path.join(SQA_DIR_IN, SQA_BIG_BENCHMARK_FILE),'r') as f:  # 2290 = len(sqa_bigb['examples'])
#    sqa_bigb = json.load(f)  # dict_keys(['canary', 'name', 'description', 'keywords', 'preferred_score', 'metrics', 'append_choices_to_input', 'examples'])



def save_single(split, outdir, ds_type, file):
    """ save a single dataset split """
    os.makedirs(outdir, exist_ok=True)
    out = [s[ds_type] for s in split]
    outfile = os.path.join(outdir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))    
    return


def save_datasets(dev, train):
    """ save uqa-formatted dataset """
    for ds_type in [OD_EXPL, EXPL_ANS, MC_ANS, OD_ANS]:
        outdir = SQA_BASE + ds_type
        print(f'Saving dataset to {outdir} ...')
        os.makedirs(outdir, exist_ok=True)
        save_single(dev, outdir, ds_type, 'dev.tsv')
        save_single(train, outdir, ds_type, 'train.tsv')
    print('Finished saving uqa-formatted explanation datasets!')
    return
    
# strategyQA has no dev split so create one as 10% of train  [update: no longer used, now using full train as eval]
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
    sqa_train[i]['evidence_flattened'] = set(utils.flatten(sqa_train[i]['evidence'])) #flatten evidence while we are at it
    for e in sqa_train[i]['evidence_flattened']:
        if sqa_para.get(e) is not None:
            sqa_para[e]['splits_used'].add(sqa_train[i]['split'])
            sqa_para[e]['qids_used'].append(sqa_train[i]['qid'])
            
    # to obtain gold set(s) of paras. Evidence = [[Annotator 1 set], [Annotator 2 set], [Annotator 3 set]]
    ann_contexts = [] # 1 entry per annotator of "titleABC: Text for titleABC. titleXYZ: Text for titleXYZ. ... "
    for annotator_set in sqa_train[i]['evidence']:
        ann_set_flattened = utils.unique_preserve_order(utils.flatten(annotator_set))
        #print(ann_set_flattened)
        curr_ann_context = ''
        for e in ann_set_flattened:
            if sqa_para.get(e) is not None:  # this drops out 'no_evidence', 'operation' etc..
                curr_ann_context += ' ' + sqa_para[e]['title'].strip() + ': ' + sqa_para[e]['content'][:600].strip()
                if curr_ann_context[-1] not in ['.', '?', '!', ':', ';']:
                    curr_ann_context += '.'
        ann_contexts.append(text_processing.format_sentence(curr_ann_context.strip()))
    sqa_train[i]['gold_contexts'] = ann_contexts                    
        

print(len([s for s in sqa_train if len(s['gold_contexts']) != 3])) # 10 - all these have len 4

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


# Create MLM task:
paras_dev = []    # paras needed by dev only
paras_train = []  # include paras needed by both train and dev
for p in sqa_para:
    if sqa_para[p]['splits_used'] == {'dev'}:
        paras_dev.append( text_processing.replace_chars(sqa_para[p]['content'] ) )
    else:
        paras_train.append( text_processing.replace_chars(sqa_para[p]['content'] ) )
        

# create self supervised task: [NOTE: 'dev in train' version created manually by adding dev.tsv to train.tsv...]
outdir = os.path.join(UQA_DIR, UQA_SQA_FACTS_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_train))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(paras_dev))

# create strategyQA question tasks:
for qa in sqa_train:
    #question = f"{text_processing.replace_chars(qa['question'])} \\n (A) yes (B) no"
    question_od = text_processing.format_sentence(qa['question'], endchar='?')
    f = ' '.join([text_processing.format_sentence(s) for s in qa['facts']])
    if qa['answer']:
        answer = 'yes'
    else:
        answer = 'no'
    #sample = f"{question}\t{answer}"
    qa[MC_ANS] = utils.create_uqa_example(question_od, "(A) yes (B) no", answer)
    qa[OD_ANS] = utils.create_uqa_example(question_od, None, answer)
    qa[OD_ANS+DS_FORCE_YN_SUFFIX] = utils.create_uqa_example(Q_FORCE_YN + question_od, None, answer)
    qa[EXPL_ANS] = utils.create_uqa_example(question_od, f, answer)
    qa[EXPL_ANS+DS_FORCE_YN_SUFFIX] = utils.create_uqa_example(Q_FORCE_YN + question_od, f, answer)
    qa[OD_EXPL] = utils.create_uqa_example(Q_PREFIX + question_od, None, f)
    for i in range(3):
        qa[GOLD_ANS+str(i)] = utils.create_uqa_example(question_od, qa['gold_contexts'][i], answer)
        qa[GOLD_ANS+str(i)+DS_FORCE_YN_SUFFIX] = utils.create_uqa_example(Q_FORCE_YN + question_od, qa['gold_contexts'][i], answer)

qa_train = []
qa_dev = []
qa_bigbench = []
for qa in sqa_train:
    if qa['split'] == 'train':
        qa_train.append(qa)
    else:
        qa_dev.append(qa)
    qa_bigbench.append(qa)  # "bigbench" eval set = full sqa train set
        
print(f"bigbench split count:{len(qa_bigbench)} train count: {len(qa_train)}  dev count: {len(qa_dev)}")    # bigbench split count:2290 train count: 2061  dev count: 229

save_datasets(qa_dev, qa_train)


###################################
# Save open domain "big-bench" ie full train set split as eval...
###################################

save_single(qa_bigbench, SQA_BASE_BB + OD_ANS, OD_ANS, 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + OD_ANS+DS_FORCE_YN_SUFFIX, OD_ANS+DS_FORCE_YN_SUFFIX, 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + EXPL_ANS, EXPL_ANS, 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + EXPL_ANS+DS_FORCE_YN_SUFFIX, EXPL_ANS+DS_FORCE_YN_SUFFIX, 'dev.tsv')

save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '0', GOLD_ANS + '0', 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '0' + DS_FORCE_YN_SUFFIX, GOLD_ANS + '0' + DS_FORCE_YN_SUFFIX, 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '1', GOLD_ANS + '1', 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '1' + DS_FORCE_YN_SUFFIX, GOLD_ANS + '1' + DS_FORCE_YN_SUFFIX, 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '2', GOLD_ANS + '2', 'dev.tsv')
save_single(qa_bigbench, SQA_BASE_BB + GOLD_ANS + '2' + DS_FORCE_YN_SUFFIX, GOLD_ANS + '2' + DS_FORCE_YN_SUFFIX, 'dev.tsv')




