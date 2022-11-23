#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:31:59 2022

@author: tim hartill

See whether Mu dev answers are (spuriously) memorised from mu training

"""

import os
import json

import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR
LDATA = eval_metrics.LDATA

base_ratd = 'out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m'

mu_dev_tsv = 'musique_mu_dev_odv2_fullwiki_bs150/dev.tsv'
mu_train_tsv = 'musique_qa_fullwiki_bs60/train.tsv'

mu_dev_gold_context = 'dev_musique_mu_dev_parasv2_predictions.json'
mu_dev_ret_context = 'dev_musique_mu_dev_odv2_fullwiki_bs150_predictions.json'

outfile = os.path.join(LDATA, 'out/mdr/logs/eval_outputs/s11/musique_memorised.txt')


mu_dev_g_preds = json.load(open(os.path.join(LDATA, base_ratd, mu_dev_gold_context))) #2417
mu_dev_r_preds = json.load(open(os.path.join(LDATA, base_ratd, mu_dev_ret_context))) #2417

mu_dev_g_preds = [m.strip().lower() for m in mu_dev_g_preds]
mu_dev_r_preds = [m.strip().lower() for m in mu_dev_r_preds]


mu_dev_g_tsv = utils.load_uqa_supervised(os.path.join(UQA_DIR, mu_dev_tsv), ans_lower=False , return_parsed=True)  #2417
mu_train_g_tsv = utils.load_uqa_supervised(os.path.join(UQA_DIR, mu_train_tsv), ans_lower=False , return_parsed=True)  #19556

mu_dev_g = [s['answer'] for s in mu_dev_g_tsv]
mu_train_g = [s['answer'].strip().lower() for s in mu_train_g_tsv]
mu_train_g_set = set(mu_train_g)
mu_train_dict = {}
for s in mu_train_g_tsv:
    ans = s['answer'].strip().lower()
    if mu_train_dict.get(ans) is None:
        mu_train_dict[ans] = []
    mu_train_dict[ans].append(s['q_only'])

mu_dev_g_ind = [s.lower() for s in mu_dev_g if type(s)==str] 
for sl in mu_dev_g:
    if type(sl)!=str:
        mu_dev_g_ind.extend([s.strip().lower() for s in sl])


dev_in_train = set([d for d in mu_dev_g_ind if d in mu_train_g_set]) #99 total -> 15 unique

F1 = eval_metrics.F1()
f1_mu_dev_rc = F1.compute_metric(mu_dev_r_preds, mu_dev_g)  #22.21

f1_0_idxs = []
train_exmatch_idxs = []
for i, fscore in enumerate(F1.all_scores):
    if fscore == 0:
        f1_0_idxs.append(i)
        if mu_dev_r_preds[i] in mu_train_g_set:
            train_exmatch_idxs.append(i)
print(f"Total mu dev: {len(mu_dev_r_preds)}  Total with f1=0: {len(f1_0_idxs)}  Total with EM in Mu train answers: {len(train_exmatch_idxs)}")
# Total mu dev: 2417  Total with f1=0: 1670  Total with EM in Mu train answers: 716

outlist = []
for i, idx in enumerate(train_exmatch_idxs):
    out = f"MU DEV Question: {mu_dev_g_tsv[idx]['q_only']} MU Dev ANS: {mu_dev_g_tsv[idx]['answer']} PRED/MU TRAIN ANS:{mu_dev_r_preds[idx]}  MU TRAIN QUESTIONS W/PRED ANS: {mu_train_dict[mu_dev_r_preds[idx]]}"
    outlist.append(out + '\n')
    #print(f"MU DEV Question: {mu_dev_g_tsv[idx]['q_only']} MU Dev ANS: {mu_dev_g_tsv[idx]['answer']} PRED/MU TRAIN ANS:{mu_dev_r_preds[idx]}  MU TRAIN QUESTIONS W/PRED ANS: {mu_train_dict[mu_dev_r_preds[idx]]}")
    #if i > 10:
    #    break

with open(outfile, 'w') as f:
    f.write(''.join(outlist))
print(f'Memorised samples written to: {outfile}')



