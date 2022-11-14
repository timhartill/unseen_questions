#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:13:56 2022

@author: tim hartill

Compare IIRC-G vs IIRC-R predictions vs gold by answer type 

"""

import os
import json

import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR
LDATA = eval_metrics.LDATA
ans_type_dir = os.path.join(UQA_DIR, 'answer_types')

file_ans_type = os.path.join(ans_type_dir, 'anstypes_iirc_test.jsonl' )
file_iirc_g = os.path.join(UQA_DIR, 'iirc_gold_context/test.tsv')
file_iirc_r = os.path.join(UQA_DIR, 'iirc_initial_context_fullwiki_bs150/test.tsv')
file_iirc_g_preds = os.path.join(LDATA, 'out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/test_iirc_gold_context_predictions.json')
file_iirc_r_preds = os.path.join(LDATA, 'out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/test_iirc_initial_context_fullwiki_bs150_predictions.json')

ans_types = utils.load_jsonl(file_ans_type)         #1301
iirc_g = utils.load_uqa_supervised(file_iirc_g, ans_lower=False , return_parsed=True)     #1301
iirc_r = utils.load_uqa_supervised(file_iirc_r, ans_lower=False , return_parsed=True)     #1301
iirc_g_preds = json.load(open(file_iirc_g_preds))   #1301
iirc_r_preds = json.load(open(file_iirc_r_preds))   #1301

combo = [{'TYPE': at['ans_type'], 'Q': g['q_only'], 'A': g['answer'], 'GP': gp, 'RP':rp, 'GC': g['context'], 'RC':r['context']} for (at, g, gp, rp, r) in zip(ans_types, iirc_g, iirc_g_preds, iirc_r_preds, iirc_r)]


