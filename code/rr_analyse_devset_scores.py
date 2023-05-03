#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:25:19 2023

@author: tim hartill

analyse rr model scoring on dev set plus some ood gold contexts

"""
import os
import json
import numpy as np
from functools import partial


import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from rr_model_dataset import RREvalDataset, predict_simple, batch_collate
from rr_model_dataset import RRModel

from mdr_config import common_args


import utils
import eval_metrics

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]'] # Actually unused. Only using: [CLS] query [SEP] Rationale [SEP]



# already scored by rr model, several dev sets in 1 file:
infile_devsets = '/large_data/thar011/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/rr_dev_predictions.jsonl'

# these require scoring by rr model. all are positive only except for iirc <No Answer> samples:
infile_tobeprocessed = ['/data/thar011/data/unifiedqa/strategy_qa_bigbench_expl_ans/dev.tsv',
                        '/data/thar011/data/unifiedqa/strategy_qa_bigbench_gold_context_0/dev.tsv', 
                        '/data/thar011/data/unifiedqa/iirc_gold_context/dev.tsv',  # use <No Answer> as neg rationales
                        '/data/thar011/data/unifiedqa/musique_mu_dev_parasv2/dev.tsv',]

tobeprocessed_srcs = [os.path.split(os.path.split(f)[0])[1] for f in infile_tobeprocessed] # ['strategy_qa_bigbench_expl_ans', 'strategy_qa_bigbench_gold_context_0', 'iirc_gold_context', 'musique_mu_dev_parasv2']

outfile_afterprocessing = '/large_data/thar011/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/extra_dev_predictions.json'


parser = common_args()

args = parser.parse_args()

args.num_workers_dev = 10
args.model_name = 'google/electra-large-discriminator'
args.init_checkpoint = '/large_data/thar011/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt'
args.predict_batch_size = 100
args.output_dir = '/large_data/thar011/out/mdr/logs'


bert_config = AutoConfig.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)

model = RRModel(bert_config, args)  
if args.init_checkpoint != "":
    print(f"Loading checkpoint: {args.init_checkpoint }")
    model = utils.load_saved(model, args.init_checkpoint )

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
model.to(device)
model.eval()

collate_fc = partial(batch_collate, pad_id=tokenizer.pad_token_id)  




rr_dev_results = utils.load_jsonl(infile_devsets)  # 30480  dict_keys(['question', 'context', 'ans', 'para_gold', 'para_pred', 'para_score', 'para_thresh', 'para_pred_dict', 'src', 'pos', '_id', 'para_acc'])

devsets = list(set([s['src'] for s in rr_dev_results]))
print(devsets)  # ['fever', 'qasc', 'hover_iter', 'creak', 'hpqa', 'worldtree']

rr_dev_results_dict = {}
for s in rr_dev_results:
    src = s['src']
    if rr_dev_results_dict.get(src) is None:
        rr_dev_results_dict[src] = []
    rr_dev_results_dict[src].append(s)


def print_dset_results(devset):
    """ print summary stats from single dev dataset
    """
    print('#'*20)
    print(f"Dev set: {devset[0]['src']}  Total N={len(devset)}")
    utils.create_grouped_metrics(None, devset, group_key='pos', metric_keys = ['para_acc', 'para_score'], verbose=False)    
    return

for ds in rr_dev_results_dict:
    print_dset_results(rr_dev_results_dict[ds])
  

#####
# Run prediction for additional datasets...
####

extra_dev_results = {}
for i, dset in enumerate(infile_tobeprocessed):
    ds = tobeprocessed_srcs[i]
    print(f"Running inference for {ds}...")
    extra_dev_results[ds] = utils.load_uqa_supervised(dset, ans_lower=False, return_parsed=True)

    # run rr model preds, record scores for this dataset:
    eval_dataset = RREvalDataset(args, tokenizer, extra_dev_results[ds], score_llm=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
    #TJH batch = next(iter(eval_dataloader))
    scores, qids = predict_simple(eval_dataloader, model)
    for i, s in enumerate(scores):
        extra_dev_results[ds][i]['para_score'] = s
print("finished inference!")

neg_match = '<No Answer>'
for ds in extra_dev_results:
    for s in extra_dev_results[ds]:
        s['src'] = ds
        if s['answer'] == neg_match:
            s['pos'] = False
        else:
            s['pos'] = True

json.dump(extra_dev_results, open(outfile_afterprocessing, 'w'))

for ds in extra_dev_results:
    print_dset_results(extra_dev_results[ds])




"""

DEV SET BREAKDOWN:

------------------------------------------------

Dev set: creak  Total N=2742
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.8475565280816922  n=1371  max: 1.000000  min: 0.000000
para_score: Mean:0.8445783553377386  n=1371  max: 0.999935  min: 0.000024
------------------------------------------------
pos: False
para_acc: Mean:0.9883296863603209  n=1371  max: 1.000000  min: 0.000000
para_score: Mean:0.011415647228488689  n=1371  max: 0.999875  min: 0.000024
------------------------------------------------
####################
Dev set: hpqa  Total N=4944
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.9381067961165048  n=2472  max: 1.000000  min: 0.000000
para_score: Mean:0.9348753395279155  n=2472  max: 0.999958  min: 0.000032
------------------------------------------------
pos: False
para_acc: Mean:0.9785598705501618  n=2472  max: 1.000000  min: 0.000000
para_score: Mean:0.021569474210684177  n=2472  max: 0.999781  min: 0.000023
------------------------------------------------
####################
Dev set: fever  Total N=12000
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.876  n=6000  max: 1.000000  min: 0.000000
para_score: Mean:0.8740408929146427  n=6000  max: 0.999957  min: 0.000025
------------------------------------------------
pos: False
para_acc: Mean:0.9795  n=6000  max: 1.000000  min: 0.000000
para_score: Mean:0.021270688445922437  n=6000  max: 0.999932  min: 0.000023
------------------------------------------------
####################
Dev set: qasc  Total N=1802
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.8956714761376249  n=901  max: 1.000000  min: 0.000000
para_score: Mean:0.8881645269739904  n=901  max: 0.999938  min: 0.000037
------------------------------------------------
pos: False
para_acc: Mean:0.9689234184239733  n=901  max: 1.000000  min: 0.000000
para_score: Mean:0.035545195610026795  n=901  max: 0.999746  min: 0.000024
------------------------------------------------
####################
Dev set: worldtree  Total N=992
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.9919354838709677  n=496  max: 1.000000  min: 0.000000
para_score: Mean:0.9915144890659165  n=496  max: 0.999957  min: 0.000130
------------------------------------------------
pos: False
para_acc: Mean:0.9979838709677419  n=496  max: 1.000000  min: 0.000000
para_score: Mean:0.003911903814802768  n=496  max: 0.996793  min: 0.000024
------------------------------------------------
####################
Dev set: hover_iter  Total N=8000
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_acc: Mean:0.92775  n=4000  max: 1.000000  min: 0.000000
para_score: Mean:0.9275285202001801  n=4000  max: 0.999957  min: 0.000029
------------------------------------------------
pos: False
para_acc: Mean:0.92775  n=4000  max: 1.000000  min: 0.000000
para_score: Mean:0.07250395085951913  n=4000  max: 0.999956  min: 0.000026


EXTRAS BREAKDOWN:

####################
Dev set: strategy_qa_bigbench_expl_ans  Total N=2290
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_score: Mean:0.3832934581333714  n=2290  max: 0.999623  min: 0.000134
------------------------------------------------
####################
Dev set: strategy_qa_bigbench_gold_context_0  Total N=2290
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_score: Mean:0.14229203980694644  n=2290  max: 0.999390  min: 0.000131
------------------------------------------------
####################
Dev set: iirc_gold_context  Total N=1301
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_score: Mean:0.6293355908538907  n=954  max: 0.999693  min: 0.000135
------------------------------------------------
pos: False
para_score: Mean:0.13009410434645252  n=347  max: 0.994893  min: 0.000143
------------------------------------------------
####################
Dev set: musique_mu_dev_parasv2  Total N=2417
------------------------------------------------
Metrics grouped by: pos
------------------------------------------------
pos: True
para_score: Mean:0.35656028985864585  n=2417  max: 0.999686  min: 0.000137
------------------------------------------------

"""

