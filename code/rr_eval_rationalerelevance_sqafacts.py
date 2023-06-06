#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:03:44 2023

@author: tim hartill

Evaluate RR, Iterator S1 and Iterator S2 models on StrategyQA gold facts.


"""

import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import copy
import random
import numpy as np
from datetime import date
from functools import partial
from tqdm import tqdm

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from rr_model_dataset import RREvalDataset, predict_simple, batch_collate
from rr_model_dataset import RRModel, S1S2EvalDataset

from mdr_config import common_args
import utils
from utils import move_to_cuda, load_saved
import eval_metrics

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]'] # Only using: [CLS] query [SEP] Rationale [SEP] for RR but use other special tokens in S1/S2 models..

UQA_DIR = eval_metrics.UQA_DIR


if __name__ == "__main__":
    parser = common_args() 
    args = parser.parse_args()

    """ Test options
    args.prefix = "RATRELEVANCE_TEST"
    args.num_workers_dev = 10
    args.model_name = 'google/electra-large-discriminator'
    args.init_checkpoint = '/large_data/thar011/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt'
    args.predict_batch_size = 100
    args.output_dir = '/large_data/thar011/out/mdr/logs'
    args.model_type = 'rr'  # 's1'
    args.predict_file = '/data/thar011/data/unifiedqa/strategy_qa_bigbench_expl_ans/dev.tsv'
    
    args.prefix = "RATRELEVANCE_TEST"
    args.num_workers_dev = 10
    args.model_name = 'google/electra-large-discriminator'
    args.init_checkpoint = '/large_data/thar011/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
    args.predict_batch_size = 100
    args.output_dir = '/large_data/thar011/out/mdr/logs'
    args.model_type = 's1'  # 
    args.predict_file = '/data/thar011/data/unifiedqa/strategy_qa_bigbench_expl_ans/dev.tsv'
    
    args.prefix = "RATRELEVANCE_TEST"
    args.num_workers_dev = 10
    args.model_name = 'google/electra-large-discriminator'
    args.init_checkpoint = '/large_data/thar011/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
    args.predict_batch_size = 100
    args.output_dir = '/large_data/thar011/out/mdr/logs'
    args.model_type = 's2'  # 
    args.predict_file = '/data/thar011/data/unifiedqa/strategy_qa_bigbench_expl_ans/dev.tsv'
    
        
    """

    model_type = args.model_type.strip().lower()
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{model_type}-{date_curr}-SQA_RAT_RELEVANCE"
    args.output_dir = os.path.join(args.output_dir, model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)    
    logger.info(f"Output log eval_log.txt will be written to: {args.output_dir}")
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))
    
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)

    if model_type in ['rr', 's1', 's2']:
        logger.info(f"Loading {model_type} Model type...")
        model = RRModel(bert_config, args)  #Note: can use RRModel for EVAL only for s1/s2 models para/evidene scoring (not sentence level) since they are all Electra
        collate_fc = partial(batch_collate, pad_id=tokenizer.pad_token_id)
    else:
        logger.info("MODEL TYPE NOT IMPLEMENTED!")
        
    if args.init_checkpoint != "":
        logger.info(f"Loading checkpoint: {args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    model.eval()

    random.seed(42)

    samples_orig = utils.load_uqa_supervised(args.predict_file, ans_lower=False, return_parsed=True)
    orig_idxs = [i for i in range(len(samples_orig))]
    
    samples = []  # list of dict_keys(['question', 'answer', 'q_only', 'mc_options', 'context'])
    for i, s in enumerate(samples_orig): #reformat to input format of ranker models using one mc option per sample
        s['_id'] = str(i)
        q = s['q_only'].strip()
        a = s['context']
        s['targets'] = {s['context']: 1}
        for j in range(4):  # make into 5-way multichoice by adding 4 irrelevant contexts
            idx = random.choice(orig_idxs)
            while idx == i:
                idx = random.choice(orig_idxs)
            s['targets'][ samples_orig[idx]['context'] ] = 0
        
        for j, option in enumerate(s['targets']):
            s_opt = {'question': q, 'answer': a, 'q_only': q, 'mc_options': '', 'context': option, '_id': str(i)+'__'+str(j)}
            samples.append(s_opt)
            
    logger.info(f"StrategyQA samples: {len(samples_orig)} Number of MC Options: {len(samples)}")
    # run rr model preds, record scores for truthfulqa
    if model_type == 'rr':
        eval_dataset = RREvalDataset(args, tokenizer, samples) 
    elif model_type == 's1':
        eval_dataset = S1S2EvalDataset(args, tokenizer, samples, model_type='s1')
    elif model_type == 's2':
        eval_dataset = S1S2EvalDataset(args, tokenizer, samples, model_type='s2')
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
    #TJH batch = next(iter(eval_dataloader))
    scores, qids = predict_simple(eval_dataloader, model)
    for i, s in enumerate(scores):
        samples[i]['score'] = s
    
    utils.saveas_jsonl(samples, os.path.join(args.output_dir, 'samples_sqa_rat_relevance_options_scored.jsonl'))
    
    # EVAL MC
    for s in samples:
        curr_q_id = int(s['_id'][:s['_id'].find('__')])
        if samples_orig[curr_q_id].get('scores') is None:
            samples_orig[curr_q_id]['scores'] = []
        samples_orig[curr_q_id]['scores'].append(s['score'])

    right = 0
    wrong = 0
    for s in samples_orig:
        pred_option_idx = int(np.argmax(s['scores']))
        s['pred_option_idx'] = pred_option_idx
        for i, option in enumerate(s['targets']):
            if s['targets'][option] == 1:
                s['gold_answer'] = option
                s['gold_option_idx'] = i
                correct = pred_option_idx == i
                s['correct'] = correct
                if correct:
                    right += 1
                else:
                    wrong += 1
                break
    logger.info(f"Accuracy: Right: {right}  Wrong: {wrong} ACC: {right/len(samples_orig)}  check:{(right+wrong) == len(samples_orig)}")
    utils.saveas_jsonl(samples_orig, os.path.join(args.output_dir, 'samples_sqa_rat_relevance_aggregated_scored.jsonl'))
    
    