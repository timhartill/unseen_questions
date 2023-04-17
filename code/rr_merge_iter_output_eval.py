#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:03:44 2023

@author: tim hartill

Create eval datasets combining LLM and Iterator explanations.

1. Load iterator-generated jsonl file, identifies best hop and extracts context and s2 ev score from that hop.
2. Load LLM tsv file containing rationales for each sample 
2.1 Run the samples through the RR model to score each rationale
3. Merge the LLM and iterator files
4. Outputs various versions in tsv format ready for inclusion in QA Model eval:
    - q[+mc][+LLM expls that score over a threshold] -> a                           (LLM if good else nothing)
    - q[+mc][+Iterator contexts with ev score over a threshold] -> a                (Iterator if good else nothing)
    - q[+mc][+LLM expls that score over a threshold else Iterator context] -> a     (ie LLM if good else Iterator)
    - q[+mc][+Iterator contexts that score over a threshold else LLM context] -> a  (ie Iterator if good else LLM)
    - q[+mc][+LLM expls that score over a threshold][+Iterator contexts with ev score over a threshold] -> a (could have one or both of LLM & Iterator expls)

"""

import argparse
import logging
import os
import json
import copy
from datetime import date
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


import utils
import eval_metrics
import mdr_searchers

UQA_DIR = eval_metrics.UQA_DIR



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", default="", type=str, help="start of log file dir name.")
    parser.add_argument("--llm_file", default="", type=str, help="Full path of tsv file output from LLM prompting. Generally dataset_llm_expl_ans/dev.tsv")
    parser.add_argument("--iter_file", default="", type=str, help="Full path of jsonl file output from Iterator.")
    parser.add_argument("--output_dir", default="", type=str, help="Output directory for log, combined jsonl file.")
    parser.add_argument("--base_dataset", default="", type=str, help="Input and output base dataset name.")
    parser.add_argument("--model_name", default="roberta-base", type=str, help="Tokenizer to check iterator context length with. Default roberta-base matches mdr_searchers usage.")
    parser.add_argument('--ctx_topk_paras', type=int, default=-1, help="Number of paras to include in final Iterator context build. -1 means include all.")
    parser.add_argument('--ctx_gold_sents_only', action="store_true", help="Iterator: If set only sentences from s2 included in final context. Otherwise 1 sentence before/after each s2 sent is included.")
 
    args = parser.parse_args()

    """ Test options
    args.prefix = 'LLM_ITER_MERGE_'
    args.llm_file = '/large_data/thar011/out/mdr/logs/LLM_TEST7_MCONLY_csqa_dev_all_on_finalv4-01-13-2023-LLM-bigscience-bloom-/llm_samples_with_context.json'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_csqa_test66_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.output_dir = '/large_data/thar011/out/mdr/logs'
    args.base_dataset = 'commonsenseqa'
    """
    
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{args.base_dataset}-{date_curr}-RR_ITER_MERGE"
    args.output_dir = os.path.join(args.output_dir, model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)    
    logger.info(f"Output log eval_log.txt will be written to: {args.output_dir}")
    n_gpu = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    samples = utils.load_jsonl(args.iter_file)
    for sample in samples:
        mdr_searchers.get_best_hop(sample)  # obtain best data from best hop by s2_ev_score
    mdr_searchers.build_context(args, None, samples, tokenizer=tokenizer, output_tsv=False) # add 'final_context_fitted' key with context
    # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'final_context_fitted'])
    # samples[0]['s2_best_preds']['s2ev_score'] contains evidence score
    # samples[0]['final_context_fitted'] contains iterator generated context using same args as mdr_searchers originally did

    # load LLM expl ans tsv
    
    # run rr model preds, record score


    # output combined file
    # output eval tsv files