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
    - q[+mc][+LLM expls that score over a threshold else Iterator context] -> a     (LLM if good else Iterator)
    - q[+mc][+Iterator contexts that score over a threshold else LLM context] -> a  (Iterator if good else LLM)
    - q[+mc][+max rr score of LLM expls or Iterator context] -> a                   (max rr of LLM / Iterator)
    - q[+mc][+LLM expls that score over a threshold][+Iterator contexts with ev score over a threshold] -> a (could have one or both of LLM & Iterator expls)

"""

import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import copy
from datetime import date
from functools import partial
from tqdm import tqdm

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from rr_model_dataset import RREvalDataset, predict_simple, batch_collate
from rr_model_dataset import RRModel

from mdr_config import common_args
import utils
from utils import move_to_cuda, load_saved
import eval_metrics
import mdr_searchers

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]'] # Actually unused. Only using: [CLS] query [SEP] Rationale [SEP]

UQA_DIR = eval_metrics.UQA_DIR


    


if __name__ == "__main__":
    parser = common_args()

    parser.add_argument("--llm_file", default="", type=str, help="Full path of tsv file output from LLM prompting. Generally dataset_llm_expl_ans/dev.tsv")
    parser.add_argument("--iter_file", default="", type=str, help="Full path of jsonl file output from Iterator.")
    parser.add_argument("--base_dataset", default="", type=str, help="Input and output base dataset name.")
    parser.add_argument("--tok_model_name", default="roberta-base", type=str, help="Tokenizer to check iterator context length with. Default roberta-base matches mdr_searchers usage.")
    parser.add_argument('--ctx_topk_paras', type=int, default=-1, help="Number of paras to include in final Iterator context build. -1 means include all.")
    parser.add_argument('--ctx_gold_sents_only', action="store_true", help="Iterator: If set only sentences from s2 included in final context. Otherwise 1 sentence before/after each s2 sent is included.")
 
    args = parser.parse_args()

    """ Test options
    args.prefix = 'LLM_ITER_MERGE_TEST0'
    args.num_workers_dev = 10
    args.model_name = 'google/electra-large-discriminator'
    args.init_checkpoint = '/large_data/thar011/out/mdr/logs/RR_test4_mcstrip0.5_notsinglepossplit_withsharednormal-04-11-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt'
    args.predict_batch_size = 100
    args.llm_file = '/data/thar011/data/unifiedqa/commonsenseqa_llm_expl/dev.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_csqa_test66_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.output_dir = '/large_data/thar011/out/mdr/logs'
    args.base_dataset = 'commonsenseqa'
    """

    split = os.path.split(args.llm_file)[-1][:-4]     
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{args.base_dataset}-{split}-{date_curr}-RR_ITER_MERGE"
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

    model = RRModel(bert_config, args)  
    if args.init_checkpoint != "":
        logger.info(f"Loading checkpoint: {args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    model.eval()


    collate_fc = partial(batch_collate, pad_id=tokenizer.pad_token_id)  

    ret_tokenizer = AutoTokenizer.from_pretrained(args.tok_model_name)

    samples = utils.load_jsonl(args.iter_file)
    for sample in samples:
        mdr_searchers.get_best_hop(sample)  # obtain best data from best hop by s2_ev_score
    mdr_searchers.build_context(args, None, samples, tokenizer=ret_tokenizer, output_tsv=False) # add 'final_context_fitted' key with context
    # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'final_context_fitted'])
    # samples[0]['s2_best_preds']['s2ev_score'] contains evidence score
    # samples[0]['final_context_fitted'] contains iterator generated context using same args as mdr_searchers originally did

    # load LLM expl ans tsv  dict_keys(['question', 'answer', 'q_only', 'mc_options', 'context'])
    samples_llm = utils.load_uqa_supervised(args.llm_file, ans_lower=False, return_parsed=True)  # for IIRC context has form "Init para title: init para. Further Explanation: generated rationale."
    
    iter_dict = {s['question'].rstrip().rstrip('?!:. ').lstrip(): s for s in samples}
    for i, s in enumerate(samples_llm):
        q = s['q_only'].rstrip().rstrip('?!:. ').lstrip()
        iter_sample = iter_dict.get(q)
        if iter_sample is None:
            logger.info(f"ERROR: Pos idx:{i}  llm q:{q}: Unable to match in iter_dict.")
        s['iter_context'] = iter_sample['final_context_fitted']  # for iirc has form init para title: init para. retrieved title a: retrived para A test. ...
        s['iter_context_ev_score'] = iter_sample['s2_best_preds']['s2ev_score']
        
    
    # run rr model preds, record scores for LLm expls
    eval_dataset = RREvalDataset(args, tokenizer, samples_llm, score_llm=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
    #TJH batch = next(iter(eval_dataloader))
    scores, qids = predict_simple(eval_dataloader, model)
    for i, s in enumerate(scores):
        samples_llm[i]['llm_rr_score'] = s
    
    
    # run rr model preds, record scores for ITER expls
    # run rr model preds, record scores for LLm expls
    eval_dataset = RREvalDataset(args, tokenizer, samples_llm, score_llm=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
    #TJH batch = next(iter(eval_dataloader))
    scores, qids = predict_simple(eval_dataloader, model)
    for i, s in enumerate(scores):
        samples_llm[i]['iter_context_rr_score'] = s
        
    utils.saveas_jsonl(samples_llm, os.path.join(args.output_dir, f'samples_llm_iter_scored.jsonl'))
    

    # output combined file
    # output eval tsv files