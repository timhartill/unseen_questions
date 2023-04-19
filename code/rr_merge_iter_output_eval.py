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
from rr_model_dataset import RRModel

from mdr_config import common_args
import utils
from utils import move_to_cuda, load_saved
import eval_metrics
import mdr_searchers

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]'] # Actually unused. Only using: [CLS] query [SEP] Rationale [SEP]

UQA_DIR = eval_metrics.UQA_DIR


def answer_in_expl(samples):
    """ Very Rough analysis of scoring by looking at whether the answer is in the expl
    CSQA: 
    LLM Mean:0.21302189518758627 (1221) Answer IN LLM Expl:0.2573089889034759 (584)  Answer NOT in LLM Expl:0.17241959890802655 (637)
    LLM Max:0.9966553449630737  Min:0.00023650623916182667
    ITER rr Mean:0.44225340958100795  ITER using rr_score: Answer IN Iter:0.525674262189266 (301)  Answer NOT in Iter:0.41496028280374087 (920)
    ITER rr Max:0.998989999294281  Min:0.0006781742558814585
    ITER ev Mean:0.0960430557541431  ITER using ev_score: Answer IN Iter:0.11044974461626385 (301)  Answer NOT in Iter:0.09132956298512315 (920)
    ITER ev Max:0.9911504983901978  Min:0.0005898156086914241
    
    SQA:
    LLM Mean:0.17280123899063612 (2290) Answer IN LLM Expl:0.17893527346471008 (491)  Answer NOT in LLM Expl:0.1711270806099967 (1799)
    LLM Max:0.9966425895690918  Min:0.00020381664216984063
    ITER rr Mean:0.7680089961612336  ITER using rr_score: Answer IN Iter:0.7686055063116949 (1162)  Answer NOT in Iter:0.7673945060948898 (1128)
    ITER rr Max:0.9994673132896423  Min:0.0032099445816129446
    ITER ev Mean:0.4878030581909179  ITER using ev_score: Answer IN Iter:0.483043173098768 (1162)  Answer NOT in Iter:0.4927064149968383 (1128)
    ITER ev Max:0.9989007711410522  Min:0.000617390382103622    
    
    ARCDA:
    LLM Mean:0.4057676666909243 (1397) Answer IN LLM Expl:0.4700740239746016 (297)  Answer NOT in LLM Expl:0.3884049502243314 (1100)
    LLM Max:0.9982719421386719  Min:0.00023336267622653395
    ITER rr Mean:0.6109208474389052  ITER using rr_score: Answer IN Iter:0.6736969843242203 (359)  Answer NOT in Iter:0.58920925481672 (1038)
    ITER rr Max:0.9992144107818604  Min:0.0027664205990731716
    ITER ev Mean:0.5978934605243488  ITER using ev_score: Answer IN Iter:0.7175364601249096 (359)  Answer NOT in Iter:0.5565140415873533 (1038)
    ITER ev Max:0.9990513920783997  Min:0.0005922522977925837    

    IIRC:
    LLM Mean:0.591702777727683 (1301) Answer IN LLM Expl:0.6024548986217041 (303)  Answer NOT in LLM Expl:0.5884383562538469 (998)
    LLM Max:0.9992634654045105  Min:0.00034453609259799123
    ITER rr Mean:0.9732178821329379  ITER using rr_score: Answer IN Iter:0.9781740914367026 (528)  Answer NOT in Iter:0.9698325283006123 (773)
    ITER rr Max:0.999610960483551  Min:0.05248409882187843
    ITER ev Mean:0.4800406305406541  ITER using ev_score: Answer IN Iter:0.5161774624279077 (528)  Answer NOT in Iter:0.4553572576603567 (773)
    ITER ev Max:0.9988969564437866  Min:0.0005918851820752025
        
    MU_DEV:
    LLM Mean:0.03587697159156379 (2417) Answer IN LLM Expl:0.11416676165259933 (117)  Answer NOT in LLM Expl:0.031894404010198066 (2300)
    LLM Max:0.9960206151008606  Min:0.00022117119806353003
    ITER rr Mean:0.9571252590444912  ITER using rr_score: Answer IN Iter:0.9804996452769454 (686)  Answer NOT in Iter:0.9478619263145872 (1731)
    ITER rr Max:0.9995593428611755  Min:0.00528826704248786
    ITER ev Mean:0.49310641057146704  ITER using ev_score: Answer IN Iter:0.7122762090550788 (686)  Answer NOT in Iter:0.40624882434399295 (1731)
    ITER ev Max:0.9991397857666016  Min:0.0005928017781116068        

    """
    
    scores_ans_in_llm = [s['llm_rr_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) in s['context']]
    scores_ans_not_in_llm = [s['llm_rr_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) not in s['context']]
    print(f"LLM Mean:{np.mean(scores_ans_in_llm+scores_ans_not_in_llm)} ({len(samples)}) Answer IN LLM Expl:{np.mean(scores_ans_in_llm)} ({len(scores_ans_in_llm)})  Answer NOT in LLM Expl:{np.mean(scores_ans_not_in_llm)} ({len(scores_ans_not_in_llm)})")
    print(f"LLM Max:{np.max(scores_ans_in_llm+scores_ans_not_in_llm)}  Min:{np.min(scores_ans_in_llm+scores_ans_not_in_llm)}")

    scores_ans_in_iter = [s['iter_context_rr_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) in s['iter_context']]
    scores_ans_not_in_iter = [s['iter_context_rr_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) not in s['iter_context']]
    print(f"ITER rr Mean:{np.mean(scores_ans_in_iter+scores_ans_not_in_iter)}  ITER using rr_score: Answer IN Iter:{np.mean(scores_ans_in_iter)} ({len(scores_ans_in_iter)})  Answer NOT in Iter:{np.mean(scores_ans_not_in_iter)} ({len(scores_ans_not_in_iter)})")
    print(f"ITER rr Max:{np.max(scores_ans_in_iter+scores_ans_not_in_iter)}  Min:{np.min(scores_ans_in_iter+scores_ans_not_in_iter)}")

    scores_ans_in_iter_ev = [s['iter_context_ev_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) in s['iter_context']]
    scores_ans_not_in_iter_ev = [s['iter_context_ev_score'] for s in samples if (s['answer'] if type(s['answer'])==str else s['answer'][0]) not in s['iter_context']]
    print(f"ITER ev Mean:{np.mean(scores_ans_in_iter_ev+scores_ans_not_in_iter_ev)}  ITER using ev_score: Answer IN Iter:{np.mean(scores_ans_in_iter_ev)} ({len(scores_ans_in_iter_ev)})  Answer NOT in Iter:{np.mean(scores_ans_not_in_iter_ev)} ({len(scores_ans_not_in_iter_ev)})")
    print(f"ITER ev Max:{np.max(scores_ans_in_iter_ev+scores_ans_not_in_iter_ev)}  Min:{np.min(scores_ans_in_iter_ev+scores_ans_not_in_iter_ev)}")

    return


def create_combo_context(llm_context, iter_context):
    """ Format combined context
    """
    idx = llm_context.find(' Further Explanation: ')
    if idx != -1:
        initial_context = llm_context[:idx].strip()
        rationale = llm_context[idx+22:].strip()
    else:
        initial_context = ''
        rationale = llm_context.strip()
    iter_context = iter_context[len(initial_context):].strip()  # remove initial context if any from retrieved sample
    if initial_context != '' and initial_context[-1] not in ['.', '!', '?', ':', ';']:
        initial_context += '.'
    if rationale != '' and rationale[-1] not in ['.', '!', '?', ':', ';']:
        rationale += '.'
    if iter_context != '' and iter_context[-1] not in ['.', '!', '?', ':', ';']:
        iter_context += '.'
    if rationale != '' and (initial_context != '' or iter_context != ''):
        new_context = (initial_context + ' Further Explanation: ' + rationale + ' ' + iter_context).strip()
    elif rationale != '':
        new_context = (initial_context + ' ' + rationale + ' ' + iter_context).strip()
    else:
        new_context = (initial_context + ' ' + iter_context).strip()
    return new_context



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
    args.num_workers_dev = 10
    args.model_name = 'google/electra-large-discriminator'
    args.init_checkpoint = '/large_data/thar011/out/mdr/logs/RR_test4_mcstrip0.5_notsinglepossplit_withsharednormal-04-11-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt'
    args.predict_batch_size = 100
    args.output_dir = '/large_data/thar011/out/mdr/logs'

    CSQA:
    args.prefix = 'LLM_ITER_MERGE_TEST0CSQA'
    args.llm_file = '/data/thar011/data/unifiedqa/commonsenseqa_llm_expl/dev.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_csqa_test66_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.base_dataset = 'commonsenseqa'
    
    SQA:
    args.prefix = 'LLM_ITER_MERGE_TEST0SQA'
    args.llm_file = '/data/thar011/data/unifiedqa/strategy_qa_bigbench_llm_expl/dev.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_sqabb_test64_b150_h4_hpqahovnqmubs250_mom-09-30-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.base_dataset = 'strategy_qa_bigbench'
    
    ARCDA:
    args.prefix = 'LLM_ITER_MERGE_TEST0ARCDA'
    args.llm_file = '/data/thar011/data/unifiedqa/arc_da_od_ans_llm_expl/test.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_arcdatst_test70_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.base_dataset = 'arc_da_od_ans'

    IIRC:
    args.prefix = 'LLM_ITER_MERGE_TEST0IIRC'
    args.llm_file = '/data/thar011/data/unifiedqa/iirc_initial_context_llm_expl/test.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_iircictst_test69_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.base_dataset = 'iirc_initial_context'

    MU_DEV:
    args.prefix = 'LLM_ITER_MERGE_TEST0MU_DEV'
    args.llm_file = '/data/thar011/data/unifiedqa/musique_mu_dev_odv2_llm_expl/dev.tsv'
    args.iter_file = '/large_data/thar011/out/mdr/logs/ITER_fullwiki_us_mudev_test71_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
    args.base_dataset = 'musique_mu_dev_odv2'
        
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
    eval_dataset = RREvalDataset(args, tokenizer, samples_llm, score_llm=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
    #TJH batch = next(iter(eval_dataloader))
    scores, qids = predict_simple(eval_dataloader, model)
    for i, s in enumerate(scores):
        samples_llm[i]['iter_context_rr_score'] = s
        
    utils.saveas_jsonl(samples_llm, os.path.join(args.output_dir, 'samples_llm_iter_scored.jsonl'))
    answer_in_expl(samples_llm)
    
    # output eval tsv files
#    llm_rr_thresholds = [0.0, 0.0003, 0.0005, 0.0008, 0.00099, 0.005, 0.05, 0.5, 0.75, 0.9]
#    iter_rr_thresholds = [0.0, 0.0003, 0.0005, 0.0008, 0.00099, 0.005, 0.05, 0.5, 0.75, 0.9]
#    iter_ev_thresholds = [0.0, 0.0003, 0.0005, 0.0008, 0.00099, 0.005, 0.05, 0.5, 0.75, 0.9]

    llm_rr_thresholds = [0.0005, 0.00099, 0.005, 0.75, 0.9]
    iter_rr_thresholds = [0.005, 0.0099, 0.05, 0.75, 0.9]  # rr scores on iter tend to be higher than on llm especially at lower values
    iter_ev_thresholds = [0.00099, 0.005, 0.75, 0.9]  # min evs around 0.0005 so exclude 0.0005 thresh

    # q[+mc][+LLM expls that score over a threshold][+Iterator contexts with ev/rr score over a threshold] -> a
    # 25 vs rr and 20 vs ev
    file = split + '.tsv'
    for llm_rr_thresh in llm_rr_thresholds:
        for iter_rr_thresh in iter_rr_thresholds:
            outdir = os.path.join(UQA_DIR, f"{args.base_dataset}_llm_expl_rr{str(llm_rr_thresh)}_fullwiki_rr{str(iter_rr_thresh)}")
            logger.info(f"Output tsv to: {outdir}")
            os.makedirs(outdir, exist_ok=True)
            out_list = []
            for s in samples_llm:
                llm_context = s['context'] if s['llm_rr_score'] > llm_rr_thresh else ''
                iter_context = s['iter_context'] if s['iter_context_rr_score'] > iter_rr_thresh else ''
                new_context = create_combo_context(llm_context, iter_context)
                out_list.append( utils.create_uqa_example(s['q_only'], 
                                          utils.create_uqa_context(s['mc_options'], new_context), 
                                          s['answer']) )
            utils.save_uqa(out_list, outdir, file)

        for iter_ev_thresh in iter_ev_thresholds:
            outdir = os.path.join(UQA_DIR, f"{args.base_dataset}_llm_expl_rr{str(llm_rr_thresh)}_fullwiki_ev{str(iter_ev_thresh)}")
            logger.info(f"Output tsv to: {outdir}")
            os.makedirs(outdir, exist_ok=True)
            out_list = []
            for s in samples_llm:
                llm_context = s['context'] if s['llm_rr_score'] > llm_rr_thresh else ''
                iter_context = s['iter_context'] if s['iter_context_ev_score'] > iter_ev_thresh else ''
                new_context = create_combo_context(llm_context, iter_context)
                out_list.append( utils.create_uqa_example(s['q_only'], 
                                          utils.create_uqa_context(s['mc_options'], new_context), 
                                          s['answer']) )
            utils.save_uqa(out_list, outdir, file)
        
    logger.info("Finished!")
    #- q[+mc][+LLM expls that score over a threshold] -> a                           (LLM if good else nothing)
    # 10 per dataset
    #- q[+mc][+Iterator contexts with ev score over a threshold] -> a                (Iterator if good else nothing)
    # 10 per dataset
    #- q[+mc][+LLM expls that score over a threshold else Iterator context] -> a     (LLM if good else Iterator)
    # 10 per dataset
    #- q[+mc][+Iterator contexts that score over a threshold else LLM context] -> a  (Iterator if good else LLM)
    # 10 per dataset
    #- q[+mc][+max rr score of LLM expls or Iterator context] -> a                   (max rr of LLM / Iterator)
    # 10 
    #- q[+mc][+LLM expls that score over a threshold][+Iterator contexts with ev score over a threshold] -> a (could have one or both of LLM & Iterator expls)
    # 10 per dataset with fixed ev threshold



    
    
    