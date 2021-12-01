#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:56:01 2021

@author: Tim Hartill

Setting this up is complicated! You may find it much easier to extract the low similarity datasets from
the tar file provided in /low_similarity_datasets!

Create "least similar" versions of eval datasets that only contain examples where there is no word overlap between
the groundtruth answer and the answer in the most similar training sample.

Before running this script it is necessary to:
    (1) Download the preprocessed unifiedqa datasets (instructions at https://github.com/allenai/unifiedqa)
    (2) Download the numerical reasoning datasets (instructions at https://github.com/ag1988/injecting_numeracy/tree/master/pre_training) 
    (3) Download the MMLU datasets (instructions at https://github.com/hendrycks/test)
    (4) Preprocess the numerical reasoning tasks using encode_numerical_literacy.py
    (5) Preprocess the MMLU tasks using encode_mmlu.py
    (6) De-duplicate evaluation datasets using deduplicate.py
    (7) Aggregate the Math MMLU tasks into a single dataset using aggregate_mmlu_math.py
    (8) Train separate models for the UQA and UQA+TDND datasets 
        (runtrain_bart_origV3.sh and runtrain_bart_indivdigitsV7.sh )
    (9) Run the evaluation routines for these models to create the eval_metrics.json files 
        (runevalall_bartlarge_pick_ckpt150k.sh and runevalall_bartlarge_indivdigits_pick_ckpt150k.sh)
        NOTE:These scripts will give errors on the final section as they try to create metrics for the 
             as-yet nonexistent low similarity datasets. Ignore these errors.
    (10) Create the sentence embeddings for all train and eval datasets.
         (runsembeddings_bart_indivdigits_tdnd_V7.sh)
    (11) Run the test-train sentence embedding similarity calculations.
         (runsim_for_sembeddings_bart_indivdigits_tdnd_V7.sh)
    (12) Update all the hard-coded directory and file names below and run this script
    
    (13) After running this script you will want to edit and run this bash script 
         to add evaluation metrics for your new "least" similar datasets to the 
         relevant eval_metrics.json files:
          run_add_lowsim_to_existing_pred_calcmetrics_semb_simjsonfile.sh

"""

import os
import json
import numpy as np
from overlap_detector import SimilarityAggregator
import eval_metrics

DSET_MAP = eval_metrics.unifiedqa_unseen_4_map
DSETSET = eval_metrics.unifiedqa_unseen_4

DATASET_DIR = '/data/thar011/data/unifiedqa/'
similarity_file = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json'
results_list_uqa = ['/data/thar011/out/unifiedqa_bart_large_v3/eval_metrics.json']
results_list_tdnd = ['/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_metrics.json']


def calc_no_answer_overlap(s, ngram='Unigram', writefile=True):
    """ Create dataset of dev set items with no answer overlap with most similar train item.
    """
    firstresultset = list(s.eval_results[DSETSET[0]].keys())[0]
    print('calculating for result set:', firstresultset)
    if s.compare_over == ['ALL']:  # +tdnd
        dir_add = '_lowsim_tdnd'
    else:
        dir_add = '_lowsim_uqa'
        
    for dset in DSETSET:     # for each eval dataset
        print(f'Processing {dset} ...')
        indir = os.path.join(DATASET_DIR, dset)
        if DSET_MAP.get(dset) is not None:
            file = DSET_MAP[dset]
        else:
            file = 'dev.tsv'
        infile = os.path.join(indir, file)
        outdir = os.path.join(DATASET_DIR, dset + dir_add)
        outfile = os.path.join(outdir, file)
        os.makedirs(outdir, exist_ok=True)     

        original = []
        out_list = []
        with open(infile, 'r') as f:
            for line in f:
                original.append(line)
        orig_num = len(original)
        ind_list = []
        most_sim_ans = []
        f1_list = []
        simscore_list = []
        for ind in range(orig_num):
            question, answer = original[ind].split('\t')   
            pred_answer = s.eval_results[dset][firstresultset]['predictions'][ind]
            combo_score = s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['combo'][ind]
            most_similar_train_answer = s.sim_results_max[dset]['max_sim_over_train']['sim_details'][ngram]['combo'][ind][1]
            most_similar_train_answer = most_similar_train_answer.split('label:')[1]
            answer_norm = eval_metrics.normalize_answer(answer)
            most_similar_train_answer_norm = eval_metrics.normalize_answer(most_similar_train_answer)
            anslist = answer_norm.split()
            train_anslist = most_similar_train_answer_norm.split()         
            set1 = set(anslist) 
            set2 = set(train_anslist) 
            if set1.intersection(set2): 
                continue 
            f1 = eval_metrics.get_f1(pred_answer, answer)
            out_list.append( original[ind] )
            ind_list.append(ind)
            most_sim_ans.append(most_similar_train_answer)
            f1_list.append(f1)
            simscore_list.append(combo_score)
        if writefile:    
            with open(outfile, 'w') as f:
                f.write(''.join(out_list))
        simscore_np = np.array(simscore_list)
        f1_np = np.array(f1_list)
        lowbucket_indices = np.where(simscore_np < 60)
        lowbucket_mean = np.mean(f1_np[lowbucket_indices])
        midbucket_indices = np.where((simscore_np >= 60) & (simscore_np < 90))
        if midbucket_indices[0].shape[0] > 0:
            midbucket_mean = np.mean(f1_np[midbucket_indices])
        else:
            midbucket_mean = -1.0
        highbucket_indices = np.where(simscore_np >= 90) 
        if highbucket_indices[0].shape[0] > 0:
            highbucket_mean = np.mean(f1_np[highbucket_indices])
        else:
            highbucket_mean = -1.0
        
        overall_mean = np.mean(f1_np)
        print(f'{dset}: Overall Count:{len(ind_list)} Lowbucket count: {lowbucket_indices[0].shape[0]} (Overall F1:{(overall_mean * 100):.2f}, Lowbucket F1:{(lowbucket_mean * 100):.2f} Midbucket F1:{midbucket_mean*100:.2f} Highbucket F1:{highbucket_mean*100:.2f}) have no answer overlap of {orig_num} originally. Written to {outfile}')
    print(f'Finished {dir_add}')    
    return




sim_results = json.load(open(similarity_file))  # cosine sim over sentence embeddings
# run these steps to produce summary by similarity bucket without breaking down by individual dataset
s_uqa_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_uqa, compare_over='UQA',
                             thresh_buckets = [0,60,90,101])

s_tdnd_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_tdnd,
                              thresh_buckets = [0,60,90,101])

calc_no_answer_overlap(s_uqa_summary, writefile=False) #test run
calc_no_answer_overlap(s_tdnd_summary, writefile=False)

calc_no_answer_overlap(s_uqa_summary)
calc_no_answer_overlap(s_tdnd_summary)


""" Log of above run:
calc_no_answer_overlap(s_uqa_summary, writefile=False)
calculating for result set: unifiedqa_bart_large_v3
Processing drop_dedup ...
drop_dedup: Overall Count:5090 Lowbucket count: 1097 (Overall F1:15.95, Lowbucket F1:34.71 Midbucket F1:10.79 Highbucket F1:-100.00) have no answer overlap of 8734 originally. Written to /data/thar011/data/unifiedqa/drop_dedup_lowsim_uqa/dev.tsv
Processing contrast_sets_drop_dedup ...
contrast_sets_drop_dedup: Overall Count:474 Lowbucket count: 145 (Overall F1:20.19, Lowbucket F1:37.62 Midbucket F1:12.51 Highbucket F1:-100.00) have no answer overlap of 945 originally. Written to /data/thar011/data/unifiedqa/contrast_sets_drop_dedup_lowsim_uqa/dev.tsv
Processing mmlu_elementary_to_college_math_test ...
mmlu_elementary_to_college_math_test: Overall Count:570 Lowbucket count: 322 (Overall F1:40.08, Lowbucket F1:45.34 Midbucket F1:33.25 Highbucket F1:-100.00) have no answer overlap of 963 originally. Written to /data/thar011/data/unifiedqa/mmlu_elementary_to_college_math_test_lowsim_uqa/test.tsv
Processing physical_iqa ...
physical_iqa: Overall Count:699 Lowbucket count: 575 (Overall F1:84.15, Lowbucket F1:84.25 Midbucket F1:83.68 Highbucket F1:-100.00) have no answer overlap of 1838 originally. Written to /data/thar011/data/unifiedqa/physical_iqa_lowsim_uqa/dev.tsv
Processing social_iqa_dedup ...
social_iqa_dedup: Overall Count:748 Lowbucket count: 385 (Overall F1:60.62, Lowbucket F1:60.35 Midbucket F1:60.91 Highbucket F1:-100.00) have no answer overlap of 1935 originally. Written to /data/thar011/data/unifiedqa/social_iqa_dedup_lowsim_uqa/dev.tsv
Processing commonsenseqa ...
commonsenseqa: Overall Count:472 Lowbucket count: 200 (Overall F1:58.69, Lowbucket F1:56.58 Midbucket F1:60.23 Highbucket F1:-100.00) have no answer overlap of 1221 originally. Written to /data/thar011/data/unifiedqa/commonsenseqa_lowsim_uqa/dev.tsv
Processing qasc ...
qasc: Overall Count:344 Lowbucket count: 148 (Overall F1:38.80, Lowbucket F1:34.74 Midbucket F1:41.87 Highbucket F1:-100.00) have no answer overlap of 926 originally. Written to /data/thar011/data/unifiedqa/qasc_lowsim_uqa/dev.tsv
Processing qasc_with_ir ...
qasc_with_ir: Overall Count:357 Lowbucket count: 86 (Overall F1:55.93, Lowbucket F1:49.42 Midbucket F1:58.00 Highbucket F1:-100.00) have no answer overlap of 926 originally. Written to /data/thar011/data/unifiedqa/qasc_with_ir_lowsim_uqa/dev.tsv
Processing ropes ...
ropes: Overall Count:522 Lowbucket count: 234 (Overall F1:47.93, Lowbucket F1:47.33 Midbucket F1:48.41 Highbucket F1:-100.00) have no answer overlap of 1688 originally. Written to /data/thar011/data/unifiedqa/ropes_lowsim_uqa/dev.tsv
Processing newsqa ...
newsqa: Overall Count:1965 Lowbucket count: 782 (Overall F1:51.54, Lowbucket F1:50.34 Midbucket F1:52.33 Highbucket F1:-100.00) have no answer overlap of 4341 originally. Written to /data/thar011/data/unifiedqa/newsqa_lowsim_uqa/dev.tsv
Finished _lowsim_uqa

calc_no_answer_overlap(s_tdnd_summary, writefile=False)
calculating for result set: unifiedqa_bart_large_v7indiv_digits_tdnd
Processing drop_dedup ...
drop_dedup: Overall Count:3102 Lowbucket count: 657 (Overall F1:30.30, Lowbucket F1:46.56 Midbucket F1:25.87 Highbucket F1:60.00) have no answer overlap of 8734 originally. Written to /data/thar011/data/unifiedqa/drop_dedup_lowsim_tdnd/dev.tsv
Processing contrast_sets_drop_dedup ...
contrast_sets_drop_dedup: Overall Count:326 Lowbucket count: 110 (Overall F1:34.45, Lowbucket F1:46.66 Midbucket F1:28.37 Highbucket F1:0.00) have no answer overlap of 945 originally. Written to /data/thar011/data/unifiedqa/contrast_sets_drop_dedup_lowsim_tdnd/dev.tsv
Processing mmlu_elementary_to_college_math_test ...
mmlu_elementary_to_college_math_test: Overall Count:485 Lowbucket count: 136 (Overall F1:44.22, Lowbucket F1:48.75 Midbucket F1:42.45 Highbucket F1:-100.00) have no answer overlap of 963 originally. Written to /data/thar011/data/unifiedqa/mmlu_elementary_to_college_math_test_lowsim_tdnd/test.tsv
Processing physical_iqa ...
physical_iqa: Overall Count:722 Lowbucket count: 588 (Overall F1:84.06, Lowbucket F1:84.11 Midbucket F1:83.87 Highbucket F1:-100.00) have no answer overlap of 1838 originally. Written to /data/thar011/data/unifiedqa/physical_iqa_lowsim_tdnd/dev.tsv
Processing social_iqa_dedup ...
social_iqa_dedup: Overall Count:753 Lowbucket count: 373 (Overall F1:59.54, Lowbucket F1:59.54 Midbucket F1:59.54 Highbucket F1:-100.00) have no answer overlap of 1935 originally. Written to /data/thar011/data/unifiedqa/social_iqa_dedup_lowsim_tdnd/dev.tsv
Processing commonsenseqa ...
commonsenseqa: Overall Count:408 Lowbucket count: 129 (Overall F1:60.60, Lowbucket F1:67.57 Midbucket F1:57.37 Highbucket F1:-100.00) have no answer overlap of 1221 originally. Written to /data/thar011/data/unifiedqa/commonsenseqa_lowsim_tdnd/dev.tsv
Processing qasc ...
qasc: Overall Count:345 Lowbucket count: 99 (Overall F1:36.87, Lowbucket F1:33.96 Midbucket F1:38.04 Highbucket F1:-100.00) have no answer overlap of 926 originally. Written to /data/thar011/data/unifiedqa/qasc_lowsim_tdnd/dev.tsv
Processing qasc_with_ir ...
qasc_with_ir: Overall Count:338 Lowbucket count: 72 (Overall F1:54.29, Lowbucket F1:51.39 Midbucket F1:55.08 Highbucket F1:-100.00) have no answer overlap of 926 originally. Written to /data/thar011/data/unifiedqa/qasc_with_ir_lowsim_tdnd/dev.tsv
Processing ropes ...
ropes: Overall Count:461 Lowbucket count: 197 (Overall F1:58.94, Lowbucket F1:43.87 Midbucket F1:70.19 Highbucket F1:-100.00) have no answer overlap of 1688 originally. Written to /data/thar011/data/unifiedqa/ropes_lowsim_tdnd/dev.tsv
Processing newsqa ...
newsqa: Overall Count:1944 Lowbucket count: 759 (Overall F1:53.57, Lowbucket F1:51.35 Midbucket F1:54.99 Highbucket F1:-100.00) have no answer overlap of 4341 originally. Written to /data/thar011/data/unifiedqa/newsqa_lowsim_tdnd/dev.tsv
Finished _lowsim_tdnd
"""



