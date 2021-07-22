#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:56:01 2021

@author: Tim Hartill


Check before and after dataset filtering answer distributions.

Edit the directories and filenames below before running...

"""

import os
import json
import numpy as np
from overlap_detector import SimilarityAggregator
import eval_metrics

DSET_MAP = eval_metrics.unifiedqa_unseen_4_map

DATASET_DIR = '/data/thar011/data/unifiedqa/'
OUT_DIR = '/data/thar011/out/unifiedqa_averages/check_answers'
os.makedirs(OUT_DIR, exist_ok=True)
outfile = os.path.join(OUT_DIR, 'tmp_numeric_vs_nonnumeric_answers_3runs.txt')
similarity_file = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json'
results_list_uqa = ['/data/thar011/out/unifiedqa_averages/comp3runs046//v3_avg3runs_eval_metrics.json']
results_list_tdnd = ['/data/thar011/out/unifiedqa_averages/comp3runs046//v7_avg3runs_eval_metrics.json']


def calc_answer_set(s, dsetset, ngram='Unigram'):
    """ Create list of answers for each dataset.
        Returns out_dict: {'dataset': {lots of keys} }
    """
    firstresultset = list(s.eval_results[dsetset[0]].keys())[0]
    print('calculating for result set:', firstresultset)
    out_dict = {}
        
    for dset in dsetset:     # for each eval dataset
        print(f'Processing {dset} ...')
        indir = os.path.join(DATASET_DIR, dset)
        file = ''
        for k in DSET_MAP:
            if dset.startswith(k):
                file = DSET_MAP[k]
        infile = os.path.join(indir, file)
             

        original = []
        out_list = []
        with open(infile, 'r') as f:
            for line in f:
                original.append(line)
        orig_num = len(original)
        out_dict[dset] ={'gt':[],
                         'gt_norm':[],
                         'predictions': [],
                         'predictions_norm': [],
                         'sim_scores': [],
                         'sim_buckets': [],
                         'pred_scores': [],
                         'gt_norm_unique': {},
                         'gt_norm_unique_count': 0,
                         'numeric_idx': [],
                         'non_numeric_idx': []
                        }
        for ind in range(orig_num):
            question, answer = original[ind].split('\t')
            out_dict[dset]['gt'].append(answer.strip())
            pred_answer = s.eval_results[dset][firstresultset]['predictions'][ind]
            out_dict[dset]['predictions'].append(pred_answer.strip())
            pred_score = s.eval_results[dset][firstresultset]['test_scores'][ind]
            out_dict[dset]['pred_scores'].append(pred_score)
            answer_norm = eval_metrics.normalize_answer(answer)
            out_dict[dset]['gt_norm'].append(answer_norm)
            if answer_norm.isdecimal():
                out_dict[dset]['numeric_idx'].append(ind)
            else:
                out_dict[dset]['non_numeric_idx'].append(ind)
                
            pred_norm = eval_metrics.normalize_answer(pred_answer)
            out_dict[dset]['predictions_norm'].append(pred_norm)
            combo_score = s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['combo'][ind]
            out_dict[dset]['sim_scores'].append(combo_score)
            if combo_score < 60:
                bucket = '0:60'
            elif combo_score < 90:
                bucket = '60:90'
            else:
                bucket = '90:101'
            out_dict[dset]['sim_buckets'].append(bucket)
            
        out_dict[dset]['gt_norm_unique'] = set(out_dict[dset]['gt_norm'])
        out_dict[dset]['gt_norm_unique_count'] = len(out_dict[dset]['gt_norm_unique'])
        out_dict[dset]['numeric_count'] = len(out_dict[dset]['numeric_idx'])
        out_dict[dset]['non_numeric_count'] = len(out_dict[dset]['non_numeric_idx'])
        out_dict[dset]['numeric_ratio'] = out_dict[dset]['numeric_count'] / out_dict[dset]['non_numeric_count']
        if out_dict[dset]['numeric_count'] > 0:
            tmp = np.array(out_dict[dset]['pred_scores'])
            tmp = tmp[out_dict[dset]['numeric_idx']]
            out_dict[dset]['numeric_pred_score_mean'] = float(tmp.mean())
        else:    
            out_dict[dset]['numeric_pred_score_mean'] = 0.0
        if out_dict[dset]['non_numeric_count'] > 0:
            tmp = np.array(out_dict[dset]['pred_scores'])
            tmp = tmp[out_dict[dset]['non_numeric_idx']]
            out_dict[dset]['non_numeric_pred_score_mean'] = float(tmp.mean())
        else:    
            out_dict[dset]['non_numeric_pred_score_mean'] = 0.0

        sim_buckets = np.array(out_dict[dset]['sim_buckets'])
        tmp_preds = np.array(out_dict[dset]['pred_scores'])
        for bucket in ['0:60', '60:90', '90:101']:
            out_dict[dset][bucket] = {'numeric':{'count': 0, 'pred_score_mean': 0.0},
                                      'non_numeric':{'count': 0, 'pred_score_mean': 0.0}}
            bucket_indices = np.where(sim_buckets == bucket)
            num = bucket_indices[0].shape[0]
            if num > 0:
                bucket_set = set(bucket_indices[0])
                numeric_set = set(out_dict[dset]['numeric_idx'])
                intersection_numeric = list(bucket_set.intersection(numeric_set))
                if len(intersection_numeric) > 0:     
                    intersection_numeric.sort()
                    out_dict[dset][bucket]['numeric']['count'] = len(intersection_numeric)
                    out_dict[dset][bucket]['numeric']['pred_score_mean'] = float(np.mean(tmp_preds[intersection_numeric]))
                non_numeric_set = set(out_dict[dset]['non_numeric_idx'])
                intersection_non_numeric = list(bucket_set.intersection(non_numeric_set))
                if len(intersection_non_numeric) > 0:     
                    intersection_non_numeric.sort()
                    out_dict[dset][bucket]['non_numeric']['count'] = len(intersection_non_numeric)
                    out_dict[dset][bucket]['non_numeric']['pred_score_mean'] = float(np.mean(tmp_preds[intersection_non_numeric]))

    print('Finished.')    
    return out_dict

def print_results(out_dict, dset, out_type):
    """ print results
    """
    outstr = f"{out_type},{dset},"    
    print(f'Dataset: {out_type} {dset}:')
    print(f"Numeric to Non-Numeric Answer Ratio: {out_dict[dset]['numeric_ratio']:.2f} Numeric Count: {out_dict[dset]['numeric_count']} Non-Numeric Count: {out_dict[dset]['non_numeric_count']}")
    print(f"Prediction Performance: Numeric:{out_dict[dset]['numeric_pred_score_mean']*100:.2f} Non Numeric: {out_dict[dset]['non_numeric_pred_score_mean']*100:.2f}")
    for bucket in ['0:60', '60:90', '90:101']:
        print(f"NUMERIC: Bucket: {bucket}  Count: {out_dict[dset][bucket]['numeric']['count']}  Mean Perf: {out_dict[dset][bucket]['numeric']['pred_score_mean']*100:.2f}")
        print(f"NON-NUMERIC: Bucket: {bucket}  Count: {out_dict[dset][bucket]['non_numeric']['count']}  Mean Perf: {out_dict[dset][bucket]['non_numeric']['pred_score_mean']*100:.2f}")
        outstr += f"{out_dict[dset][bucket]['numeric']['pred_score_mean']*100:.2f} ({out_dict[dset][bucket]['numeric']['count']}),{out_dict[dset][bucket]['non_numeric']['pred_score_mean']*100:.2f} ({out_dict[dset][bucket]['non_numeric']['count']}),"
    outstr = outstr[:-1]    
    return outstr



sim_results = json.load(open(similarity_file))  # cosine sim over sentence embeddings
# run these steps to produce summary by similarity bucket without breaking down by individual dataset
s_uqa_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_uqa, compare_over='UQA',
                             thresh_buckets = [0,60,90,101])

s_tdnd_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_tdnd,
                              thresh_buckets = [0,60,90,101])

orig_uqa = calc_answer_set(s_uqa_summary, dsetset=eval_metrics.unifiedqa_unseen_4)
lowsim_uqa = calc_answer_set(s_uqa_summary, dsetset=eval_metrics.unifiedqa_unseen_6)

orig_tdnd = calc_answer_set(s_tdnd_summary, dsetset=eval_metrics.unifiedqa_unseen_4)
lowsim_tdnd = calc_answer_set(s_tdnd_summary, dsetset=eval_metrics.unifiedqa_unseen_6)

outlist = ['Model,Eval Dataset,NUM:0:60,NONNUM:0:60,NUM:60:90,NONNUM:60:90,NUM:90:101,NUM:90:101']
for dset in eval_metrics.unifiedqa_unseen_4:
    outstr = print_results(orig_uqa, dset, 'UQA')
    outlist.append(outstr)
    outstr = print_results(orig_tdnd, dset, 'UQA+TDND')   
    outlist.append(outstr)
    outstr = print_results(lowsim_uqa, dset+'_lowsim_tdnd', 'UQA')
    outlist.append(outstr)
    outstr = print_results(lowsim_tdnd, dset+'_lowsim_tdnd', 'UQA+TDND')    
    outlist.append(outstr)
with open(outfile, 'w') as f:
    f.write('\r\n'.join(outlist))
print(f"Output to: {outfile}")

"""
LOG:
Dataset: UQA drop_dedup:
Numeric to Non-Numeric Answer Ratio: 1.67 Numeric Count: 5457 Non-Numeric Count: 3277
Prediction Performance: Numeric:7.49 Non Numeric: 40.57
NUMERIC: Bucket: 0:60  Count: 178  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 1344  Mean Perf: 41.55
NUMERIC: Bucket: 60:90  Count: 5273  Mean Perf: 7.76
NON-NUMERIC: Bucket: 60:90  Count: 1913  Mean Perf: 39.90
NUMERIC: Bucket: 90:101  Count: 6  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 20  Mean Perf: 39.00

Dataset: UQA+TDND drop_dedup:
Numeric to Non-Numeric Answer Ratio: 1.67 Numeric Count: 5457 Non-Numeric Count: 3277
Prediction Performance: Numeric:12.39 Non Numeric: 46.34
NUMERIC: Bucket: 0:60  Count: 5  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 867  Mean Perf: 44.99
NUMERIC: Bucket: 60:90  Count: 4814  Mean Perf: 11.18
NON-NUMERIC: Bucket: 60:90  Count: 2363  Mean Perf: 46.61
NUMERIC: Bucket: 90:101  Count: 638  Mean Perf: 21.58
NON-NUMERIC: Bucket: 90:101  Count: 47  Mean Perf: 57.73

Dataset: UQA drop_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.66 Numeric Count: 1238 Non-Numeric Count: 1864
Prediction Performance: Numeric:3.88 Non Numeric: 40.53
NUMERIC: Bucket: 0:60  Count: 84  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 1045  Mean Perf: 42.09
NUMERIC: Bucket: 60:90  Count: 1154  Mean Perf: 4.16
NON-NUMERIC: Bucket: 60:90  Count: 819  Mean Perf: 38.55
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND drop_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.66 Numeric Count: 1238 Non-Numeric Count: 1864
Prediction Performance: Numeric:6.58 Non Numeric: 46.06
NUMERIC: Bucket: 0:60  Count: 5  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 652  Mean Perf: 46.92
NUMERIC: Bucket: 60:90  Count: 1229  Mean Perf: 6.39
NON-NUMERIC: Bucket: 60:90  Count: 1211  Mean Perf: 45.64
NUMERIC: Bucket: 90:101  Count: 4  Mean Perf: 75.00
NON-NUMERIC: Bucket: 90:101  Count: 1  Mean Perf: 0.00

Dataset: UQA contrast_sets_drop_dedup:
Numeric to Non-Numeric Answer Ratio: 0.98 Numeric Count: 467 Non-Numeric Count: 478
Prediction Performance: Numeric:9.71 Non Numeric: 30.87
NUMERIC: Bucket: 0:60  Count: 2  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 209  Mean Perf: 37.08
NUMERIC: Bucket: 60:90  Count: 465  Mean Perf: 9.75
NON-NUMERIC: Bucket: 60:90  Count: 269  Mean Perf: 26.04
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND contrast_sets_drop_dedup:
Numeric to Non-Numeric Answer Ratio: 0.98 Numeric Count: 467 Non-Numeric Count: 478
Prediction Performance: Numeric:10.42 Non Numeric: 40.69
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 154  Mean Perf: 45.21
NUMERIC: Bucket: 60:90  Count: 370  Mean Perf: 8.11
NON-NUMERIC: Bucket: 60:90  Count: 324  Mean Perf: 38.54
NUMERIC: Bucket: 90:101  Count: 97  Mean Perf: 19.24
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA contrast_sets_drop_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.23 Numeric Count: 61 Non-Numeric Count: 265
Prediction Performance: Numeric:0.00 Non Numeric: 32.45
NUMERIC: Bucket: 0:60  Count: 2  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 156  Mean Perf: 38.35
NUMERIC: Bucket: 60:90  Count: 59  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 109  Mean Perf: 24.02
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND contrast_sets_drop_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.23 Numeric Count: 61 Non-Numeric Count: 265
Prediction Performance: Numeric:8.20 Non Numeric: 40.50
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 110  Mean Perf: 46.66
NUMERIC: Bucket: 60:90  Count: 60  Mean Perf: 8.33
NON-NUMERIC: Bucket: 60:90  Count: 155  Mean Perf: 36.12
NUMERIC: Bucket: 90:101  Count: 1  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA mmlu_elementary_to_college_math_test:
Numeric to Non-Numeric Answer Ratio: 0.82 Numeric Count: 433 Non-Numeric Count: 530
Prediction Performance: Numeric:27.02 Non Numeric: 27.17
NUMERIC: Bucket: 0:60  Count: 85  Mean Perf: 24.71
NON-NUMERIC: Bucket: 0:60  Count: 350  Mean Perf: 26.00
NUMERIC: Bucket: 60:90  Count: 348  Mean Perf: 27.59
NON-NUMERIC: Bucket: 60:90  Count: 180  Mean Perf: 29.44
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND mmlu_elementary_to_college_math_test:
Numeric to Non-Numeric Answer Ratio: 0.82 Numeric Count: 433 Non-Numeric Count: 530
Prediction Performance: Numeric:24.94 Non Numeric: 27.55
NUMERIC: Bucket: 0:60  Count: 3  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 217  Mean Perf: 23.96
NUMERIC: Bucket: 60:90  Count: 430  Mean Perf: 25.12
NON-NUMERIC: Bucket: 60:90  Count: 313  Mean Perf: 30.03
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA mmlu_elementary_to_college_math_test_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.40 Numeric Count: 139 Non-Numeric Count: 346
Prediction Performance: Numeric:25.18 Non Numeric: 27.46
NUMERIC: Bucket: 0:60  Count: 55  Mean Perf: 20.00
NON-NUMERIC: Bucket: 0:60  Count: 252  Mean Perf: 25.40
NUMERIC: Bucket: 60:90  Count: 84  Mean Perf: 28.57
NON-NUMERIC: Bucket: 60:90  Count: 94  Mean Perf: 32.98
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND mmlu_elementary_to_college_math_test_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.40 Numeric Count: 139 Non-Numeric Count: 346
Prediction Performance: Numeric:25.90 Non Numeric: 27.46
NUMERIC: Bucket: 0:60  Count: 3  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 133  Mean Perf: 23.31
NUMERIC: Bucket: 60:90  Count: 136  Mean Perf: 26.47
NON-NUMERIC: Bucket: 60:90  Count: 213  Mean Perf: 30.05
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA physical_iqa:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1838
Prediction Performance: Numeric:0.00 Non Numeric: 62.51
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 1365  Mean Perf: 61.54
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 473  Mean Perf: 65.33
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND physical_iqa:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1838
Prediction Performance: Numeric:0.00 Non Numeric: 62.95
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 1354  Mean Perf: 62.78
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 484  Mean Perf: 63.43
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA physical_iqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 722
Prediction Performance: Numeric:0.00 Non Numeric: 63.99
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 598  Mean Perf: 63.04
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 124  Mean Perf: 68.55
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND physical_iqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 722
Prediction Performance: Numeric:0.00 Non Numeric: 62.60
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 588  Mean Perf: 61.22
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 134  Mean Perf: 68.66
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA social_iqa_dedup:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1935
Prediction Performance: Numeric:0.00 Non Numeric: 53.54
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 736  Mean Perf: 52.85
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 1199  Mean Perf: 53.96
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND social_iqa_dedup:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1935
Prediction Performance: Numeric:0.00 Non Numeric: 54.47
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 718  Mean Perf: 52.65
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 1217  Mean Perf: 55.55
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA social_iqa_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 753
Prediction Performance: Numeric:0.00 Non Numeric: 57.64
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 383  Mean Perf: 56.40
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 370  Mean Perf: 58.92
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND social_iqa_dedup_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 753
Prediction Performance: Numeric:0.00 Non Numeric: 56.31
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 373  Mean Perf: 55.23
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 380  Mean Perf: 57.37
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA commonsenseqa:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1221
Prediction Performance: Numeric:0.00 Non Numeric: 55.77
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 335  Mean Perf: 57.01
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 886  Mean Perf: 55.30
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND commonsenseqa:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 1221
Prediction Performance: Numeric:0.00 Non Numeric: 55.61
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 233  Mean Perf: 63.95
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 988  Mean Perf: 53.64
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA commonsenseqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 408
Prediction Performance: Numeric:0.00 Non Numeric: 59.56
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 155  Mean Perf: 57.42
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 253  Mean Perf: 60.87
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND commonsenseqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 408
Prediction Performance: Numeric:0.00 Non Numeric: 60.05
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 129  Mean Perf: 66.67
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 279  Mean Perf: 56.99
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA qasc:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 923
Prediction Performance: Numeric:33.33 Non Numeric: 38.89
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 218  Mean Perf: 33.49
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 33.33
NON-NUMERIC: Bucket: 60:90  Count: 704  Mean Perf: 40.48
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 1  Mean Perf: 100.00

Dataset: UQA+TDND qasc:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 923
Prediction Performance: Numeric:33.33 Non Numeric: 35.43
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 139  Mean Perf: 33.81
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 33.33
NON-NUMERIC: Bucket: 60:90  Count: 783  Mean Perf: 35.63
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 1  Mean Perf: 100.00

Dataset: UQA qasc_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 1 Non-Numeric Count: 344
Prediction Performance: Numeric:0.00 Non Numeric: 39.24
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 142  Mean Perf: 33.10
NUMERIC: Bucket: 60:90  Count: 1  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 202  Mean Perf: 43.56
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND qasc_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 1 Non-Numeric Count: 344
Prediction Performance: Numeric:0.00 Non Numeric: 36.34
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 99  Mean Perf: 32.32
NUMERIC: Bucket: 60:90  Count: 1  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 245  Mean Perf: 37.96
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA qasc_with_ir:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 923
Prediction Performance: Numeric:66.67 Non Numeric: 56.99
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 115  Mean Perf: 53.91
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 66.67
NON-NUMERIC: Bucket: 60:90  Count: 808  Mean Perf: 57.43
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND qasc_with_ir:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 923
Prediction Performance: Numeric:0.00 Non Numeric: 57.85
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 95  Mean Perf: 51.58
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 828  Mean Perf: 58.57
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA qasc_with_ir_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 1 Non-Numeric Count: 337
Prediction Performance: Numeric:0.00 Non Numeric: 56.68
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 81  Mean Perf: 48.15
NUMERIC: Bucket: 60:90  Count: 1  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 256  Mean Perf: 59.38
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND qasc_with_ir_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 1 Non-Numeric Count: 337
Prediction Performance: Numeric:0.00 Non Numeric: 54.30
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 72  Mean Perf: 50.00
NUMERIC: Bucket: 60:90  Count: 1  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 265  Mean Perf: 55.47
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA ropes:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 1685
Prediction Performance: Numeric:0.00 Non Numeric: 43.58
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 339  Mean Perf: 51.09
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 1346  Mean Perf: 41.69
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND ropes:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 3 Non-Numeric Count: 1685
Prediction Performance: Numeric:0.00 Non Numeric: 47.69
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 307  Mean Perf: 46.41
NUMERIC: Bucket: 60:90  Count: 3  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 1378  Mean Perf: 47.97
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA ropes_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 461
Prediction Performance: Numeric:0.00 Non Numeric: 45.99
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 197  Mean Perf: 44.34
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 264  Mean Perf: 47.22
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND ropes_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.00 Numeric Count: 0 Non-Numeric Count: 461
Prediction Performance: Numeric:0.00 Non Numeric: 58.94
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 197  Mean Perf: 43.87
NUMERIC: Bucket: 60:90  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 60:90  Count: 264  Mean Perf: 70.19
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA newsqa:
Numeric to Non-Numeric Answer Ratio: 0.05 Numeric Count: 222 Non-Numeric Count: 4119
Prediction Performance: Numeric:72.76 Non Numeric: 54.70
NUMERIC: Bucket: 0:60  Count: 4  Mean Perf: 41.67
NON-NUMERIC: Bucket: 0:60  Count: 1225  Mean Perf: 49.65
NUMERIC: Bucket: 60:90  Count: 218  Mean Perf: 73.33
NON-NUMERIC: Bucket: 60:90  Count: 2894  Mean Perf: 56.83
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND newsqa:
Numeric to Non-Numeric Answer Ratio: 0.05 Numeric Count: 222 Non-Numeric Count: 4119
Prediction Performance: Numeric:73.23 Non Numeric: 56.49
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 1204  Mean Perf: 50.54
NUMERIC: Bucket: 60:90  Count: 222  Mean Perf: 73.23
NON-NUMERIC: Bucket: 60:90  Count: 2915  Mean Perf: 58.94
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA newsqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.04 Numeric Count: 73 Non-Numeric Count: 1871
Prediction Performance: Numeric:67.12 Non Numeric: 50.79
NUMERIC: Bucket: 0:60  Count: 3  Mean Perf: 33.33
NON-NUMERIC: Bucket: 0:60  Count: 767  Mean Perf: 50.52
NUMERIC: Bucket: 60:90  Count: 70  Mean Perf: 68.57
NON-NUMERIC: Bucket: 60:90  Count: 1104  Mean Perf: 50.97
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00

Dataset: UQA+TDND newsqa_lowsim_tdnd:
Numeric to Non-Numeric Answer Ratio: 0.04 Numeric Count: 73 Non-Numeric Count: 1871
Prediction Performance: Numeric:63.70 Non Numeric: 53.18
NUMERIC: Bucket: 0:60  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 0:60  Count: 759  Mean Perf: 51.35
NUMERIC: Bucket: 60:90  Count: 73  Mean Perf: 63.70
NON-NUMERIC: Bucket: 60:90  Count: 1112  Mean Perf: 54.42
NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
NON-NUMERIC: Bucket: 90:101  Count: 0  Mean Perf: 0.00
"""


