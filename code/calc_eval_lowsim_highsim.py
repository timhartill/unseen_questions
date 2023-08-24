#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:56:01 2021

@author: Tim Hartill


Main results for our memorisation paper

If > 1 (three) runs for each of uqa and uqa_tdnd are done then the mean and std dev over all runs will be output
otherwise just the result for the individual runs

We took the mean/std over three runs for each of uqa and uqa+tdnd using seeds:
    UQA seeds: 42, 22, 11
    UQA+TDND seeds: 42, 22, 11


"""

import os
import json
import argparse
import numpy as np
from overlap_detector import SimilarityAggregator
import eval_metrics
import utils

DSET_MAP = eval_metrics.unifiedqa_unseen_4_map
UQA_DIR = eval_metrics.UQA_DIR


def aggregate_scores_by_bucket(s, eval_datasets):
    """ Aggregate scores for a model run for a set of datasets
    
    s.eval_results['drop_dedup']['unifiedqa_bart_large_v3']['test_scores']
    
    returns out_dict['eval_dataset'] = {'all_scores': [...], 'all_score': score,
                                        'mostsim_scores': [...], 'mostsim_score': score,
                                        'leastsim_scores': [...], 'leastsim_score': score,
                                        }
    """
    firstresultset = list(s.eval_results[eval_datasets[0]].keys())[0]  # resultset = eval_metrics.json. Only one in practice.
    print('Calculating for result set:', firstresultset)
    out_dict = {}
    for dset in eval_datasets:
        print(f'Processing {dset} ...')
        out_dict[dset] = {'all_scores': [], 'all_score': 0.0,
                          'mostsim_scores': [], 'mostsim_score': 0.0,
                          'leastsim_scores': [], 'leastsim_score': 0.0}
        indir = os.path.join(UQA_DIR, dset)
        file = ''
        for k in DSET_MAP:  # find tsv filename
            if dset.startswith(k):
                file = DSET_MAP[k]
        if file == '':
            file = 'dev.tsv'
        infile = os.path.join(indir, file)
        original = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True)    # [{'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        
        for ind, sample in enumerate(original):
            assert type(sample['answer']) == str
            if dset in eval_metrics.textual_answers_only:  # skip numeric answer samples for drop etc
                answer_norm = eval_metrics.normalize_answer_squad(sample['answer'])
                if answer_norm.isdecimal():
                    continue
            pred_score = s.eval_results[dset][firstresultset]['test_scores'][ind]
            combo_score = s.sim_results_max[dset]['max_sim_over_train']['sim_scores']['Unigram']['combo'][ind]
            out_dict[dset]['all_scores'].append(pred_score)
            if combo_score < 60:
                bucket = 'leastsim_scores'
            else:
                bucket = 'mostsim_scores'
            out_dict[dset][bucket].append(pred_score)
            
    for dset in out_dict:   # calc means
        out_dict[dset]['all_count'] = len(out_dict[dset]['all_scores'])
        if len(out_dict[dset]['all_scores']) > 0:
            out_dict[dset]['all_score'] = sum(out_dict[dset]['all_scores']) / len(out_dict[dset]['all_scores'])
        out_dict[dset]['mostsim_count'] = len(out_dict[dset]['mostsim_scores'])
        if len(out_dict[dset]['mostsim_scores']) > 0:
            out_dict[dset]['mostsim_score'] = sum(out_dict[dset]['mostsim_scores']) / len(out_dict[dset]['mostsim_scores'])   
        out_dict[dset]['leastsim_count'] = len(out_dict[dset]['leastsim_scores'])
        if len(out_dict[dset]['leastsim_scores']) > 0:
            out_dict[dset]['leastsim_score'] = sum(out_dict[dset]['leastsim_scores']) / len(out_dict[dset]['leastsim_scores'])
        del out_dict[dset]['all_scores']
        del out_dict[dset]['mostsim_scores']
        del out_dict[dset]['leastsim_scores']
            
    return out_dict                
            
        
def calc_means_by_bucket(sim_results, metrics_files, train_datasets, eval_datasets):
    """ calculate the metric means by bucket over a set of model runs (set could be 1)
    """
    print(f"Calculating similarity for each eval sample over train datasets: {train_datasets}")
    means_list = []
    for f in metrics_files:
        # calculate sim between each eval dataset in eval_metrics.json (f) and train_datasets
        s_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=[f], 
                                         compare_over=train_datasets,
                                         thresh_buckets = [0,60,90,101])  # we aggregate 60:90 into 90+ as "most similar"
        means_list.append( aggregate_scores_by_bucket(s_summary, eval_datasets) )
    return means_list


def calc_mean_std_over_runs(means_list):
    """ Calculate mean and std over over runs
    return mean_over_runs
    """
    all_mean_over_runs = {}
    for i, m in enumerate(means_list):
        for dset in m:
            if i == 0:
                all_mean_over_runs[dset] = {}
                all_mean_over_runs[dset]['all_count'] = m[dset]['all_count']
                all_mean_over_runs[dset]['mostsim_count'] = m[dset]['mostsim_count']
                all_mean_over_runs[dset]['leastsim_count'] = m[dset]['leastsim_count']
                all_mean_over_runs[dset]['all_scores'] = []
                all_mean_over_runs[dset]['mostsim_scores'] = []
                all_mean_over_runs[dset]['leastsim_scores'] = []
                
            all_mean_over_runs[dset]['all_scores'].append(m[dset]['all_score'])
            all_mean_over_runs[dset]['mostsim_scores'].append(m[dset]['mostsim_score'])
            all_mean_over_runs[dset]['leastsim_scores'].append(m[dset]['leastsim_score'])
            
    for dset in all_mean_over_runs:
        all_mean_over_runs[dset]['all_score'] = float(np.mean(all_mean_over_runs[dset]['all_scores'])*100.0)
        all_mean_over_runs[dset]['all_score_std'] = float(np.std(all_mean_over_runs[dset]['all_scores'])*100.0)
        all_mean_over_runs[dset]['mostsim_score'] = float(np.mean(all_mean_over_runs[dset]['mostsim_scores'])*100.0)
        all_mean_over_runs[dset]['mostsim_score_std'] = float(np.std(all_mean_over_runs[dset]['mostsim_scores'])*100.0)
        all_mean_over_runs[dset]['leastsim_score'] = float(np.mean(all_mean_over_runs[dset]['leastsim_scores'])*100.0)
        all_mean_over_runs[dset]['leastsim_score_std'] = float(np.std(all_mean_over_runs[dset]['leastsim_scores'])*100.0)
        del all_mean_over_runs[dset]['all_scores']
        del all_mean_over_runs[dset]['mostsim_scores']
        del all_mean_over_runs[dset]['leastsim_scores']
    return all_mean_over_runs
    

def output_single(uqa_mean_over_runs, uqatdnd_mean_over_runs, 
                 eval_datasets_unfiltered, eval_datasets_filtered, key):
    """ create output for one of means, std dev or counts
    """
    outheader = 'eval_dataset,all_uqa,all_uqatdnd,mostsim_uqa,mostsim_uqatdnd,leastsim_uqa,leastsim_uqatdnd,filtered_leastsim_uqa,filtered_leastsim_uqatdnd'
    outlist = [outheader]
    for i, dset in enumerate(eval_datasets_unfiltered):
        dset_filtered = eval_datasets_filtered[i]
        row=f"{dset},{uqa_mean_over_runs[dset]['all_'+key]},{uqatdnd_mean_over_runs[dset]['all_'+key]},"
        row += f"{uqa_mean_over_runs[dset]['mostsim_'+key]},{uqatdnd_mean_over_runs[dset]['mostsim_'+key]},"
        row += f"{uqa_mean_over_runs[dset]['leastsim_'+key]},{uqatdnd_mean_over_runs[dset]['leastsim_'+key]},"
        row += f"{uqa_mean_over_runs[dset_filtered]['leastsim_'+key]},{uqatdnd_mean_over_runs[dset_filtered]['leastsim_'+key]}"
        outlist.append(row)
        
    return outlist


def output_means(uqa_mean_over_runs, uqatdnd_mean_over_runs, 
                 eval_datasets_unfiltered, eval_datasets_filtered,
                 output_dir):
    """ Output means, std dev and counts in csv files for:
        
        all_uqa, all_uqatdnd, mostsim_uqa, mostsim_uqatdnd, leastsim_uqa, leastsim_uqatdnd, filtered_leastsim_uqa, filtered_leastsim_uqatdnd
    """
    
    key = 'count'
    outlist = output_single(uqa_mean_over_runs, uqatdnd_mean_over_runs, eval_datasets_unfiltered, eval_datasets_filtered, key)
    outfile = os.path.join(output_dir, 'uqa_uqatdnd_bucket_counts.txt')
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    print(f"Output to: {outfile}")        

    key = 'score'
    outlist = output_single(uqa_mean_over_runs, uqatdnd_mean_over_runs, eval_datasets_unfiltered, eval_datasets_filtered, key)
    outfile = os.path.join(output_dir, 'uqa_uqatdnd_bucket_meanscores.txt')
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    print(f"Output to: {outfile}")        

    key = 'score_std'
    outlist = output_single(uqa_mean_over_runs, uqatdnd_mean_over_runs, eval_datasets_unfiltered, eval_datasets_filtered, key)
    outfile = os.path.join(output_dir, 'uqa_uqatdnd_bucket_meanscores_std.txt')
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    print(f"Output to: {outfile}")         
    return


def build_metrics_files(run_subdirs):
    run_subdirs_indiv =  run_subdirs.split(',')
    metrics_files = [os.path.join(args.in_log_dir, sd, 'eval_metrics.json') for sd in run_subdirs_indiv]
    print(f"Loading from metrics files: {metrics_files}")
    for f in metrics_files:
        assert os.path.exists(f)
    return metrics_files


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_log_dir", default="/data/thar011/out", type=str, help="Log directory from which model runs will be read.")
    parser.add_argument("--uqa_run_subdirs", default="unifiedqa_bart_large_v3,unifiedqa_bart_large_V3_run4,unifiedqa_bart_large_V3_run6", type=str, help="Subdirectories under --in_log_dir from which eval_metrics.json can be read.")
    parser.add_argument("--uqatdnd_run_subdirs", default="unifiedqa_bart_large_v7indiv_digits_tdnd,unifiedqa_bart_large_v7_run4_indiv_digits_tdnd,unifiedqa_bart_large_v7_run6_indiv_digits_tdnd", type=str, help="Subdirectories under --in_log_dir from which eval_metrics.json can be read.")
    parser.add_argument("--sim_file", default="/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json", type=str, help="Full path to train-test similarity json file.")
    parser.add_argument("--output_dir", default="/large_data/thar011/out/mdr/logs/eval_outputs/uqa_3runs", type=str, help="Output directory.")
    
    args = parser.parse_args()

    #args.run_subdirs = 'unifiedqa_bart_large_v7indiv_digits_tdnd,unifiedqa_bart_large_v7_run4_indiv_digits_tdnd,unifiedqa_bart_large_v7_run6_indiv_digits_tdnd'
    
    metrics_files_uqa = build_metrics_files(args.uqa_run_subdirs)
    metrics_files_uqatdnd = build_metrics_files(args.uqatdnd_run_subdirs)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Outputs will be in {args.output_dir}")

    train_datasets_uqa = eval_metrics.unifiedqa_base_train_orig
    train_datasets_uqatdnd = eval_metrics.unifiedqa_base_train_orig + ['synthetic_textual', 'synthetic_numeric']

    print(f"Loading similarity file (takes a while): {args.sim_file}")
    sim_results = json.load(open(args.sim_file))  # cosine sim over sentence embeddings
    
    eval_datasets = eval_metrics.unifiedqa_unseen_6_unfiltered + eval_metrics.unifiedqa_unseen_6
    print(f"Eval samples from: {eval_datasets}")


    uqa_means = calc_means_by_bucket(sim_results, metrics_files_uqa, train_datasets_uqa, eval_datasets)
    uqatdnd_means = calc_means_by_bucket(sim_results, metrics_files_uqatdnd, train_datasets_uqatdnd, eval_datasets)
    
    json.dump(uqa_means, open(os.path.join(args.output_dir, 'uqa_means.json'), 'w'))
    json.dump(uqatdnd_means, open(os.path.join(args.output_dir, 'uqatdnd_means.json'), 'w'))

    uqa_mean_over_runs = calc_mean_std_over_runs(uqa_means)
    uqatdnd_mean_over_runs = calc_mean_std_over_runs(uqatdnd_means)

    json.dump(uqa_mean_over_runs, open(os.path.join(args.output_dir, 'uqa_mean_over_runs.json'), 'w'))
    json.dump(uqatdnd_mean_over_runs, open(os.path.join(args.output_dir, 'uqatdnd_mean_over_runs.json'), 'w'))

    
    output_means(uqa_mean_over_runs, uqatdnd_mean_over_runs, 
                 eval_metrics.unifiedqa_unseen_6_unfiltered, eval_metrics.unifiedqa_unseen_6,
                 args.output_dir)
    


