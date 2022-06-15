#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:31:35 2022

@author: tim hartill

test configurations of stopping criteria relative to cls/evidentiality label

input file format is json 
dict_keys(['question', 'context', 
           'ans', 'ans_pred', 'ans_pred_score', 'ans_insuff_score', 
           'sp', 'sp_pred', 'sp_labels', 'sp_scores', 'sp_thresh', 'sp_pred_dict', 
           'para_gold', 'para_pred', 'para_score', 'para_thresh', 'para_pred_dict', 
           'ev_pred', 'ev_scores', 
           'src', 'pos', 'full', 'act_hops', '_id', 
           'answer_em', 'answer_f1', 
           'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 
           'joint_em', 'joint_f1', 
           'para_acc', 'ev_acc'])

{'question': 'The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.',
 'context': ' [SEP] yes no [unused0] [SEP] [unused1] Life Goes On (Fergie song) | The song serves as the third single from Fergie\'s second studio album, following "M.I.L.F. $". [unused1] M.I.L.F. $ | It was produced by Polow da Don and released as the second single from the record following "L.A. Love (La La)" on July 1, 2016 by Interscope and will.i.am Music Group. [unused1] Life Goes On (Fergie song) | "Life Goes On" is a song recorded by American singer Fergie for her second studio album, "Double Dutchess" (2017). [unused1] Life Goes On (Fergie song) | It was released as single on November 11, 2016, by Interscope and will.i.am Music Group. [unused1] Life Goes On (Fergie song) | "Life Goes On" was written by Fergie, Tristan Prettyman, Keith Harris and Toby Gad. [unused1] M.I.L.F. $ | "M.I.L.F. $" (pronounced "MILF money") is a song recorded by American singer Fergie for her second studio album, "Double Dutchess" (2017). [unused1] M.I.L.F. $ | It debuted at number 34 on the US "Billboard" Hot 100 with 65,000 in first-week sales. [unused1] BMG Rights Management | BMG Rights Management GmbH (also known simply as BMG) is an international music company based in Berlin, Germany. [unused1] BMG Rights Management | It combines the activities of a music publisher and a record label.',
 'ans': ['yes'],
 'ans_pred': 'yes',
 'ans_pred_score': 16.16513442993164,
 'ans_insuff_score': 7.10194206237793,
 'sp': [0, 1],
 'sp_pred': [0, 1, 5],
 'sp_labels': [1, 1, 0, 0, 0, 0, 0, 0, 0],
 'sp_scores': [0.9818174242973328,
  0.9702593088150024,
  0.34780269861221313,
  0.0037457782309502363,
  0.006887461990118027,
  0.6930526494979858,
  0.003191587748005986,
  0.00016746899927966297,
  0.00015098678704816848],
 'sp_thresh': 0.5,
 'sp_pred_dict': {'1e-05': [0, 1, 2, 3, 4, 5, 6, 7, 8],
  '0.0001': [0, 1, 2, 3, 4, 5, 6, 7, 8],
  '0.001': [0, 1, 2, 3, 4, 5, 6],
  '0.003125': [0, 1, 2, 3, 4, 5, 6],
  '0.00625': [0, 1, 2, 4, 5],
  '0.0125': [0, 1, 2, 5],
  '0.025': [0, 1, 2, 5],
  '0.05': [0, 1, 2, 5],
  '0.1': [0, 1, 2, 5],
  '0.15': [0, 1, 2, 5],
  '0.2': [0, 1, 2, 5],
  '0.25': [0, 1, 2, 5],
  '0.3': [0, 1, 2, 5],
  '0.35': [0, 1, 5],
  '0.4': [0, 1, 5],
  '0.45': [0, 1, 5],
  '0.5': [0, 1, 5],
  '0.55': [0, 1, 5],
  '0.6': [0, 1, 5],
  '0.7': [0, 1]},
 'para_gold': 1,
 'para_pred': 1,
 'para_score': 0.9953369498252869,
 'para_thresh': 0.5,
 'para_pred_dict': {'0.003125': 1,
  '0.00625': 1,
  '0.0125': 1,
  '0.025': 1,
  '0.05': 1,
  '0.1': 1,
  '0.15': 1,
  '0.2': 1,
  '0.25': 1,
  '0.3': 1,
  '0.35': 1,
  '0.4': 1,
  '0.45': 1,
  '0.475': 1,
  '0.4875': 1,
  '0.5': 1,
  '0.5125': 1,
  '0.525': 1,
  '0.55': 1,
  '0.6': 1,
  '0.65': 1,
  '0.7': 1,
  '0.75': 1,
  '0.8': 1,
  '0.85': 1,
  '0.9': 1,
  '0.95': 1},
 'ev_pred': 1,
 'ev_scores': [-2.5351622104644775, 3.09674334526062],
 'src': 'hover',
 'pos': True,
 'full': 1,
 'act_hops': 2,
 '_id': '042339bf-0374-4ab3-ab49-6df5f12d868e',
 'answer_em': 1,
 'answer_f1': 1.0,
 'sp_em': 0.0,
 'sp_f1': 0.8,
 'sp_prec': 0.6666666666666666,
 'sp_recall': 1.0,
 'joint_em': 0.0,
 'joint_f1': 0.8,
 'para_acc': 1,
 'ev_acc': 1}


"""

import utils
import numpy as np
import copy

infile = '/large_data/thar011/out/mdr/logs/stage2_test2_hpqa_hover_fever_new_sentMASKforcezerospweight1_addevcombinerhead-06-12-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/stage2_dev_predictions Bat Step 95999 Glob Step 12000 Train loss 4.36 para_acc 83.27 epoch2.jsonl'
infile = '/large_data/thar011/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/stage2_dev_predictions.jsonl'


samples = utils.load_jsonl(infile)


def create_grouped_metrics(sample_list, group_key='src',
                           metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 
                                          'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc', 'ev_acc']):
    """ output metrics by group
    """
    grouped_metrics = {}
    for sample in sample_list:
        if grouped_metrics.get(sample[group_key]) is None:
            grouped_metrics[sample[group_key]] = {}
        for key in metric_keys:
            if grouped_metrics[sample[group_key]].get(key) is None:
                grouped_metrics[sample[group_key]][key] = []
            grouped_metrics[sample[group_key]][key].append( sample[key] )
    print("------------------------------------------------")
    print(f"Metrics grouped by: {group_key}")
    print("------------------------------------------------")
    for group in grouped_metrics:
        mgroup = grouped_metrics[group]
        print(f"{group_key}: {group}")
        for key in metric_keys:
            n = len(mgroup[key])
            val = np.mean( mgroup[key] ) if n > 0 else -1
            print(f'{key}: {val}  n={n}')
        print("------------------------------------------------")
    return  


def add_combo_metrics(samples, ans_diff_thresh=5.0, para_pos_conf_thresh = 0.87):
    """ Add different para_acc calcs, see if any are better
    """
    for sample in samples:
        sample['ans_diff_score'] = sample['ans_pred_score'] - sample['ans_insuff_score']
        if sample['para_pred'] == 1:
            if sample['ans_pred'] != '[unused0]':
                sample['calc_para_pred'] = 1  
            elif sample['para_score'] < para_pos_conf_thresh:  # not highly confident in para score
                sample['calc_para_pred'] = 0
            else:
                sample['calc_para_pred'] = 1

        else: # cls pred 0
            if sample['ans_pred'] == '[unused0]':
                sample['calc_para_pred'] = 0
            elif sample['ans_diff_score'] > ans_diff_thresh:  # highly confident in non-insuff answer
                sample['calc_para_pred'] = 1
            else:
                sample['calc_para_pred'] = 0   # not confident in non-insuff answer, go with para_pred
        sample['calc_para_acc'] = int(sample['calc_para_pred'] == sample['para_gold'])
    return


def filter_metrics(samples, key, val=['hover', 'fever']):
    out_samples = []
    for sample in samples:
        if sample[key] in val:
            out_samples.append(copy.deepcopy(sample))
    return out_samples


def add_combo_key(samples, keys=['src', 'ans'], newkeyname='src_ans'):
    for sample in samples:
        newkeyval = '_'.join([str(sample[k]) for k in keys])
        sample[newkeyname] = newkeyval
    return 

def filter_fever(samples):
    """ where multiple gold sents there are issues with "and" vs "or" in fever labelling..
    """
    out_samples = []
    for sample in samples:
        if sample['src'] == 'hover' or sample['para_gold'] == 1 or (sample['para_gold'] == 0 and sum(sample['sp_labels']) == 0):
            out_samples.append(copy.deepcopy(sample))
    return out_samples
        

fever_hover_only = filter_metrics(samples, key='src', val=['hover', 'fever'])  
add_combo_key(fever_hover_only, keys=['src', 'ans'], newkeyname='src_ans')
create_grouped_metrics(fever_hover_only, group_key='src_ans', metric_keys = ['answer_em', 'sp_em', 'sp_recall', 
                                                                      'para_acc', 'ev_acc', 
                                                                      'ans_pred_score', 'ans_insuff_score',
                                                                      'para_score'])  
add_combo_key(fever_hover_only, keys=['src', 'ans', 'ans_pred'], newkeyname='src_ans_predans')
create_grouped_metrics(fever_hover_only, group_key='src_ans_predans', metric_keys = ['answer_em', 'sp_em', 'sp_recall', 
                                                                      'para_acc', 'ev_acc', 
                                                                      'ans_pred_score', 'ans_insuff_score',
                                                                      'para_score'])

fever_hover_singlesent = filter_fever(fever_hover_only)
create_grouped_metrics(fever_hover_singlesent, group_key='src_ans', metric_keys = ['answer_em', 'sp_em', 'sp_recall', 
                                                                      'para_acc', 'ev_acc', 
                                                                      'ans_pred_score', 'ans_insuff_score',
                                                                      'para_score'])


create_grouped_metrics(samples, group_key='src')
#create_grouped_metrics(samples, group_key='para_pred', metric_keys = ['answer_em', 'sp_em', 'sp_recall', 
#                                                                      'para_acc', 'ev_acc', 
#                                                                      'ans_pred_score', 'ans_insuff_score',
#                                                                      'para_score'])

add_combo_metrics(samples, ans_diff_thresh=1.0, para_pos_conf_thresh = 1.0) #\
create_grouped_metrics(samples, group_key='calc_para_pred', metric_keys = ['answer_em', 'sp_em', 'sp_recall', 
                                                                      'para_acc', 'ev_acc', 'calc_para_acc',
                                                                      'ans_pred_score', 'ans_insuff_score',
                                                                      'para_score'])







