#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:08:55 2021

@author: timhartill

Plot dev accuracies over training

Modify infiles, desclist, devset_keys and disp_labels and selected_labels as needed

"""

import matplotlib.pyplot as plt
import numpy as np
import json


infiles = ['/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v1/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v2_dev_in_train/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v3_no_facts/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v4_ctr_dev_in_train/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v5__ssm1_masktokonly_dev_in_train/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_bart_large_V3_run2/log-train-history.jsonl']
desclist = ['v1 no_de_in_tr m_', 
            'v2 de_in_tr m_',
            'v3 no_facts m_',
            'v4 de_in_tr m_ctr',
            'v5 de_in_tr m ssm_only',
            'v0 no sqa']

step_key = 'curr_global_step'
devset_keys = ['em_narrativeqa', 'em_ai2_science_middle', 'em_ai2_science_elementary',
               'em_arc_hard', 'em_arc_easy', 'em_mctest_corrected_the_separator',
               'em_squad1_1', 'em_squad2', 'em_boolq', 'em_race_string', 'em_openbookqa',
               'em_strategy_qa', 'em_strategy_qa_facts_selfsvised', 
               'em_strategy_qa_facts_dev_in_train_selfsvised']
disp_labels = ['NQA', 'SCI_M', 'SCI_E', 'ARC_Hard', 'ARC_Easy', 'MCTest', 
               'SQUAD1_1', 'SQUAD2', 'BoolQ', 'RACE', 'OBQA', 'SQA', 
               'SQA_Facts', 'SQA_Facts']


def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    num_jsons = len(all_json_list)
    if verbose:
        print('JSON as text successfully loaded. Number of json messages in file is ', num_jsons)
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print('Text successfully converted to JSON.')
    return all_json_list


def grab_metrics(all_json_list, step_key, devset_keys, disp_labels):
    """ Return list of global steps, list of metric labels and list of lists of metric values
    """
    steps = []
    labels = [d for d in devset_keys if all_json_list[0].get(d) is not None]
    values = [ [] for l in labels ]
    for ts in all_json_list:
        steps.append( ts[step_key] )
        for i, k in enumerate( labels ):
            if ts.get(k) is not None:
                values[i].append( ts[k] )
    labels = [disp_labels[i] for i, d in enumerate(devset_keys) if d in labels  ]
    return steps, labels, values


def plot_metrics(steps, labels, values, desc, selected_labels, bbox_to_anchor=(1.35, 1)):
    if type(selected_labels) != list:
        selected_labels = [selected_labels]
    for i, label in enumerate(labels):
        if label in selected_labels or selected_labels == ['ALL']:
            plt.plot(steps, values[i], label=label+' '+ desc)  # plt.plot(x, y, series label)
    plt.title("Dev EM")
    plt.xlabel("Step")
    plt.ylabel("Acc.")
    plt.grid()
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc="upper right")
    plt.show()


def comp_metrics(stepslist, labelslist, valueslist, desclist,
                 selected_labels, bbox_to_anchor=(1.35, 1)):
    if type(selected_labels) != list:
        selected_labels = [selected_labels]    
    for j, labels in enumerate(labelslist):
        for i, label in enumerate(labels):
            if label in selected_labels or selected_labels == ['ALL']:
                plt.plot(stepslist[j], valueslist[j][i], label=label+' '+ desclist[j])  # plt.plot(x, y, series label)
    plt.title("Dev EM")
    plt.xlabel("Step")
    plt.ylabel("Acc.")
    plt.grid()
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc="upper right")
    plt.show()
                

all_jsons = [] 
all_steps = []
all_labels = []
all_values = []               

for i, infile in enumerate(infiles):
    all_json_list = load_jsonl(infile)
    all_jsons.append(all_json_list)
    steps, labels, values = grab_metrics(all_json_list, step_key, devset_keys, disp_labels)
    all_steps.append(steps)
    all_labels.append(labels)
    all_values.append(values)
    plot_metrics(steps, labels, values, desclist[i], selected_labels=['SQA', 'SQA_Facts'], bbox_to_anchor=(1.65, 1.0))
comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['SQA', 'SQA_Facts'], bbox_to_anchor=(1.65, 1.0))
comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['SQA_Facts'], bbox_to_anchor=(1.65, 1.0))
comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['SQA'], bbox_to_anchor=(1.65, 1.0))

comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['ARC_Hard'], bbox_to_anchor=(1.65, 1.0))
comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['SQUAD1_1'], bbox_to_anchor=(1.65, 1.0))
comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=['OBQA'], bbox_to_anchor=(1.65, 1.0))


"""
all_json_list1 = load_jsonl(infiles[0])
steps1, labels1, values1 = grab_metrics(all_json_list1, step_key, devset_keys, disp_labels)
plot_metrics(steps1, labels1, values1, selected_labels=['SQA', 'SQA_Facts'])

all_json_list2 = load_jsonl(infiles[1])
steps2, labels2, values2 = grab_metrics(all_json_list2, step_key, devset_keys, disp_labels)
plot_metrics(steps2, labels2, values2, selected_labels=['SQA', 'SQA_Facts'])

comp_metrics([steps1,steps2],[labels1,labels2],[values1,values2],['v1', 'v2'],selected_labels=['SQA', 'SQA_Facts'])
"""





