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

import utils


infiles = ['/large_data/thar011/out/mdr/logs/UQA_s9_v2_numlit_wikissvise_idt_errsamp_fixdecode/log-train-history.jsonl',
           '/data/thar011/out/unifiedqa_averages/s11/t5large-partial-log-train-history.jsonl']
desclist = ['s9_v2_bart', 's9_v5_t5lge']

step_key = 'curr_global_step'
#devset_keys = ['em_narrativeqa', 'em_ai2_science_middle', 'em_ai2_science_elementary',
#               'em_arc_hard', 'em_arc_easy', 'em_mctest_corrected_the_separator',
#               'em_squad1_1', 'em_squad2', 'em_boolq', 'em_race_string', 'em_openbookqa',
#               'em_strategy_qa', 'em_strategy_qa_facts_selfsvised', 
#               'em_strategy_qa_facts_dev_in_train_selfsvised']
#disp_labels = ['NQA', 'SCI_M', 'SCI_E', 'ARC_Hard', 'ARC_Easy', 'MCTest', 
#               'SQUAD1_1', 'SQUAD2', 'BoolQ', 'RACE', 'OBQA', 'SQA', 
#               'SQA_Facts', 'SQA_Facts']


def get_em_keys():
    """ The actual metrics are in keys starting with 'em_'
    """
    all_json_list1 = utils.load_jsonl(infiles[0])    
    keys = [k for k in all_json_list1[0].keys() if k.startswith('em_')]
    return keys

devset_keys = get_em_keys()
disp_labels = devset_keys
print(f"Metrics: {devset_keys}")


def grab_metrics(all_json_list, step_key, devset_keys, disp_labels):
    """ Return list of global steps, list of metric labels (devset_keys eg em_squad2) 
        and list of lists of metric values
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


def plot_metrics(steps, labels, values, desc, selected_labels, bbox_to_anchor=(1.35, 1), loc="upper right"):
    """ Plot with step on x axis and Acc of each metric/model combo on y as lines
    """
    if type(selected_labels) != list:
        selected_labels = [selected_labels]
    for i, label in enumerate(labels):
        if label in selected_labels or selected_labels == ['ALL']:
            plt.plot(steps, values[i], label=label+' '+ desc)  # plt.plot(x, y, series label)
    plt.title("Dev EM")
    plt.xlabel("Step")
    plt.ylabel("Acc.")
    plt.grid()
    if bbox_to_anchor is None:
        plt.legend(loc=loc)
    else:
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    plt.show()


def comp_metrics(stepslist, labelslist, valueslist, desclist,
                 selected_labels, bbox_to_anchor=(1.35, 1), loc="best"):
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
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    plt.show()
                

def plot_all(infiles, desclist, selected_labels=['ALL'], bbox_to_anchor=(2,1), loc='best'):
    """ Plot metrics for individual models then combined..
    """
    all_jsons = [] 
    all_steps = []
    all_labels = []
    all_values = []               
    
    for i, infile in enumerate(infiles):
        all_json_list = utils.load_jsonl(infile)
        all_jsons.append(all_json_list)
        steps, labels, values = grab_metrics(all_json_list, step_key, devset_keys, disp_labels)
        all_steps.append(steps)
        all_labels.append(labels)
        all_values.append(values)
        plot_metrics(steps, labels, values, desclist[i], selected_labels=selected_labels, bbox_to_anchor=bbox_to_anchor, loc=loc)
    comp_metrics(all_steps, all_labels, all_values, desclist, selected_labels=selected_labels, bbox_to_anchor=bbox_to_anchor, loc=loc)
    return



selected_labels=[   'em_enwiki_20200801_selfsvised', 
                    'em_synthetic_num_signed_arith', 
                    'em_poetsql_select_count', 
                    'em_tt_composition_2_hop',
                    'em_tt_numeric_superlatives']

plot_all(infiles, desclist, selected_labels=selected_labels, bbox_to_anchor=(2,1), loc='best')

# plot selected metrics for single model:
all_json_list1 = utils.load_jsonl(infiles[0])
steps1, labels1, values1 = grab_metrics(all_json_list1, step_key, devset_keys, disp_labels)
plot_metrics(steps1, labels1, values1, desclist[0], selected_labels=['ALL'], bbox_to_anchor=(2, 1), loc="best")

plot_metrics(steps1, labels1, values1, desclist[0], 
             selected_labels=selected_labels, 
             bbox_to_anchor=(2, 1), loc="best")


"""
all_json_list1 = utils.load_jsonl(infiles[0])
print(all_json_list1[0].keys())
steps1, labels1, values1 = grab_metrics(all_json_list1, step_key, devset_keys, disp_labels)
#plot_metrics(steps1, labels1, values1, desclist[0], selected_labels=['ALL'], bbox_to_anchor=(1.35, 1), loc="upper right")
plot_metrics(steps1, labels1, values1, desclist[0], selected_labels=['ALL'], bbox_to_anchor=(2, 1), loc="best")
#plot_metrics(steps1, labels1, values1, desclist[0], selected_labels=['ALL'], bbox_to_anchor=None, loc="best")
#plot_metrics(steps1, labels1, values1, selected_labels=['SQA', 'SQA_Facts'])

all_json_list2 = load_jsonl(infiles[1])
steps2, labels2, values2 = grab_metrics(all_json_list2, step_key, devset_keys, disp_labels)
plot_metrics(steps2, labels2, values2, selected_labels=['SQA', 'SQA_Facts'])

comp_metrics([steps1,steps2],[labels1,labels2],[values1,values2],['v1', 'v2'],selected_labels=['SQA', 'SQA_Facts'])
"""





