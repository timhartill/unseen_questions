#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:01:55 2021

@author: Tim Hartill

Evaluation metrics calculation utilities and dataset attributes
Also contains the function to produce cross tables comparing prediction performance
across models. This functionality is designed to be run interactively rather than from the command line. 
To generate tables comparing different model runs, see the instructions in the docstring 
for the OutputResults class below and/or run fn: run_all(logdir, results_list).

During training all validation is on EM

For final eval datasets we use the metrics "most commonly used" for that dataset as per UnifiedQA Paper, specifically:
Dataset type:
    EX: F1
    AB: F1 or rougeL except NatQA uses EM
    MC: Select option with highest F1 score between prediction and each option
    YN: YN calc determines prediction
"""
import numpy as np
import re
import string
import collections
import json
import os
import copy

import datasets



#Map dataset types to relevant metrics to calculate and preferred reporting metric
metric_groups = {
    'EX': {'compute':['EM', 'F1'], 'prefer':'F1'},
    'AB': {'compute':['EM', 'F1', 'RL'], 'prefer':'RL'},
    'MC': {'compute':['EM', 'F1', 'SS'], 'prefer':'SS'},
    'YN': {'compute':['EM', 'F1', 'YN'], 'prefer':'YN'}
}


#Map datasets to dataset types and optional override for preferred reporting metric (must be one in above 'compute' key)
dataset_attribs = {
    'ai2_science_elementary': {'type':'MC', 'prefer':''},
    'ai2_science_middle': {'type':'MC', 'prefer':''},
    'ambigqa': {'type':'AB', 'prefer':''},
    'ambigqa_dedup': {'type':'AB', 'prefer':''},
    'arc_easy': {'type':'MC', 'prefer':''},
    'arc_easy_dev': {'type':'MC', 'prefer':''},
    'arc_easy_with_ir': {'type':'MC', 'prefer':''},
    'arc_easy_with_ir_dev': {'type':'MC', 'prefer':''},
    'arc_hard': {'type':'MC', 'prefer':''},
    'arc_hard_dev': {'type':'MC', 'prefer':''},
    'arc_hard_with_ir': {'type':'MC', 'prefer':''},
    'arc_hard_with_ir_dev': {'type':'MC', 'prefer':''},
    'boolq': {'type':'YN', 'prefer':''},
    'boolq_np': {'type':'YN', 'prefer':''},
    'boolq_np_dedup': {'type':'YN', 'prefer':''},
    'commonsenseqa': {'type':'MC', 'prefer':''},
    'commonsenseqa_test': {'type':'MC', 'prefer':''},
    'contrast_sets_boolq': {'type':'YN', 'prefer':''},
    'contrast_sets_boolq_dedup': {'type':'YN', 'prefer':''},
    'contrast_sets_drop': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_quoref': {'type':'EX', 'prefer':''},
    'contrast_sets_quoref_dedup': {'type':'EX', 'prefer':''},
    'contrast_sets_ropes': {'type':'EX', 'prefer':''},
    'contrast_sets_ropes_dedup': {'type':'EX', 'prefer':''},
    'drop': {'type':'AB', 'prefer':'F1'},
    'drop_dedup': {'type':'AB', 'prefer':'F1'},
    'mctest': {'type':'MC', 'prefer':''},
    'mctest_corrected_the_separator': {'type':'MC', 'prefer':''},
    'multirc': {'type':'YN', 'prefer':''},
    'narrativeqa': {'type':'AB', 'prefer':''},
    'narrativeqa_dev': {'type':'AB', 'prefer':''},
    'natural_questions': {'type':'AB', 'prefer':'EM'},
    'natural_questions_direct_ans': {'type':'AB', 'prefer':'EM'},
    'natural_questions_direct_ans_test': {'type':'AB', 'prefer':'EM'},
    'natural_questions_with_dpr_para': {'type':'AB', 'prefer':'EM'},
    'natural_questions_with_dpr_para_test': {'type':'AB', 'prefer':'EM'},
    'newsqa': {'type':'EX', 'prefer':''},
    'openbookqa': {'type':'MC', 'prefer':''},
    'openbookqa_dev': {'type':'MC', 'prefer':''},
    'openbookqa_with_ir': {'type':'MC', 'prefer':''},
    'openbookqa_with_ir_dev': {'type':'MC', 'prefer':''},
    'physical_iqa': {'type':'MC', 'prefer':''},
    'qasc': {'type':'MC', 'prefer':''},
    'qasc_test': {'type':'MC', 'prefer':''},
    'qasc_with_ir': {'type':'MC', 'prefer':''},
    'qasc_with_ir_test': {'type':'MC', 'prefer':''},
    'quoref': {'type':'EX', 'prefer':''},
    'quoref_dedup': {'type':'EX', 'prefer':''},
    'race_string': {'type':'MC', 'prefer':''},
    'race_string_dev': {'type':'MC', 'prefer':''},
    'ropes': {'type':'EX', 'prefer':''},
    'social_iqa': {'type':'MC', 'prefer':''},
    'social_iqa_dedup': {'type':'MC', 'prefer':''},
    'squad1_1': {'type':'EX', 'prefer':''},
    'squad2': {'type':'EX', 'prefer':''},
    'winogrande_l': {'type':'MC', 'prefer':''},
    'winogrande_m': {'type':'MC', 'prefer':''},
    'winogrande_s': {'type':'MC', 'prefer':''},
    'winogrande_test': {'type':'MC', 'prefer':''},
    'winogrande_xl': {'type':'MC', 'prefer':''},
    'winogrande_xs': {'type':'MC', 'prefer':''},
    'mmlu_elementary_mathematics_test': {'type':'MC', 'prefer':''},
    'mmlu_elementary_mathematics_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_business_ethics_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_accounting_test': {'type':'MC', 'prefer':''},
     'mmlu_college_mathematics_test': {'type':'MC', 'prefer':''},
     'mmlu_public_relations_test': {'type':'MC', 'prefer':''},
     'mmlu_public_relations_test_dedup': {'type':'MC', 'prefer':''}, 
     'mmlu_philosophy_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_government_and_politics_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_medicine_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_biology_test': {'type':'MC', 'prefer':''},
     'mmlu_moral_disputes_test': {'type':'MC', 'prefer':''},
     'mmlu_moral_scenarios_test': {'type':'MC', 'prefer':''},
     'mmlu_clinical_knowledge_test': {'type':'MC', 'prefer':''},
     'mmlu_college_computer_science_test': {'type':'MC', 'prefer':''},
     'mmlu_jurisprudence_test': {'type':'MC', 'prefer':''},
     'mmlu_logical_fallacies_test': {'type':'MC', 'prefer':''},   
     'mmlu_us_foreign_policy_test': {'type':'MC', 'prefer':''},
     'mmlu_us_foreign_policy_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_high_school_statistics_test': {'type':'MC', 'prefer':''},
     'mmlu_virology_test': {'type':'MC', 'prefer':''},
     'mmlu_formal_logic_test': {'type':'MC', 'prefer':''},
     'mmlu_security_studies_test': {'type':'MC', 'prefer':''},
     'mmlu_machine_learning_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_us_history_test': {'type':'MC', 'prefer':''},
     'mmlu_world_religions_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_chemistry_test': {'type':'MC', 'prefer':''},
     'mmlu_prehistory_test': {'type':'MC', 'prefer':''},
     'mmlu_electrical_engineering_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_european_history_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_psychology_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_psychology_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_high_school_world_history_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_geography_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_computer_science_test': {'type':'MC', 'prefer':''},
     'mmlu_human_aging_test': {'type':'MC', 'prefer':''},
     'mmlu_marketing_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_mathematics_test': {'type':'MC', 'prefer':''},
     'mmlu_conceptual_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_abstract_algebra_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_psychology_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_psychology_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_management_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_macroeconomics_test': {'type':'MC', 'prefer':''},
     'mmlu_sociology_test': {'type':'MC', 'prefer':''},
     'mmlu_nutrition_test': {'type':'MC', 'prefer':''},
     'mmlu_college_biology_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_law_test': {'type':'MC', 'prefer':''},
     'mmlu_astronomy_test': {'type':'MC', 'prefer':''},
     'mmlu_college_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_physics_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_miscellaneous_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_microeconomics_test': {'type':'MC', 'prefer':''},
     'mmlu_computer_security_test': {'type':'MC', 'prefer':''},
     'mmlu_international_law_test': {'type':'MC', 'prefer':''},
     'mmlu_global_facts_test': {'type':'MC', 'prefer':''},
     'mmlu_human_sexuality_test': {'type':'MC', 'prefer':''},
     'mmlu_econometrics_test': {'type':'MC', 'prefer':''},
     'mmlu_anatomy_test': {'type':'MC', 'prefer':''},
     'mmlu_medical_genetics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_medicine_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_chemistry_test': {'type':'MC', 'prefer':''},
     'mmlu_elementary_to_college_math_test': {'type':'MC', 'prefer':''},
     'synthetic_numeric': {'type':'AB', 'prefer':'EM'},
     'synthetic_textual': {'type':'AB', 'prefer':'EM'},
    'drop_dedup_lowsim_uqa': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup_lowsim_uqa': {'type':'AB', 'prefer':'F1'},
    'mmlu_elementary_to_college_math_test_lowsim_uqa': {'type':'MC', 'prefer':''},
    'physical_iqa_lowsim_uqa': {'type':'MC', 'prefer':''},
    'social_iqa_dedup_lowsim_uqa': {'type':'MC', 'prefer':''},
    'commonsenseqa_lowsim_uqa': {'type':'MC', 'prefer':''},
    'qasc_lowsim_uqa': {'type':'MC', 'prefer':''},
    'qasc_with_ir_lowsim_uqa': {'type':'MC', 'prefer':''},
    'ropes_lowsim_uqa': {'type':'EX', 'prefer':''},
    'newsqa_lowsim_uqa': {'type':'EX', 'prefer':''},
    'drop_dedup_lowsim_tdnd': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup_lowsim_tdnd': {'type':'AB', 'prefer':'F1'},
    'mmlu_elementary_to_college_math_test_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'physical_iqa_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'social_iqa_dedup_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'commonsenseqa_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'qasc_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'qasc_with_ir_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'ropes_lowsim_tdnd': {'type':'EX', 'prefer':''},
    'newsqa_lowsim_tdnd': {'type':'EX', 'prefer':''},
    'strategy_qa': {'type':'MC', 'prefer':''},
    'cwwv': {'type':'MC', 'prefer':''},
    'cskg': {'type':'MC', 'prefer':''},
    'atomic': {'type':'MC', 'prefer':''}
    }

unifiedqa_base_train = ["narrativeqa", "ai2_science_middle", "ai2_science_elementary",
                        "arc_hard", "arc_easy", "mctest_corrected_the_separator",
                        "squad1_1", "squad2", "boolq", "race_string", "openbookqa"]

# where same dataset in difft formats, just calc sim against one format and map similarity for others against that...
replace_sim_with = {'cwwv_selfsvised': 'cwwv', 
                'atomic_selfsvised': 'atomic', 
                'cwwv_premask_selfsvised': 'cwwv', 
                'atomic_premask_selfsvised': 'atomic'}

def replace_sim(datasets, mixture_file_key):
    """ Where same dataset in difft formats, just calc sim against one format and map similarity for others against that...
    """
    new_datasets = []
    for ds in datasets:
        if replace_sim_with.get(ds) is not None:
            new_datasets.append(replace_sim_with[ds])
            mixture_file_key = mixture_file_key.replace(ds, replace_sim_with[ds], 1)  # ds should be unique in string...
        else:
            new_datasets.append(ds)
    return new_datasets, mixture_file_key
            

def parse_mixture(mixture):
    """ Parse args.mixture and return list of datasets to include plus a key to add 
        to the pretokenised file name.
        args.mixture format: --mixture unifiedqa,extradataset1,extradataset2
    """
    unified_dataset  = []
    mixture_file_key = ''
    mixturelist = mixture.split(',')
    for ds in mixturelist:
        mixture_file_key = mixture_file_key + '_' + ds
        if ds == 'unifiedqa':
            unified_dataset.extend(unifiedqa_base_train)
        else:
            unified_dataset.append(ds)
    return unified_dataset, mixture_file_key


    
# Not used
unifiedqa_unseen_1 = [
    'newsqa',
    'quoref',
    'contrast_sets_quoref',
    'ropes',
    'contrast_sets_ropes',
    'drop',
    'contrast_sets_drop',
    'qasc',
    'commonsenseqa',
    'boolq_np',
    'contrast_sets_boolq',
    'multirc'
    ]   

# Not used
unifiedqa_unseen_2 = [
    'qasc_with_ir',
    'winogrande_xl',
    'physical_iqa',
    'social_iqa',
    'natural_questions_with_dpr_para'
    ]

# Not used
unifiedqa_unseen_3 = [
    'drop_dedup',
    'contrast_sets_drop_dedup',
    'mmlu_elementary_mathematics_test_dedup',
    'mmlu_high_school_mathematics_test',
    'mmlu_high_school_statistics_test',
    'mmlu_college_mathematics_test',
    'physical_iqa',
    'social_iqa_dedup',
    'commonsenseqa',
    'qasc',
    'qasc_with_ir',
    'ropes',
    'newsqa'
    ]

# The 10 unseen evaluation datasets used in our paper
unifiedqa_unseen_4 = [
    'drop_dedup',
    'contrast_sets_drop_dedup',
    'mmlu_elementary_to_college_math_test',
    'physical_iqa',
    'social_iqa_dedup',
    'commonsenseqa',
    'qasc',
    'qasc_with_ir',
    'ropes',
    'newsqa',
    'strategy_qa'
    ]

unifiedqa_unseen_4_map = {
    'drop_dedup': 'dev.tsv',
    'contrast_sets_drop_dedup': 'dev.tsv',
    'mmlu_elementary_to_college_math_test': 'test.tsv',
    'physical_iqa': 'dev.tsv',
    'social_iqa_dedup': 'dev.tsv',
    'commonsenseqa': 'dev.tsv',
    'qasc': 'dev.tsv',
    'qasc_with_ir': 'dev.tsv',
    'ropes': 'dev.tsv',
    'newsqa': 'dev.tsv',
    'strategy_qa': 'dev.tsv'
    }

# Not used
unifiedqa_unseen_5 = [
    'drop_dedup_lowsim_uqa',
    'contrast_sets_drop_dedup_lowsim_uqa',
    'mmlu_elementary_to_college_math_test_lowsim_uqa',
    'physical_iqa_lowsim_uqa',
    'social_iqa_dedup_lowsim_uqa',
    'commonsenseqa_lowsim_uqa',
    'qasc_lowsim_uqa',
    'qasc_with_ir_lowsim_uqa',
    'ropes_lowsim_uqa',
    'newsqa_lowsim_uqa'
    ]

# The filtered versions of the 10 unseen datasets used in our paper
unifiedqa_unseen_6 = [
    'drop_dedup_lowsim_tdnd',
    'contrast_sets_drop_dedup_lowsim_tdnd',
    'mmlu_elementary_to_college_math_test_lowsim_tdnd',
    'physical_iqa_lowsim_tdnd',
    'social_iqa_dedup_lowsim_tdnd',
    'commonsenseqa_lowsim_tdnd',
    'qasc_lowsim_tdnd',
    'qasc_with_ir_lowsim_tdnd',
    'ropes_lowsim_tdnd',
    'newsqa_lowsim_tdnd'
    ]


# datasets unifiedqa trained on (only the non-IR versions but treating w/IR as "seen") 
# but only those with a separate labelled test set
unifiedqa_seen_1 = [
    'openbookqa',
    'openbookqa_with_ir',
    'arc_easy',
    'arc_easy_with_ir',
    'arc_hard',
    'arc_hard_with_ir',
    'race_string',
    'ai2_science_elementary',
    'ai2_science_middle',
    'strategy_qa',
    'cwwv',
    'atomic'
    ]

# The 57 mmlu datasets. 
mmlu_unseen_1 = [
     'mmlu_elementary_mathematics_test',
     'mmlu_business_ethics_test',
     'mmlu_professional_accounting_test',
     'mmlu_college_mathematics_test',
     'mmlu_public_relations_test',
     'mmlu_philosophy_test',
     'mmlu_high_school_government_and_politics_test',
     'mmlu_professional_medicine_test',
     'mmlu_high_school_biology_test',
     'mmlu_moral_disputes_test',
     'mmlu_moral_scenarios_test',
     'mmlu_clinical_knowledge_test',
     'mmlu_college_computer_science_test',
     'mmlu_jurisprudence_test',
     'mmlu_logical_fallacies_test',
     'mmlu_us_foreign_policy_test',
     'mmlu_high_school_statistics_test',
     'mmlu_virology_test',
     'mmlu_formal_logic_test',
     'mmlu_security_studies_test',
     'mmlu_machine_learning_test',
     'mmlu_high_school_us_history_test',
     'mmlu_world_religions_test',
     'mmlu_high_school_chemistry_test',
     'mmlu_prehistory_test',
     'mmlu_electrical_engineering_test',
     'mmlu_high_school_european_history_test',
     'mmlu_high_school_psychology_test',
     'mmlu_high_school_world_history_test',
     'mmlu_high_school_geography_test',
     'mmlu_high_school_computer_science_test',
     'mmlu_human_aging_test',
     'mmlu_marketing_test',
     'mmlu_high_school_mathematics_test',
     'mmlu_conceptual_physics_test',
     'mmlu_abstract_algebra_test',
     'mmlu_professional_psychology_test',
     'mmlu_management_test',
     'mmlu_high_school_macroeconomics_test',
     'mmlu_sociology_test',
     'mmlu_nutrition_test',
     'mmlu_college_biology_test',
     'mmlu_professional_law_test',
     'mmlu_astronomy_test',
     'mmlu_college_physics_test',
     'mmlu_miscellaneous_test',
     'mmlu_high_school_microeconomics_test',
     'mmlu_computer_security_test',
     'mmlu_international_law_test',
     'mmlu_global_facts_test',
     'mmlu_human_sexuality_test',
     'mmlu_econometrics_test',
     'mmlu_anatomy_test',
     'mmlu_medical_genetics_test',
     'mmlu_college_medicine_test',
     'mmlu_high_school_physics_test',
     'mmlu_college_chemistry_test']


# the standard "squad" normalization
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        exclude.add('\u2047')  #TJH Added from unifiedqa solver.py
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def replace_punctuation(instr):
    return instr.replace("\"", "").replace("'", "")


## from unifiedqa solver.py
# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(instr):
    return re.sub("[{}^\\\\`\u2047<]", " ", instr)


# adapted from unifiedqa solver.py
def score_string_similarity(str1, str2, usesolver_preproc=False, use_f1=True):
    f1_str1 = str1
    f1_str2 = str2
    
    if str1 == str2:
        return 3.0  # Better than perfect token match
    if usesolver_preproc:
        str1 = fix_buggy_characters(replace_punctuation(str1))
        str2 = fix_buggy_characters(replace_punctuation(str2))
    else:  # empirically normalize_answer usually yields slightly higher scores than the solver preproc
        str1 = normalize_answer(str1)
        str2 = normalize_answer(str2)
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        if not use_f1:
            str1_split = str1.split(" ")
            str2_split = str2.split(" ")
            overlap = list(set(str1_split) & set(str2_split))  #TJH: This part does the same as num_same calc in f1
            return len(overlap) / max(len(str1_split), len(str2_split))
        else:
            return get_f1(f1_str1, f1_str2)  #TJH: Empirically calculating overlap with F1 does very slightly better
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


# from https://github.com/huggingface/datasets/blob/86e66e7be32f96a625314b8e7d4b16d703eba82d/metrics/squad_v2/evaluate.py#L104
def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


# from https://github.com/huggingface/datasets/blob/86e66e7be32f96a625314b8e7d4b16d703eba82d/metrics/squad_v2/evaluate.py#L104
def get_f1(prediction, groundtruth):
    gold_toks = get_tokens(groundtruth)
    pred_toks = get_tokens(prediction)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_yn(prediction, groundtruth):
    """ # yes/no question scoring
    """
    gt = normalize_answer(groundtruth)
    if gt == 'yes':
        opposite_label = 'no'
    else: 
        opposite_label = 'yes'
    pred = get_tokens(prediction)
    if gt in pred and opposite_label not in pred:
        yn = True
    else:
        yn = False
    return yn


class StringSimilarity:
    """ Return MC string similarity accuracy metric
    Unlike other metrics we must first parse the questions and determine what the possible answer choices are.
    Then we score the similarity of the prediction with each answer choice and prediction will be the choice with highest similarity
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text']
        questions = ['q1 text', 'q2 text']
    """
    def __init__(self):
        self.last_scores = {}
        self.simscores = []
        self.choices = []  # list of choices parsed from the questions
        self.newpreds = [] # list of the new predictions calculated from the similarity score

    def compute_metric(self, predictions, groundtruths, questions=[], usesolver_preproc=False, use_f1=False):
        if not self.choices:
            self.parse_questions(questions)
        self.newpreds = []
        self.simscores = []
        for i, (prediction, groundtruth) in enumerate(zip(predictions, groundtruths)):
            scores = [score_string_similarity(x, prediction.lower(), usesolver_preproc=usesolver_preproc, use_f1=use_f1) for x in self.choices[i]]
            max_idx = np.argmax(scores)
            self.newpreds.append(self.choices[i][max_idx])
            em = get_exact_match(self.choices[i][max_idx], groundtruth)
            self.simscores.append(em)
        ss = (100.0 * sum(self.simscores)) / len(self.simscores)
        self.last_scores['ss'] = ss
        return ss
                    
    def parse_questions(self, questions):  
        self.choices = []
        choicestr = "abcdefghijklmnopqrstuvwxyz"   #"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        keylist = ['(' + c + ')' for c in choicestr]
        qsplit = [q.split('\\n')[1] for q in questions]  # choices always follow 1st \n
        qsplit = [q.lower().split() for q in qsplit]  # each (A) , (B) etc must be followed by <space> for this to work..
        for qchoices in qsplit:
            currchoices = []
            currchoice = ''
            for tok in qchoices:
                currtok = tok.strip()
                if currtok not in keylist:
                    currchoice = currchoice + ' ' + currtok
                else:
                    if currchoice.strip() != '': 
                        currchoices.append(currchoice.strip())
                        currchoice = '' 
            if currchoice:        
                currchoices.append(currchoice.strip())  
            self.choices.append(currchoices)                  


class F1:
    """ Return f1 metric as defined for squad 
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text']
    """
    def __init__(self):
        self.last_scores = {}
        self.f1s = []
        
    def compute_metric(self, predictions, groundtruths):
        self.f1s = []
        for prediction, groundtruth in zip(predictions, groundtruths):
            f1 = get_f1(prediction, groundtruth)
            self.f1s.append(f1)
        f1 = (100.0 * sum(self.f1s)) / len(self.f1s)
        self.last_scores['f1'] = f1
        return f1


class EM:
    """ Return em accuracy metric
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text']
    """
    def __init__(self):
        self.last_scores = {}
        self.ems = []
        
    def compute_metric(self, predictions, groundtruths):
        self.ems = []
        for prediction, groundtruth in zip(predictions, groundtruths):
            em = get_exact_match(prediction, groundtruth)
            self.ems.append(em)
        em = (100.0 * sum(self.ems)) / len(self.ems)
        self.last_scores['em'] = em
        return em


class Rouge:
    """ Return rougeL metric
    Usage: rscorer = Rouge()
    rougeLscore = rscorer.compute_metric(predictions, groundtruth) 
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text']
    """
    def __init__(self):
        self.rouge = datasets.load_metric('rouge')
        self.last_scores = None
        
    def compute_metric(self, predictions, groundtruths, norm=False):
        if norm:
            predictions_norm = [normalize_answer(p) for p in predictions]
            groundtruths_norm = [normalize_answer(gt) for gt in groundtruths]
        else:   #Note: empirically rougel slightly better WITHOUT normalising
            predictions_norm = predictions
            groundtruths_norm = groundtruths
        self.last_scores = self.rouge.compute(predictions=predictions_norm, references=groundtruths_norm) 
        return self.last_scores["rougeL"].mid.fmeasure * 100.0
        

class YN:
    """ Return accuracy metric for yes/no datasets (label is always yes or no)
    Per unifiedqa paper if prediction is single word then exact match otherwise
    for multi-word prediction, correct means the label is in the prediction AND the opposite class isnt
    """
    def __init__(self):
        self.last_scores = {}
        self.yn= []
        
    def compute_metric(self, predictions, groundtruths):
        for prediction, groundtruth in zip(predictions, groundtruths):
            yn = get_yn(prediction, groundtruth)
            self.yn.append(yn)
        yn = (100.0 * sum(self.yn)) / len(self.yn)
        self.last_scores['yn'] = yn
        return yn


class DatasetMetrics:
    """ Utilities for opening/viewing the json file of results for a single model with the above metrics
    json file format:
        {'dataset_name': {'prefer': 'preferred output metric eg 'SS' at the time the file was created' , 
                          'type: ': 'dataset type one of EX AB MC YN', 
                          'comp_metrics': [metrics actually appearing below subset of: 'EM', 'F1', 'RL', 'SS', 'YN'], 
                          'eval_file_type': 'dev' or 'test', 
                          'gt_file': 'eg '/data/thar011/data/unifiedqa/newsqa/dev.tsv'', 
                          'gt_file_tokenized': eg 'test-uncased-xbos-BartTokenizedFast.json',
                          'groundtruths_tokenized': from decoder_input_ids: eg [563, 256, 9, 0, 1, 1, 1],
                          'groundtruths': [actual textual answers: 'gt1', 'gt2'..], 
                          'predictions': [actual texttual preds: 'pred1', 'pred2'..], 
                          'EM': {'score': exact match accuracy over data (* 100), 
                                 'scores': [per pred score - all metrics except RL. NOT * 100], 
                                 'newpreds': [only for MC/SS: list of new predictions], 
                                 'choices': [only for MC/SS: the parsed MC options]}, 
                          'F1': {as per EM - F1 Avg over data}, 
                          'RL': {as per EM - RougeL score over data},
                          'SS': {as per EM - Multichoice Similarity Score Accuracy over data},
                          'YN': {as per EM - Yes/No Accuracy over data}
                          }
         }
         'dataset_name_2'...
         ...
        }
    
    Usage: 
    dsmetrics = DatasetMetrics('/data/outdir/eval_metrics.json')
    
    dsmetrics.get_single_pred_vs_groundtruth(dsmetrics.datasets[0]) # display prediction vs gt plus metrics for a single sample
    

    ds_values = dsmetrics.get_pref_values_set('unifiedqa_unseen_1')  # return list of metric values for a list of datasets
    print(ds_values[0])  # ('newsqa', 54.433245958483546, 'F1')
    
    ds_values = dsmetrics.get_pref_values_set('mmlu_unseen_1') #  # return list of metric values for a list of mmlu datasets
    
    dsmetrics.results_dict['mmlu_college_chemistry_test']['SS']['choices'][0]  #view parsed choices for a MC question
    
    for ds in dsmetrics.results_dict:
    if 'SS' in dsmetrics.results_dict[ds]:
        print(f"{ds}: Number of choices: {len(dsmetrics.results_dict[ds]['SS']['choices'][0])}") #print num of choices for 1st sample in each MC dataset (all mmlu=4)
    
    """
    def __init__(self, results_file):
        self.results_file = results_file
        self.short_name = results_file.split('/')[-2]
        self.results_dict = json.load(open(results_file)) 
        self.datasets = list(self.results_dict.keys())        

        
    def get_value(self, ds, metric, key='score'):
        """ Return a value from the input metric of a particular dataset        
        By default the value will be that metric's score but can be any key eg 'scores', 'newpreds', 'choices' 
        """
        if self.results_dict.get(ds) is not None:
            m = self.results_dict[ds].get(metric, -1)   
            if m != -1:
                retval = m[key]
            else:    
                print(f"No metric {metric} for dataset {ds}. Returning -1 instead" )
                retval = -1
        else:
            print(f"Dataset {ds} does not exist in this results set. Returning -1 instead")
            retval = -1
        return retval
    
    def get_pref_metric(self, ds, use_current=True):
        """ Get preferred metric for this ds
        If use_current then use the currently preferred metric for this dataset as defined by dataset_attribs and metric_groups
        Else use the preferred metric saved in the results file at the time it was created.
        """
        if use_current:
            if dataset_attribs.get(ds) is not None:
                pref_metric = dataset_attribs[ds]['prefer']
                if not pref_metric:
                    dsettype = dataset_attribs[ds]['type']
                    pref_metric = metric_groups[dsettype]['prefer']
            else:
                pref_metric = 'NA'
        else:    
            if self.results_dict.get(ds) is not None:
                pref_metric = self.results_dict[ds]['prefer']
            else:
                pref_metric = 'NA'
        return pref_metric
        
    def get_pref_value(self, ds, key='score', use_current=True):
        """ Return a value from the preferred output metric of a particular dataset
        By default the value will be that metric's score but can be any key eg 'scores', 'newpreds', 'choices' 
        """
        pref_metric = self.get_pref_metric(ds, use_current=use_current)
        return self.get_value(ds, pref_metric, key=key)
    

    def get_single_pred_vs_groundtruth(self, ds, index=0, metric='ALL'):
        """ Return prediction and groundtruth for a single example
        Usage: tst.get_single_pred_vs_groundtruth('arc_hard', index=2, metric='ALL')
        """
        ret_dict = {}
        ret_dict['groundtruth'] = self.results_dict[ds]['groundtruths'][index]
        if self.results_dict[ds].get('groundtruths_tokenized', None) is not None:
            ret_dict['groundtruths_tokenized'] = self.results_dict[ds]['groundtruths_tokenized'][index]
        ret_dict['prediction'] = self.results_dict[ds]['predictions'][index]
        if metric is not None:
            if metric == 'ALL':
                for m in self.results_dict[ds]['comp_metrics']:
                    if m == 'SS':
                        ret_dict[m+'_new_prediction'] = self.results_dict[ds][m]['newpreds'][index]
                        ret_dict[m+'_from_choices'] = self.results_dict[ds][m]['choices'][index]
                    ret_dict[m+'_score'] = self.results_dict[ds][m]['scores'][index] * 100
            else:
                if metric == 'SS':
                    ret_dict['new_prediction'] = self.results_dict[ds][metric]['newpreds'][index]
                    ret_dict['from_choices'] = self.results_dict[ds][metric]['choices'][index]
                ret_dict['score'] = self.results_dict[ds][metric]['scores'][index] * 100
        return ret_dict

                
    def get_all_pref_values(self, dsets, use_current=True):
        """ Return the preferred output metric scores for each dataset in dsets
        dsets: ['dataset1', 'dataset2', ...]
        Returns list of ('datasetname', score, metric name) tuples
        """
        retvals = []        
        for ds in dsets:
            retvals.append( (ds, self.get_pref_value(ds, use_current=use_current), self.get_pref_metric(ds, use_current=use_current) ) )
        return retvals            

    def get_pref_values_set(self, dsetset):
        """ Return the preferred output metric scores for each dataset 
        in a particular set.
        """
        if dsetset == 'unifiedqa_unseen_1':
            retvals = self.get_all_pref_values(unifiedqa_unseen_1)
        elif dsetset == 'unifiedqa_unseen_2':    
            retvals = self.get_all_pref_values(unifiedqa_unseen_2)
        elif dsetset == 'unifiedqa_unseen_3':    
            retvals = self.get_all_pref_values(unifiedqa_unseen_3)
        elif dsetset == 'unifiedqa_unseen_4':    
            retvals = self.get_all_pref_values(unifiedqa_unseen_4)
        elif dsetset == 'unifiedqa_unseen_5':    
            retvals = self.get_all_pref_values(unifiedqa_unseen_5)
        elif dsetset == 'unifiedqa_unseen_6':    
            retvals = self.get_all_pref_values(unifiedqa_unseen_6)
        elif dsetset == 'unifiedqa_seen_1':    
            retvals = self.get_all_pref_values(unifiedqa_seen_1)
        elif dsetset == 'mmlu_unseen_1':
            retvals = self.get_all_pref_values(mmlu_unseen_1)
        else:
            print(f"Unknown dataset set: {dsetset}")
        return retvals
    
    def save_eval_file(self, new_file = ''):
        """ Save json file, optionally under a new name """
        if new_file:
            save_file = new_file
        else:
            save_file = self.results_file
        with open(save_file, 'w') as f:
            json.dump(self.results_dict, f)
        print(f"Saved to {save_file}")
            
    def update_rouge_ss(self):
        """ Early results contained an error in how rougeL was calculated 
        and also used the solver.py overlap calculation instead of f1 for the SS metric
        """
        for ds in self.results_dict:
            result_ds = self.results_dict[ds]
            if 'RL' in result_ds['comp_metrics']:
                scorer = Rouge()
                score = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              norm=False)
                print(f'Dataset: {ds} {result_ds["eval_file_type"]}: RougeL: {score}')
                result_ds['RL'] = {'score': score,
                           'scores': [],
                           'newpreds': [],
                           'choices': []}
            if 'SS' in result_ds['comp_metrics']:
                scorer = StringSimilarity()
                scorer.choices = result_ds['SS']['choices']
                score = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              usesolver_preproc=False, use_f1=True)
                print(f'Dataset: {ds} {result_ds["eval_file_type"]}: SS: {score}')
                result_ds['SS'] = {'score': score,
                           'scores': scorer.simscores,
                           'newpreds': scorer.newpreds,
                           'choices': scorer.choices}
        
    
    def test_rouge_ss(self):
        """ test rougeL with/without normalising prediction first and ss using difft normalisation and overlap approaches
        """
        for ds in self.results_dict:
            result_ds = self.results_dict[ds]
            if 'RL' in result_ds['comp_metrics']:
                scorer = Rouge()
                score_norm = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              norm=True)
                score_nonorm = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              norm=False)
                print(f'Dataset: {ds} {result_ds["eval_file_type"]}: RougeL with norm: {score_norm} without norm:{score_nonorm}')

            if 'SS' in result_ds['comp_metrics']:
                scorer = StringSimilarity()
                scorer.choices = result_ds['SS']['choices']
                score_norm = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              usesolver_preproc=False, use_f1=False)
                score_nonorm = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              usesolver_preproc=True, use_f1=False)
                score_f1 = scorer.compute_metric(predictions=result_ds['predictions'], 
                                              groundtruths=result_ds['groundtruths'],
                                              usesolver_preproc=False, use_f1=True)
                print(f'Dataset: {ds} {result_ds["eval_file_type"]}: SS with norm: {score_norm} without norm (solver preproc instead):{score_nonorm} with F1: {score_f1}')
                            

class OutputResults:
    """ Output results across different models/training runs into a 
    consolidated results table.

    # set up output directory
    logdir = '/data/outdir_unifiedqa_averages/comp3runs046/'
    os.makedirs(logdir, exist_ok=True)

    # to view individual runs:
    results_list = [
    '/data/outdir_v3run0/eval_metrics.json',   
    '/data/outdir_v3run1/eval_metrics.json', 
    '/data/outdir_v3run2/eval_metrics.json',     
    '/data/outdir_v4/eval_metrics.json',  
    '/data/outdir_v5/eval_metrics.json',  
    '/data/outdir_v6/eval_metrics.json',  
    '/data/outdir_v7run0/eval_metrics.json',  
    '/data/outdir_v7run1/eval_metrics.json',      
    '/data/outdir_v7run2/eval_metrics.json'
    ]
    res = OutputResults(results_list, logdir)
    #output crosstab over the above runs for the unseen4 set of datasets:
    res.crosstab_x_tasks(dsetset='unseen4', outname='tmp_eval_across_models_mmluagg_us4.txt') # 'final' set - 10 unrestricted eval datasets

    #To create averaged files:
    results_list = [
    '/data/outdir_v3run0/eval_metrics.json',   
    '/data/outdir_v3run1/eval_metrics.json', 
    '/data/outdir_v3run2/eval_metrics.json'     
    ]
    res = OutputResults(results_list, logdir)
    avg_results_dict = res.create_mean_over_runs(newfile=logdir+'v3_avg3runs_eval_metrics.json')
    

    results_list = [
    '/data/outdir_v7run0/eval_metrics.json',  
    '/data/outdir_v7run1/eval_metrics.json',      
    '/data/outdir_v7run2/eval_metrics.json'
    ]
    res = OutputResults(results_list, logdir)
    avg_results_dict = res.create_mean_over_runs(newfile=logdir+'v7_avg3runs_eval_metrics.json')
    
    
    #create summary report using averaged files:
    results_list = [
    '/data/outdir_unifiedqa_averages/comp3runs046/v3_avg3runs_eval_metrics.json',   
    '/data/outdir_v4/eval_metrics.json',  
    '/data/outdir_v5/eval_metrics.json',  
    '/data/outdir_v6/eval_metrics.json',  
    '/data/outdir_unifiedqa_averages/comp3runs046/v7_avg3runs_eval_metrics.json'  
    ]
    res = OutputResults(results_list, logdir)
    #create crosstab over averaged runs for the unseen4 set of datasets:
    res.crosstab_x_tasks(dsetset='unseen4', outname='tmp_eval_across_models_mmluagg_us4_avg3runs.txt') # 'final' set - 10 unrestricted eval datasets
    
    General Usage: 
        logdir = '/data/thar011/out/unifiedqa_averages/comp3runs046/'  #linux      
        os.makedirs(logdir, exist_ok=True)    
        res = OutputResults(results_list, logdir)
        
        res.display_results()  # quick test
        res.crosstab_x_tasks(dsetset='unseen4', outname='tmp_eval_across_models_mmluagg_us4.txt') # 'final' set - 10 unrestricted eval datasets
        res.crosstab_x_tasks(dsetset='unseen6', outname='tmp_eval_across_models_mmluagg_us6lowsimtdnd.txt') # 'final' set - 10 unrestricted eval datasets        
    """
    def __init__(self, results_list, logdir=''):
        self.logdir = logdir
        self.results_list = results_list
        self.shortnames = [o.split('/')[-2] for o in results_list ]
        self.results_dict = {} # {'outputdir': {'dataset set': [(), (), ..]}}
        self.dataset_metrics = {}  # dict of dataset_metrics objects
        for i, results_file in enumerate(self.results_list):
            print(f'Loading {results_file}')
            out_metrics = DatasetMetrics(results_file)
            self.dataset_metrics[results_file] = out_metrics   # access as eg outlist.dataset_metrics['/data/thar011/out/unifiedqa_bart_large_v3/eval_metrics.json'].results_dict['narrativeqa']['RL']['score']
            unseen1 = out_metrics.get_pref_values_set('unifiedqa_unseen_1')  # [ ('dataset1', pref score, 'metric type' ), (...)]
            unseen2 = out_metrics.get_pref_values_set('unifiedqa_unseen_2')
            unseen3 = out_metrics.get_pref_values_set('unifiedqa_unseen_3')
            unseen4 = out_metrics.get_pref_values_set('unifiedqa_unseen_4')
            unseen5 = out_metrics.get_pref_values_set('unifiedqa_unseen_5')
            unseen6 = out_metrics.get_pref_values_set('unifiedqa_unseen_6')
            seen1 = out_metrics.get_pref_values_set('unifiedqa_seen_1') 
            #seen2 = out_metrics.get_pref_values_set('numeric_seen_1')
            mmlu_unseen1 = out_metrics.get_pref_values_set('mmlu_unseen_1')
            self.results_dict[results_file] = {'unseen1':unseen1,    
                                               'unseen2':unseen2,    
                                               'unseen3':unseen3,
                                               'unseen4':unseen4,  # dedup
                                               'unseen5':unseen5,  # dedup lowsim uqa
                                               'unseen6':unseen6,  # dedup lowsim tdnd
                                               'seen1': seen1,       
                                               'mmlu_unseen1': mmlu_unseen1  #mmlu test datasets
                                               }
            
    def save_all(self):
        """ Save each eval json file after doing some mass update..  """
        for out_dir in self.dataset_metrics:
            print(f"Saving {out_dir} ...")
            ds_metric_obj = self.dataset_metrics[out_dir]
            ds_metric_obj.save_eval_file()
        print('Finished saving eval json files.')
        

    def update_all_rouge_ss(self):
        """ Update rouge and ss metrics for each output dir """
        for out_dir in self.dataset_metrics:
            print(f"Updating {out_dir} ...")
            ds_metric_obj = self.dataset_metrics[out_dir]
            ds_metric_obj.update_rouge_ss()
        print('Finished updating Rouge and SS. Dont forget to execute save_all separately..')
        

    def create_mean_over_runs(self, newfile):
        """ Create and save a new results_dict object which contains the average scores over
            a series of model runs e.g. with different seeds.
            Note: Only the overall and individual scores are averaged. Other values
                  are simply copied from the first result set in the list.
                  
            Usage: 
                avg_results_dict = res.create_mean_over_runs(newfile='/data/thar011/out/unifiedqa_averages/v3_avg3runs_eval_metrics.json')

        """
        firstset = list(self.dataset_metrics.keys())[0]
        print(f'Copying {firstset} into new results_dict...')
        avg_results_dict = copy.deepcopy(self.dataset_metrics[firstset].results_dict)
        for dset in avg_results_dict.keys():   # for each eval dataset
            print(f'Averaging {dset} ...')
            for metric in avg_results_dict[dset]['comp_metrics']:
                avg_score = 0.0
                avg_scores = np.zeros((len(avg_results_dict[dset]['predictions'])))
                cnt_score = 0
                cnt_scores = 0
                for res_set in self.dataset_metrics.keys():
                    cnt_score += 1
                    curr_score = self.dataset_metrics[res_set].results_dict[dset][metric]['score']
                    avg_score += curr_score
                    curr_scores = self.dataset_metrics[res_set].results_dict[dset][metric]['scores']
                    if curr_scores != []:   # 'RL' type has no individual scores
                        cnt_scores += 1
                        avg_scores += np.array(curr_scores)
                avg_score = avg_score / cnt_score
                if cnt_scores > 0:
                    avg_scores = list(avg_scores / cnt_scores)
                else:
                    avg_scores = []
                avg_results_dict[dset][metric]['score'] = avg_score
                avg_results_dict[dset][metric]['scores'] = avg_scores
        base, fname = os.path.split(newfile)
        os.makedirs(base, exist_ok=True)    
        with open(newfile, 'w') as f:
            json.dump(avg_results_dict, f)
        print(f"Saved to {newfile}")
        return avg_results_dict
                
            
    def display_results(self):
        """ Very rough printout for sanity checking purposes """
        for results_file in self.results_dict:
            print('###################################################')
            print(f'Results for {results_file}:')
            print('###################################################')
            result = self.results_dict[results_file]
            for key in result:
                print(f'Dataset Set: {key}:')
                tableset = result[key]
                summer = 0.0
                counter = 1
                for singleresult in tableset:
                    counter += 1
                    summer += singleresult[1]
                    print(singleresult)
                print(f'Summary Average:{summer/counter}')
            
                
    def crosstab_x_tasks(self, dsetset='unseen1', outname='tmp_eval_across_models.txt'):
        """ compute a crosstab of y axis being datasets in a particular set and x being output results ie difft model runs
        The output can be copied and pasted into a spreadsheet
        Usage: outlist.crosstab_x_tasks(dsetset='unseen1')
        outlist.crosstab_x_tasks(dsetset='unseen2')
        outlist.crosstab_x_tasks(dsetset='seen1')
        outlist.crosstab_x_tasks(dsetset='mmlu_unseen1')
        """
        outfile = os.path.join(self.logdir, outname)
        xtab = {}  # {'dataset1': [ ('output_dir1', score, 'metric type') ... ] } 
        for i, results_file in enumerate(self.results_dict):
            result = self.results_dict[results_file][dsetset]  #result = list of tuples ('dataset1', score, 'metric type')
            for dset in result:
                if xtab.get(dset[0], None) is None:
                    xtab[dset[0]] = []
                xtab[dset[0]].append( (self.shortnames[i], dset[1], dset[2]) )
        outlist = []
        header = 'Eval Dataset,Metric'        
        for shortname in self.shortnames:
            header = header + ',' + shortname
        print(header)
        outlist.append(header)
        
        for dset in xtab:
            row = xtab[dset]
            outstr = dset + ',' + row[0][2]
            for col in row:
                outstr = outstr + ',' + str(col[1])
            print(outstr)    
            outlist.append(outstr)
        with open(outfile, 'w') as f:
            f.write('\r\n'.join(outlist))
             
                
def run_all(logdir, results_list):
    """ Runs reports involving comparing model runs...
    Usage: 
        results_list = ['/data/thar011/out/unifiedqa_averages/comp3runs046/v3_avg3runs_eval_metrics.json',
                        '/data/thar011/out/unifiedqa_averages/comp3runs046/v7_avg3runs_eval_metrics.json',
                        
                       ]
        results_list = ['/data/thar011/out/unifiedqa_bart_large_v3/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v2_dev_in_train/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v3_no_facts/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v6_sqa_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s5_v2_sqafacts_dev_in_train_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s3_v1_cwwv/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s3_v2_cwwv_atomic/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v2_cwwv_premask_atomic_premask/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v3_cwwv_ssvise_atomic_ssvise/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v1_qasc_dev_facts/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s5_v1_qasc_facts/eval_metrics.json',
                       ]
        logdir='/data/thar011/out/unifiedqa_averages/s2s3s4s5_v1/'
        run_all(logdir, results_list)
    """
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Reports will be output to {logdir}')
    os.makedirs(logdir, exist_ok=True)
    res = OutputResults(results_list, logdir)
    res.crosstab_x_tasks(dsetset='seen1', outname='eval_across_models_seen1.txt') # 'final' set - 10 unrestricted eval datasets
    res.crosstab_x_tasks(dsetset='unseen4', outname='eval_across_models_us4.txt') # 'final' set - 10 unrestricted eval datasets
    res.crosstab_x_tasks(dsetset='unseen6', outname='eval_across_models_us6lowsimtdnd.txt') # 'final' set - 10 unrestricted eval datasets        
    res.crosstab_x_tasks(dsetset='mmlu_unseen1', outname='eval_across_models_mmlu_us1.txt') # 'final' set - 10 unrestricted eval datasets
    return





