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
Dataset type (configured per dataset in dataset_attributes.py):
    EX: F1
    AB: F1 or rougeL except NatQA uses EM
    MC: Select option with highest F1 score between prediction and each option
    YN: YN calc determines prediction

NOTE: Any dataset used in evaluation metrics calculation and/or similarity calculation 
      must be added to dataset_attribs below...


"""
import numpy as np
import re
import string
import collections
import json
import os
import copy
import argparse

import datasets  # NOTE: must also pip install rouge_score 
import eval_drop
from sari import SARI
from dataset_attributes import dev_eval, test_eval, metric_groups, dataset_attribs, replace_sim_with
from dataset_attributes import unifiedqa_base_train, synth_num_base_train, poet_base_train, tt_base_train
from dataset_attributes import q_paras_train, q_paras_noanswer_train, q_od_train, q_mc_train, q_mc_paras_train
from dataset_attributes import q_ret_paras_train, q_ret_paras_maxp4_train, q_ret_paras_train_v2, q_ret_paras_maxp4_train_v2
from dataset_attributes import unifiedqa_unseen_1, unifiedqa_unseen_2, unifiedqa_unseen_3, unifiedqa_unseen_4, unifiedqa_unseen_4_map, unifiedqa_unseen_5, unifiedqa_unseen_6, unifiedqa_seen_1, mmlu_unseen_1
from dataset_attributes import UQA_DIR, SVISED_EXPL_ANS, selfsupervisedkey, add_explanationkey, EXPL_COMP_KEY, special_tokens_dict
from dataset_attributes import LDATA, HDATA
from dataset_attributes import create_datasets_dynamic, get_gt_file_path
from dataset_attributes import eval_set, answer_type_map


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
        
        eg mixture = 'unifiedqa,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised'
        eg mixture = 'tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised'
    """
    unified_dataset  = []
    mixture_file_key = ''
    mixturelist = mixture.split(',')
    for ds in mixturelist:
        mixture_file_key = mixture_file_key + '_' + ds
        if ds == 'unifiedqa':
            unified_dataset.extend(unifiedqa_base_train)
        elif ds == 'tt_all':
            unified_dataset.extend(tt_base_train)
        elif ds == 'poet_all':
            unified_dataset.extend(poet_base_train)
        elif ds == 'synth_num_all':
            unified_dataset.extend(synth_num_base_train)  
        elif ds == 'q_paras_all':
            unified_dataset.extend(q_paras_train)  
        elif ds == 'q_paras_noanswer_all':
            unified_dataset.extend(q_paras_noanswer_train)
        elif ds == 'q_od_all':
            unified_dataset.extend(q_od_train)
        elif ds == 'q_mc_all':
            unified_dataset.extend(q_mc_train)
        elif ds == 'q_mc_paras_all':
            unified_dataset.extend(q_mc_paras_train)
        elif ds == 'q_ret_paras_all':
            unified_dataset.extend(q_ret_paras_train)
        elif ds == 'q_ret_paras_maxp4_all':
            unified_dataset.extend(q_ret_paras_maxp4_train)
        elif ds == 'q_ret_paras_all_v2':
            unified_dataset.extend(q_ret_paras_train_v2)
        elif ds == 'q_ret_paras_maxp4_all_v2':
            unified_dataset.extend(q_ret_paras_maxp4_train_v2)
        else:
            unified_dataset.append(ds)
    return unified_dataset, mixture_file_key


def load_uqa_supervised(file, ans_lower=True, verbose=True):
    """ Load a unifiedqa formatted .tsv file and return question+context as list of str and answers as list of str
    """
    if verbose:
        print(f"Reading {file}...")
    questions = []
    answers = []
    ctr = 1
    with open(file, "r") as f:
        for line in f:
            try:
                question, answer = line.split("\t")
            except:
                print(f"ERROR loading line: {ctr} ##{line}##")
            if ans_lower:
                answer = answer.lower()
            questions.append( question.strip() )
            answers.append ( answer.strip() )
            ctr += 1
    if verbose:
        print(f"Successfully loaded {len(questions)} rows.")
    return questions, answers


# the standard "squad" normalization
def normalize_answer_squad(s):
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


# DROP-style normalization - same as squad only number formats are normalised as well eg 1,234 -> 1234.0
def normalize_answer(s):
    return eval_drop._normalize_answer(s)


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
            return get_f1(f1_str1, f1_str2)  #TJH: Empirically, calculating overlap with F1 does very slightly better
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
        return int(np.max([get_exact_match(prediction, gt) for gt in groundtruth]))
    return int(normalize_answer(prediction) == normalize_answer(groundtruth))


def get_f1(prediction, groundtruth):
    """ DROP-style f1 where if there is a number in the answer and it doesnt match then F1=0 regardless of other matches
    Note: Some Drop samples for encoder models have gold & pred  = [span1, span2] for multispan answers and both spans must be matched. 
          However in seq2seq models these spans are concatenated together into a single string and a gold answer in list form indicates *any* element can match
          In practice this means in seq2seq DROP etc always have both pred and gold as single strings and other datasets may have multiple valid answers in list form.
    """
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0.0
        return float(np.max([get_f1(prediction, gt) for gt in groundtruth]) )
    return float(eval_drop.f1_metrics(prediction, groundtruth))


# adapted from https://github.com/huggingface/datasets/blob/86e66e7be32f96a625314b8e7d4b16d703eba82d/metrics/squad_v2/evaluate.py#L104
def get_f1_squad(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0.0
        return float(np.max([get_f1(prediction, gt) for gt in groundtruth]) )
    gold_toks = get_tokens(groundtruth)
    pred_toks = get_tokens(prediction)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
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


def get_sari(question, prediction, groundtruth, sari=None):
    """ Score using SARI metric - see sari.py for paper ref and details
    """
    if sari is None:
        sari = SARI()
    sources = [question.replace("?", " ?").split()]  # [ ['word1', 'word2', ...'wordn', '?'] ]
    if type(prediction) == list:
        predictions = [" ".join(prediction).replace("?", " ?").split()]  # [ ['decomp1 word1', 'decomp 1word2', ...'decompn wordn', 'decompn ?'] ]
    else:
        predictions = [prediction.replace("##","").replace("?", " ?").split()]  # [ ['decomp1 word1', 'decomp 1word2', ...'decompn wordn', 'decompn ?'] ]
    if type(groundtruth) == list:
        targets = [[" ".join(groundtruth).replace("?", " ?").split()]] # # [ [ ['decomp1 word1', 'decomp 1word2', ...'decompn wordn', 'decompn ?'] ] ]
    else:
        targets = [[groundtruth.replace("##","").replace("?", " ?").split()]] # # [ [ ['decomp1 word1', 'decomp 1word2', ...'decompn wordn', 'decompn ?'] ] ]
    score = float(sari(sources, predictions, targets))
    return score


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


class Sari:
    """ Return SARI metric
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text']
        questions = ['question1 text', 'question2 text']
        Each groundtruth & pred of form "overall answer ## decomp1 q? decomp1 ans ## decomp2 q? decomp2 ans"
        although what could be passed in could be minus overall answer and/or minus decomp answers and/or minus decomps
    """
    def __init__(self):
        self.last_scores = {}
        self.saris = []

    def compute_metric(self, predictions, groundtruths, questions):
        self.saris = []
        sari_scorer = SARI()
        for prediction, groundtruth, question in zip(predictions, groundtruths, questions):
            s = get_sari(question, prediction, groundtruth, sari_scorer)
            self.saris.append(s)
        sari_score = sari_scorer.get_metric()['SARI'] * 100.0
        self.last_scores['sari'] = sari_score
        return sari_score


class F1:
    """ Return f1 metric as defined for squad 
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = ['gt1 text', 'gt2 text'] or [['gt1.1 text', 'gt1.2 text'], ['gt2.1 text', 'gt2.2 text']]
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
        groundtruths = ['gt1 text', 'gt2 text'] or [['gt1.1 text', 'gt1.2 text'], ['gt2.1 text', 'gt2.2 text']]
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


class Bleu:
    """ Return Bleu metric
    print(metric.inputs_description) for details
    Usage: bscorer = Bleu()
    bleu2score = bscorer.compute_metric(predictions, groundtruth) 
        predictions = ['pred1 text', 'pred2 text']
        groundtruths = [['gt1.1 text', 'gt1.2 text'], ['gt2.1 text', 'gt2.2 text']]  #note extra nesting
    """
    def __init__(self, N=2, smooth=False):
        self.metric = datasets.load_metric('bleu')
        self.N = N            # max n-gram datasets default 4
        self.smooth = smooth  # Lin 2004 smoothing datasets default False
        self.last_scores = None
        
    def compute_metric(self, predictions, groundtruths, norm=False):
        if norm:
            predictions_norm = [get_tokens(p) for p in predictions]
            groundtruths_norm = [[get_tokens(gt) for gt in gtl] for gtl in groundtruths]
        else:   #Note: empirically 
            predictions_norm = [p.lower().split() for p in predictions]
            groundtruths_norm = [[gt.lower().split() for gt in gtl] for gtl in groundtruths]
        self.last_scores = self.metric.compute(predictions=predictions_norm, references=groundtruths_norm, 
                                               max_order = self.N, smooth=self.smooth) 
        return self.last_scores["bleu"] * 100.0


class SelfBleu:
    """ Calculate self-bleu from:
        Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A Benchmarking Platform for Text Generation Models. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 1097–1100, Ann Arbor, MI, USA. Association for Computing Machinery.
        As measure of diversity of list of sentences
    """
    def __init__(self, N=2, smooth=False):
        self.bleu = Bleu(N=N, smooth=smooth)
        self.N = N            # max n-gram datasets default 4
        self.smooth = smooth  # Lin 2004 smoothing datasets default False. If true: Add 1 to both numerator and denominator
        self.last_scores = {}
        self.self_bleus = []
        
    def compute_metric(self, predictions, norm=False):
        self.self_bleus = []
        ref_count = len(predictions)
        for idx in range(ref_count):
            prediction = predictions[idx]
            references = predictions[:idx] + predictions[idx+1:]
            score = self.bleu.compute_metric([prediction], [references], norm=norm)
            self.self_bleus.append(score)
        self.last_scores['selfbleu'] = sum(self.self_bleus) / len(self.self_bleus)
        return self.last_scores['selfbleu']
    

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
                          'gt_file': 'eg '/data/thar011/data/unifiedqa/newsqa/dev.tsv', 
                          #'gt_file_tokenized': eg 'test-uncased-xbos-BartTokenizedFast.json',
                          #'groundtruths_tokenized': from decoder_input_ids: eg [563, 256, 9, 0, 1, 1, 1],
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
    dsmetrics = DatasetMetrics('/data/thar011/out/unifiedqa_bart_large_s6_v8_musique_qa_decomp_ans_plus_new_decomps/eval_metrics.json')
    dsmetrics = DatasetMetrics('/data/outdir/eval_metrics.json')
    
    ds = 'musique_qa_decomp_ans'
    ds = 'arc_easy'

    dsmetrics.get_single_pred_vs_groundtruth(ds)
    dsmetrics.get_summary_plus_single_preds(ds, number_samples=1)
    dsmetrics.get_single_pred_vs_groundtruth(dsmetrics.datasets[0]) # display prediction vs gt plus metrics for a single sample

    dsmetrics.get_single_pred_vs_groundtruth(dsmetrics.datasets[2]) # display prediction vs gt plus metrics for a single sample
    

    ds_values = dsmetrics.get_pref_values_set('unifiedqa_unseen_1')  # return list of metric values for a list of datasets
    print(ds_values[0])  # ('newsqa', 54.433245958483546, 'F1')
    
    ds_values = dsmetrics.get_pref_values_set('mmlu_unseen_1') #  # return list of metric values for a list of mmlu datasets
    
    dsmetrics.results_dict['mmlu_college_chemistry_test']['SS']['choices'][0]  #view parsed choices for a MC question

    dsmetrics.results_dict['musique_mu_dev_qa_paras_decomp_ans']['groundtruths'][1]    
    dsmetrics.results_dict['musique_mu_dev_qa_paras_decomp_ans']['predictions'][1]    
    
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
    

    def get_single_pred_vs_groundtruth(self, ds, index=0, metric='ALL', incl_toks=False, incl_choices=False):
        """ Return prediction and groundtruth for a single example
        Usage: tst.get_single_pred_vs_groundtruth('arc_hard', index=2, metric='ALL')
        """
        ret_dict = {}
        ret_dict['groundtruth'] = self.results_dict[ds]['groundtruths'][index]
        if self.results_dict[ds].get('groundtruths_tokenized', None) is not None and incl_toks:
            ret_dict['groundtruths_tokenized'] = self.results_dict[ds]['groundtruths_tokenized'][index]
        ret_dict['prediction'] = self.results_dict[ds]['predictions'][index]
        if metric is not None:
            if metric == 'ALL':
                for m in self.results_dict[ds]['comp_metrics']:
                    if m == 'SS':
                        ret_dict[m+'_new_prediction'] = self.results_dict[ds][m]['newpreds'][index]
                        if incl_choices:
                            ret_dict[m+'_from_choices'] = self.results_dict[ds][m]['choices'][index]
                    ret_dict[m+'_score'] = self.results_dict[ds][m]['scores'][index] * 100
            else:
                if metric == 'SS':
                    ret_dict['new_prediction'] = self.results_dict[ds][metric]['newpreds'][index]
                    if incl_choices:
                        ret_dict['from_choices'] = self.results_dict[ds][metric]['choices'][index]
                ret_dict['score'] = self.results_dict[ds][metric]['scores'][index] * 100
        return ret_dict

    
    def get_summary_plus_single_preds(self, ds, number_samples=5, metric='ALL'):
        """ Return summary of metrics for a single dataset including "x" random individual samples """
        gt_file = get_gt_file_path(self.results_dict[ds]['gt_file'])
        questions, answers = load_uqa_supervised(gt_file, ans_lower=False)
        num_q = len(questions)
        np.random.seed(42)
        sel_indices = np.random.choice(num_q, number_samples, replace=False)
        ret_dict = {'eval_dataset': ds, 'metrics':[], 'rand_samples':[]}
        for curr_metric in self.results_dict[ds]['comp_metrics']:
            ret_dict['metrics'].append( {curr_metric + '_score': self.results_dict[ds][curr_metric]['score']} )
        for i in sel_indices:
            single_dict = {'index': int(i), 'question': questions[i]}
            single_dict_details = self.get_single_pred_vs_groundtruth(ds, index=i, metric=metric)
            single_dict.update(single_dict_details)
            ret_dict['rand_samples'].append(single_dict)
        return ret_dict

    
    def format_eval_summary(self, ret_dict):
        """ Format ret_dict for text file output """
        out_list = []
        summary = f"{self.short_name},{ret_dict['eval_dataset']}"
        out_list.append(self.short_name +': EVAL DATASET:' + ret_dict['eval_dataset'])
        out_str = 'OVERALL METRICS: '
        for k in ret_dict['metrics']:
            key = list(k.keys())[0]
            val = list(k.values())[0]
            out_str += f"{key}: {val:.2f}   "
            summary += f",{val:.2f}"
        out_list.append(out_str.strip())
        out_list.append('')
        out_list.append('Random Samples:')
        for k in ret_dict['rand_samples']:
            out_list.append('['+str(k['index'])+'] INPUT: ' + k['question'].strip())
            out_list.append('LABL: ' + k['groundtruth'].strip())
            out_list.append('PRED: ' + k['prediction'].strip())
            out_str = 'SAMPLE METRICS: '
            for key in k.keys():
                if key.endswith('_score'):
                    val = k[key]
                    out_str += f"{key}: {val:.2f}   "
            out_list.append(out_str.strip())
            out_list.append('')
        return out_list, summary
                    
               
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
             

def output_summary(logdir, results_list, include_list, number_samples=3, metric='ALL', outname='eval_dataset_performance_summary.txt'):
    """ Output a summary of evaluation dataset performance
        include_list = ['musique_mu_dev_qa', 'musique_mu_dev_qa_decomp_ans', 
                        'musique_mu_dev_qa_paras', 'musique_mu_dev_qa_paras_decomp_ans', 
                        'musique_qa', 'musique_qa_decomp_ans', 'musique_qa_paras', 'musique_qa_paras_decomp_ans']  #Eval datasets to include
        results_list = [ # model outputs to include
                        '/data/thar011/out/unifiedqa_bart_large_s6_v5_musique_qa_decomp_ans_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v8_musique_qa_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v9_musique_qa_plus_qa_decomp_ans_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v10_musique_qa_plus_qa_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v11_musique_qa_paras_plus_qa_paras_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v12_musique_qa_paras_plus_qa_paras_decomp_ans/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v13_musique_qa_plus_qa_decomp_ans_full_plus_new_decomps_full/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v14_musique_qa_paras_plus_qa_paras_decomp_ans_full/eval_metrics.json',
                       ]
        logdir='/data/thar011/out/unifiedqa_averages/s2s3s4s5s6_v1/'
        output_summary(logdir, results_list, include_list, number_samples=3, metric='ALL', outname='eval_dataset_performance_summary.txt')
    """
    if logdir[-1] != '/':
        logdir += '/'        
    os.makedirs(logdir, exist_ok=True)
    outfile = os.path.join(logdir, outname)
    print(f'Detail Report will be output to {outfile}')
    summary_file = 'summary_' + outname
    summary_file = os.path.join(logdir, summary_file)
    print(f'Summary Report will be output to {summary_file}')
    out_list = []
    out_summary = ['Dataset,Eval Dataset,EM,F1-OverallAns,F1-DecompAns,SARI-Decomp,SARI-DecompAns']
    for result in results_list:
        print(f"Processing: {result}")
        dsmetrics = DatasetMetrics(result)
        for ds in include_list:
            ret_dict = dsmetrics.get_summary_plus_single_preds(ds, number_samples=number_samples, metric=metric)
            ds_output_list, summary = dsmetrics.format_eval_summary(ret_dict)
            ds_output_list.append('')
            ds_output_list.append('')
            out_list += ds_output_list
            out_summary.append(summary)
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(out_list))
    with open(summary_file, 'w') as f:
        f.write('\r\n'.join(out_summary))
    
    return

                
def run_all(logdir, results_list, include_list=['unseen4', 'seen1', 'unseen6', 'mmlu_unseen1']):
    """ Runs reports involving comparing model runs...
    Usage: 
        results_list = ['/data/thar011/out/unifiedqa_bart_large_TEST/eval_metrics.json']
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
                        '/data/thar011/out/unifiedqa_bart_large_s6_v3_musique_qa_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v7_musique_qa_decomp_ans_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v4_musique_qa_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v5_musique_qa_decomp_ans_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v9_musique_qa_plus_qa_decomp_ans_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v10_musique_qa_plus_qa_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v8_musique_qa_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v6_musique_qa_paras_plus_all_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v11_musique_qa_paras_plus_qa_paras_decomp_ans_plus_new_decomps/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v12_musique_qa_paras_plus_qa_paras_decomp_ans/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v13_musique_qa_plus_qa_decomp_ans_full_plus_new_decomps_full/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s6_v14_musique_qa_paras_plus_qa_paras_decomp_ans_full/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s7_v1_uqa_sqa_mqa_expl_mswq_explans_msw/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s7_v3_uqa_ms_explans_msw/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s7_v4_uqa_ms_expl_mswq_explans_msw_ssviseall_mswq/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s7_v5_uqa_msan_expl_mswqa_explans_mswa_ssviseall_mswq/eval_metrics.json',
                       ]
        
        results_list = ['/large_data/thar011/out/mdr/logs/UQA_s11_v1_all_g1_qa_g2_numlit_wikissvise_COPY_AT810Ksteps/eval_metrics.json',
                        '/large_data/thar011/out/mdr/logs/UQA_s11_v1_all_g1_qa_g2_numlit_wikissvise/eval_metrics.json',
                        '/large_data/thar011/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_COPY_AT810Ksteps/eval_metrics.json',
                        '/large_data/thar011/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/eval_metrics.json',
                        '/large_data/thar011/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds/eval_metrics.json',
                        '/large_data/thar011/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_COPY_AT_770Ksteps/eval_metrics.json',
                       ]

        
        logdir='/data/thar011/out/unifiedqa_averages/s11/'
        run_all(logdir, results_list, include_list=['unseen4', 'seen1'])
    """
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Reports will be output to {logdir}')
    os.makedirs(logdir, exist_ok=True)
    res = OutputResults(results_list, logdir)
    if 'seen1' in include_list:
        res.crosstab_x_tasks(dsetset='seen1', outname='eval_across_models_seen1.txt') # 'final' set - 10 unrestricted eval datasets
    if 'unseen4' in include_list:
        res.crosstab_x_tasks(dsetset='unseen4', outname='eval_across_models_us4.txt') # 'final' set - 10 unrestricted eval datasets
    if 'unseen5' in include_list:
        res.crosstab_x_tasks(dsetset='unseen5', outname='eval_across_models_us5dyn.txt') # 'final' set - 10 unrestricted eval datasets
    if 'unseen6' in include_list:
        res.crosstab_x_tasks(dsetset='unseen6', outname='eval_across_models_us6lowsimtdnd.txt') # 'final' set - 10 unrestricted eval datasets        
    if 'mmlu_unseen1' in include_list:
        res.crosstab_x_tasks(dsetset='mmlu_unseen1', outname='eval_across_models_mmlu_us1.txt') # 'final' set - 10 unrestricted eval datasets
    return


def run_eval_set(args):
    """ Run eval for particular eval set ('default' by default)
    Outputs for 'unseen4' and 'seen1' lists of datasets. To output other lists run run_all(..) manually
    """
    logdir = eval_set[args.eval_set]['output_dir']
    results_list = eval_set[args.eval_set]['models']
    run_all(logdir, results_list, include_list=['unseen4', 'seen1'])
    return


def run_answer_type_breakdown(args):
    """ run breakdown for DROP/IIRC/TATQA by answer type
    res.dataset_metrics['/large_data/thar011/out/mdr/logs/UQA_s11_v1_all_g1_qa_g2_numlit_wikissvise_COPY_AT810Ksteps/eval_metrics.json'].results_dict['drop']['F1']['scores']
    
    data = [json.loads(line.strip()) for line in open(data_path).readlines()]
    """    
    logdir = eval_set[args.eval_set]['output_dir']
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Report will be output to {logdir} as answer_types_breakdown.txt')
    os.makedirs(logdir, exist_ok=True)
    outfile = os.path.join(logdir, 'answer_types_breakdown.txt')

    results_list = eval_set[args.eval_set]['models']
    res = OutputResults(results_list, logdir)
    firstmodel = list(res.dataset_metrics.keys())[0] 

    atypedir = os.path.join(UQA_DIR, 'answer_types')
    atypefiles = list(answer_type_map.keys())
    atypedict = {}
    for file in atypefiles:
        fullpath = os.path.join(atypedir, file)
        print(f"Loading answer types file: {fullpath}")
        atypedict[file] = [json.loads(line.strip()) for line in open(fullpath).readlines()]

    xtab = {}  # {'datasetname':{'metric': 'F1', 'ans_type': {'modelshortname1': [scores]}}
    for file in answer_type_map:
        ans_types = [a['ans_type'] for a in atypedict[file]]
        for dataset in answer_type_map[file]:
            xtab[dataset] = {}
            prefmetric = res.dataset_metrics[firstmodel].get_pref_metric(dataset)
            xtab[dataset]['metric'] = prefmetric 
            xtab[dataset]['ans_types'] = {}
            for model in res.dataset_metrics.keys():
                shortname = res.dataset_metrics[model].short_name
                scores = res.dataset_metrics[model].results_dict[dataset][prefmetric]['scores']
                assert len(ans_types) == len(scores)
                for ans_type, score in zip(ans_types, scores):
                    if xtab[dataset]['ans_types'].get(ans_type) is None:
                        xtab[dataset]['ans_types'][ans_type] = {}  # {'modelshortname1': [scores]}
                    if xtab[dataset]['ans_types'][ans_type].get(shortname) is None:
                        xtab[dataset]['ans_types'][ans_type][shortname] = []
                    xtab[dataset]['ans_types'][ans_type][shortname].append(score)
    
    outlist = []
    header = 'Eval Dataset,Ans Type,Metric'
    for shortname in res.shortnames:
        header = header + ',' + shortname
    print(header)
    outlist.append(header)

    for dataset in xtab:
        for ans_type in xtab[dataset]['ans_types']:
            outstr = dataset + ',' + ans_type + ',' + xtab[dataset]['metric']
            for shortname in xtab[dataset]['ans_types'][ans_type]:
                score = str(np.mean(xtab[dataset]['ans_types'][ans_type][shortname])*100.0)
                outstr += ',' + score
            print(outstr)    
            outlist.append(outstr)
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    print(f"Answer types breakdown out to: {outfile}")    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", default="default", type=str, help="key of eval_set (defined in dataset_attributes.py).")
    args = parser.parse_args()
    run_eval_set(args)
    run_answer_type_breakdown(args)


