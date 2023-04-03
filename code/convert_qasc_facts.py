#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:51:50 2021

@author: tim hartill

Combine pos + neg rationales from:

pos:
- qasc gold facts
- eqasc alternative gold facts      https://github.com/harsh19/Reasoning-Chains-MultihopQA
- fake "the answer must be .." and "thus of the choices..." style pos contexts by modifying existing gold contexts (only applicable in conjunction with mc options in rr model trainer)

neg:
- qasc negs from llm neg greedy prompt (T...)
- qasc negs from llm neg sampling prompt (T...)



rationale reranker 'rr' training format:

    Output format:
    [ {'question': 'question text EXCLUDING MC options and preceding initial ctxt if any',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'hpqa',
       'pos_paras': [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]], 'mc_only': False}, ...],
       'neg_paras': [], #Same format as pos_paras but filled in later
       'mc_options':  '(A) banana (B) ...'  #key only present if multichoice options exist...
       'context': 'An initial para or other necessary context if exists'  #key only present if initial para exists...
       }, {...}, ..
     
    ]



also Convert QASC facts into explanation datasets qasc_mc_expl_ans

Note: making explanation the 2 gold facts, could alternatively have been the single gold fact but this usually has the answer in it as a single lookup.

"""
import os
import copy
import random
import json

import eval_metrics
import utils
import text_processing

#MAX_OUTPUT_TOKENS = 127 #max token size of total explanation text excl BOS & EOS tokens

UQA_DIR = eval_metrics.UQA_DIR


QASC_DIR = '/home/thar011/data/qasc/QASC_Dataset/'

QASC_UQA_DIR = '/data/thar011/data/unifiedqa/qasc_'
MC_COMPLETION = 'mc_expl'
OD_COMPLETION = 'od_expl'
MC_ANS = 'mc_expl_ans'
OD_ANS = 'od_ans'
Q_PREFIX = 'Add Explanation: '

EQASC_DIR_IN = '/home/thar011/data/eqasc/'
EQASC_DEV_FILE = EQASC_DIR_IN + 'eqasc_dev_grc.json'
EQASC_TRAIN_FILE = EQASC_DIR_IN + 'eqasc_train_grc.json'


# LLM neg rationales
file_rr_dev_negs = ['/large_data/thar011/out/mdr/logs/LLM_TEST28_SPANYN_hpqa_dev_using_muv2_1krandord-02-02-2023-LLM-bigscience-bloom-maxsmpls1000-randTrue/llm_samples_with_context.json', 
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T43_HPQA_R4C_DEV_onv6_sample-03-02-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T44_HPQA_R4C_DEV_onv6mod2_sample-03-03-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
               ]
file_rr_train_negs = ['/large_data/thar011/out/mdr/logs/LLM_TEST29_SPANYN_hpqa_train_using_muv2_10krandord-02-03-2023-LLM-bigscience-bloom-maxsmpls10000-randTrue/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T45_HPQA_R4C_TRAIN_onv6_sample-03-04-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T46_HPQA_R4C_TRAIN_onv6mod2_sample-03-05-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 ]

# the full final sample files with potentially multiple pos and negs including samples without negs:
rr_dev = QASC_DIR + 'qasc_dev_rr_all_pos_neg.jsonl'
rr_train = QASC_DIR + 'qasc_train_rr_all_pos_neg.jsonl'
# the full final sample files with potentially multiple pos and negs excluding samples without negs:
rr_dev_exclposonly = QASC_DIR + 'qasc_dev_rr_all_pos_neg_exclposonly.jsonl'
rr_train_exclposonly = QASC_DIR + 'qasc_train_rr_all_pos_neg_exclposonly.jsonl'




qasc_dev = utils.load_jsonl(os.path.join(QASC_DIR,'dev.jsonl'))
qasc_train = utils.load_jsonl(os.path.join(QASC_DIR,'train.jsonl'))


def create_explanation(split):
    """ Mainly to create explanation key """
    for s in split:
        f = [text_processing.format_sentence(s['fact1']), text_processing.format_sentence(s['fact2'])]
        random.shuffle(f)
        s['explanation'] = ' '.join(f)
        q = s['question']['stem']
        mc = ''
        for c in s['question']['choices']:
            if s['answerKey'] == c['label']:
                ans = c['text']
            mc += ' (' + c['label'] + ') ' + c['text']
        mc = mc.strip()  
        s['mc_options'] = mc
        s['answer'] = ans.strip()
        s[MC_COMPLETION] = utils.create_uqa_example(Q_PREFIX + q, mc, s['explanation'])
        s[OD_COMPLETION] = utils.create_uqa_example(Q_PREFIX + q, None, s['explanation'])
        s[MC_ANS] = utils.create_uqa_example(q, mc + '\\n' + s['explanation'], ans)
        s[OD_ANS] = utils.create_uqa_example(q, s['explanation'], ans)
    return


def save_single(split, outdir, ds_type, file):
    """ save a single dataset split """
    out = [s[ds_type] for s in split]
    outfile = os.path.join(outdir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))    
    return


def save_datasets(dev, train):
    """ save uqa-formatted dataset """
    for ds_type in [MC_COMPLETION, OD_COMPLETION, MC_ANS, OD_ANS]:
        outdir = QASC_UQA_DIR + ds_type
        print(f'Saving dataset to {outdir} ...')
        os.makedirs(outdir, exist_ok=True)
        save_single(dev, outdir, ds_type, 'dev.tsv')
        save_single(train, outdir, ds_type, 'train.tsv')
    print('Finished saving uqa-formatted explanation datasets!')
    return


def simplify(eqasc):
    """ extract only valid reasoning chains where all facts are different from the original gold facts since some qasc single facts are sufficient..
    and invalid reasoning chains which dont contain either original gold fact or any altnative reasoning chain fact
    """
    num_valid = 0
    num_invalid = 0
    out = []
    for row in eqasc:
        out_dict = {'question': row['question']['stem'], 'id': row['id'], 'fact1': row['fact1'], 'fact2': row['fact2'],
                    }
        facts = set([row['fact1'].strip().lower(), row['fact2'].strip().lower()])
        invalid_facts = set()
        correct_choice = row[ 'answerKey' ]  # in eqasc always '1' and the mc options are reordered s.t. the correct choice is always first in the list..
        out_dict['valid_chains'] = []
        out_dict['invalid_chains'] = []
        for c in row[ 'question' ][ 'choices' ]: # each question has 8 choices typically (each mc option tested) but only 1 is valid
            if correct_choice == c['label']:
                out_dict['choice_text']  = c["text"]
            for j, chain in enumerate(c['chains']):
                chain2 = chain[2]
                f1, f2 = chain[0]['text'].strip(), chain[1]['text'].strip()
                if correct_choice != c['label']: # for incorrect mc options, turk_label key isnt there but we know these are false
                    turk_label = 'no'
                elif chain2.get('turk_label') is None:  # correct mc option..
                    turk_label = 'X'
                elif chain2['turk_label'].get('label') is None:
                    turk_label = 'XX'
                elif chain2['turk_label']['label'].strip().lower() == 'yes':
                    turk_label = 'yes'
                elif chain2['turk_label']['label'].strip().lower() == 'no':
                    turk_label = 'no'
                else:  # various labels other than yes or no exist...
                    turk_label = 'XXX'
                # take a pos if either fact not in current set of gold facts
                if turk_label == 'yes' and (f1.lower() not in facts or f2.lower() not in facts):
                        out_dict['valid_chains'].append( {'fact1': f1, 'fact2': f2, 'choice': c['text']} )
                        facts.add(f1.lower())
                        facts.add(f2.lower())
                        num_valid += 1
                # take neg if both facts are not in gold facts since if either is there is possibility that it is sufficient by itself
                # also exclude if the first fact is already present in invalid facts since these are often repeated and only the second fact is different
                elif turk_label == 'no' and (f1.lower() not in facts and f2.lower() not in facts) and f1.lower() not in invalid_facts:
                    out_dict['invalid_chains'].append( {'fact1': f1, 'fact2': f2, 'choice': c['text']} )
                    invalid_facts.add(f1.lower())
                    invalid_facts.add(f2.lower())
                    num_invalid += 1
        out.append(out_dict)
        
    print(f'Num valid chains={num_valid}  Num invalid chains={num_invalid}')
    return out


random.seed(42)
create_explanation(qasc_dev)
create_explanation(qasc_train)

# Below code outputs qasc_mc_expl_ans tsv formatted datasets
save_datasets(qasc_dev, qasc_train)

#################################
# Create rr formats below
#################################

dev_rr_format = [utils.create_rr_format(s['question']['stem'], s['explanation'], s['answer'],
                                        sentence_spans=None, _id=s['id'], src='qasc', append_q_char='?',
                                        mc_options=s['mc_options']) for s in qasc_dev]

train_rr_format = [utils.create_rr_format(s['question']['stem'], s['explanation'], s['answer'],
                                        sentence_spans=None, _id=s['id'], src='qasc', append_q_char='?',
                                        mc_options=s['mc_options']) for s in qasc_train]

utils.saveas_jsonl(dev_rr_format, rr_dev)
utils.saveas_jsonl(train_rr_format, rr_train)

with open(EQASC_DEV_FILE, 'r') as f:
    eqasc_dev = json.load(f)  #926

with open(EQASC_TRAIN_FILE, 'r') as f:
    eqasc_train = json.load(f)  #8134 questions

eqasc_dev_simplified = simplify(eqasc_dev) #Num valid chains=1559  Num invalid chains=14173
eqasc_train_simplified = simplify(eqasc_train) #Num valid chains=16161  Num invalid chains=118742




