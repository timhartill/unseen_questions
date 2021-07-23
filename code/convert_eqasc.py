#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:32:33 2021

@author: tim hartill

Convert QASC and eQASC facts into self aupervised format para1 \\n \n para2 \\n \n

QASC Paper: 'A Dataset for Question Answering via Sentence Composition'
eQASC Paper: 'Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering'

Outputs facts files into:
    UQA_QASC_FACTS_DIR combining QASC and EQASC gold facts
    UQA_QASC_GRC_FACTS_DIR (generalised reasoning chain versions from eQASC only which also have the qasc golds in them)
    UQA_QASC_CORPUS_FACTS_EXCL_TRAINDEV The QASC corpus excluding facts in QASC+eQASC train and dev
    
Edit the below directories and filenames before running...

"""

import os
import json
import numpy as np

UQA_DIR = '/data/thar011/data/unifiedqa/'
UQA_QASC_FACTS_DIR = 'qasc_facts_selfsvised'
UQA_QASC_GRC_FACTS_DIR = 'qasc_grc_facts_selfsvised'
UQA_QASC_CORPUS_FACTS_EXCL_TRAINDEV = 'qasc_corpus_excl_traindev_selfsvised'
QASC_DIR_IN = '/data/thar011/data/qasc/QASC_Dataset/'
QASC_CORPUS_FILE = '/data/thar011/data/qasc/QASC_Corpus/QASC_Corpus.txt'
EQASC_DIR_IN = '/data/thar011/data/eqasc/'
EQASC_DEV_FILE = 'eqasc_dev_grc.json'
EQASC_TRAIN_FILE = 'eqasc_train_grc.json'


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

print(f"Loading {QASC_CORPUS_FILE}..")
with open(QASC_CORPUS_FILE, 'r') as f:
    qasc_corpus = f.read()
corpus_facts = qasc_corpus.split('\n')  # 16987131
corpus_facts = set(corpus_facts)
print(f"Corpus facts before removing train/dev ones: {len(corpus_facts)}")
    
qasc_train = load_jsonl(os.path.join(QASC_DIR_IN, 'train.jsonl')) #8135  dict_keys(['id', 'question', 'answerKey', 'fact1', 'fact2', 'combinedfact', 'formatted_question'])
qasc_dev = load_jsonl(os.path.join(QASC_DIR_IN, 'dev.jsonl')) #927 

train_facts = set()
dev_facts = set()

for q in qasc_train:
    train_facts.add(q['fact1'])
    train_facts.add(q['fact2'])
    
for q in qasc_dev:
    dev_facts.add(q['fact1'])
    dev_facts.add(q['fact2'])

both_facts = train_facts.intersection(dev_facts)

print(f"Train facts: {len(train_facts)}  Dev Facts: {len(dev_facts)}  Both: {len(both_facts)}") # Train facts: 6815  Dev Facts: 856  Both: 63

train_facts = train_facts - both_facts  #remove facts necessary for dev from the train set. Note this is the opposite of what we did for strategyqa

corpus_facts = corpus_facts - train_facts
corpus_facts = corpus_facts - dev_facts
print(f"Corpus facts after removing QASC train/dev matches: {len(corpus_facts)}")  # 16979681

with open(os.path.join(EQASC_DIR_IN, EQASC_TRAIN_FILE),'r') as f:
    eqasc_train = json.load(f)  #8134 questions

with open(os.path.join(EQASC_DIR_IN, EQASC_DEV_FILE),'r') as f:
    eqasc_dev = json.load(f)  #926


def simplify(eqasc):
    """ extract only valid reasoning chains
    """
    num_valid = 0
    out = []
    facts = set()
    gfacts = set()
    for row in eqasc:
        out_dict = {'question': row['question']['stem'],
                    'correct_choice': '0'}
        correct_choice = row[ 'answerKey' ]
        for c in row[ 'question' ][ 'choices' ]: # each question has 8 choices typically (each mc option tested) but only 1 is valid
            if correct_choice != c['label']:
                continue
            out_dict['correct_choice'] = c['label']
            out_dict['correct_choice_text']  = c[ "text" ]
            out_dict['valid_chains'] = []
            for j, chain in enumerate(c['chains']):
                chain2 = chain[2]
                if chain2.get('turk_label') is not None and chain2['turk_label'].get('label') is not None and  chain2['turk_label']['label'].strip().lower() == 'yes':
                    f1, f2 = chain[ 0 ][ 'text' ], chain[ 1 ][ 'text' ]
                    gf1 = chain2['grc'][0].replace(' [unused','X').replace('] ','').strip()+'.'
                    gf2 = chain2['grc'][1].replace(' [unused','X').replace('] ','').strip()+'.'
                    out_dict['valid_chains'].append( {'fact1': f1, 'fact2': f2, 
                                                      'gfact1': gf1, 
                                                      'gfact2': gf2} )
                    facts.add(f1)
                    facts.add(f2)
                    gfacts.add(gf1)
                    gfacts.add(gf2)
                    num_valid += 1
        out.append(out_dict)
    print(f'Num valid chains = {num_valid}')
    return out, facts, gfacts

eqasc_train_simplified, eqasc_train_facts, eqasc_train_gfacts = simplify(eqasc_train) #20398 valid chains
eqasc_dev_simplified, eqasc_dev_facts, eqasc_dev_gfacts = simplify(eqasc_dev) #2052 valid chains

print(f"eQASC train facts: {len(eqasc_train_facts)}  dev facts: {len(eqasc_dev_facts)}")  #eQASC train facts: 16529  dev facts: 1883

corpus_facts = corpus_facts - eqasc_train_facts
corpus_facts = corpus_facts - eqasc_dev_facts

print(f"Corpus facts excl train/dev eqasc facts: {len(corpus_facts)}")  #16965692

eqasc_both_facts = eqasc_train_facts.intersection(eqasc_dev_facts) #195
eqasc_train_facts = eqasc_train_facts - eqasc_both_facts

train_facts = train_facts.union(eqasc_train_facts)
dev_facts = dev_facts.union(eqasc_dev_facts)

print(f"Final Train facts: {len(train_facts)}  Dev Facts: {len(dev_facts)}")

eqasc_both_gfacts = eqasc_train_gfacts.intersection(eqasc_train_gfacts) #122

eqasc_train_gfacts = eqasc_train_gfacts - eqasc_both_gfacts

print(f"Final Train gfacts: {len(eqasc_train_gfacts)}  Dev gfacts: {len(eqasc_dev_gfacts)}") # Final Train gfacts: 28511  Dev gfacts: 3039


outdir = os.path.join(UQA_DIR, UQA_QASC_FACTS_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(train_facts))
outfile = os.path.join(outdir, 'dev.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(dev_facts))

outdir = os.path.join(UQA_DIR, UQA_QASC_GRC_FACTS_DIR)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(eqasc_train_gfacts))
outfile = os.path.join(outdir, 'dev.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(eqasc_train_gfacts))

num_q = len(corpus_facts)
corpus_facts = list(corpus_facts)
corpus_dev = []
corpus_train = []
np.random.seed(42)
dev_indices = set(np.random.choice(num_q, 500, replace=False))
for i in range(num_q):
    if i in dev_indices:
        corpus_dev.append(corpus_facts[i])
    else:
        corpus_train.append(corpus_facts[i])
        
print(f"Final Corpus Train: {len(corpus_train)}  Dev: {len(corpus_dev)}")  # Final Corpus Train: 16965192  Dev: 500

outdir = os.path.join(UQA_DIR, UQA_QASC_CORPUS_FACTS_EXCL_TRAINDEV)
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(corpus_train))
outfile = os.path.join(outdir, 'dev.txt')
with open(outfile, 'w') as f:
    f.write('\\n \n'.join(corpus_dev))

print('Finished')


        
