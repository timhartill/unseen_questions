#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:35:44 2021

@author: tim hartill

Import and convert WorldTree data 

Zhengnan Xie, Sebastian Thiem, Jaycie Martin, Elizabeth Wainwright, Steven Marmorstein, and Peter Jansen. 2020. 
WorldTree V2: A corpus of science-domain structured explanations and inference patterns supporting multi-hop inference. 
In Proceedings of the 12th Language Resources and Evaluation Conference, pages 5456-5473


"""
import os
import pandas as pd
import copy

import utils


worldtree_dir = '/home/thar011/data/worldtree/WorldtreeExplanationCorpusV2.1_Feb2020/'
explanation_dir = os.path.join(worldtree_dir, 'tablestore/v2.1/tables/')
question_dir = os.path.join(worldtree_dir, 'questions/')

def load_facts(explanation_dir, verbose=False):
    """ Load individual explanation components (facts)
    """
    files = os.listdir(explanation_dir)
    fact_dict = {}
    for file in files:
        curr_file = os.path.join(explanation_dir, file)
        print(f'Reading {curr_file}...')
        df = pd.read_csv(curr_file, sep='\t', header=0)
        for i in range(df.shape[0]):
            row = dict(df.iloc[i])
            keys = list(row.keys())
            uid = row['[SKIP] UID']
            factstr = ''
            for k in keys:
                colstr = str(row[k]).strip()
                if k in ['[SKIP] COMMENTS','[SKIP] COMMENT','[SKIP] Comments']:
                    break
                elif k == '[SKIP] UID':
                    print(f"ERROR: {file} {uid} Did not find [SKIP] COMMENTS column...")
                if colstr != 'nan':
                    factstr += ' ' + colstr
            factstr = factstr.strip() 
            row['sentence'] = factstr
            row['FILE'] = file
            if fact_dict.get(uid) is not None:
                print(f'{file}: uid: {uid}: DUPLICATE DETECTED, UPDATING TO LAST SEEN.')
            fact_dict[uid] = copy.deepcopy(row)
    return fact_dict
            

def load_questions(question_file, fact_dict, verbose=False):
    """ Load WorldTree questions for train, dev or test split
    """
    print(f'Reading {question_file}...')
    df = pd.read_csv(question_file, sep='\t', header=0)
    if verbose:
        print(df.info())
    outlist = []
    for i in range(df.shape[0]):
        row = dict(df.iloc[i])
        row['explanation_sentences'] = []
        row['explanation_roles'] = []
        row['explanation_parse1'] = []
        if type(row['explanation']) != str:
            print('No Explanation Provided:')
            print(row)
        else:    
            row['explanation_parse1'] = row['explanation'].split(' ')
            row['explanation_count'] = len(row['explanation_parse1'])
            for expl in row['explanation_parse1']:
                uid, role = expl.split('|')
                fact = fact_dict[uid]
                row['explanation_roles'].append(role)
                row['explanation_sentences'].append(fact['sentence'])
        row['explanation_count'] = len(row['explanation_parse1'])
        outlist.append(row)
    return outlist
    
fact_dict = load_facts(explanation_dir)

questions_dev = load_questions(os.path.join(question_dir, 'questions.dev.tsv'), fact_dict)
questions_test = load_questions(os.path.join(question_dir, 'questions.test.tsv'), fact_dict)
questions_train = load_questions(os.path.join(question_dir, 'questions.train.tsv'), fact_dict)

#TODO: convert questions to UQA format.
#TODO: Add '.' to each expl sentence and decide which ROLES to include.
#TODO: COULD output as supervised dataset
#TODO: FORMAT FOR GPT-J


