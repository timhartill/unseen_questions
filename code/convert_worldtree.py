#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:35:44 2021

@author: tim hartill

Import and convert WorldTree data into explanation datasets and self-supervised datasets of facts

Zhengnan Xie, Sebastian Thiem, Jaycie Martin, Elizabeth Wainwright, Steven Marmorstein, and Peter Jansen. 2020. 
WorldTree V2: A corpus of science-domain structured explanations and inference patterns supporting multi-hop inference. 
In Proceedings of the 12th Language Resources and Evaluation Conference, pages 5456-5473


1. Download and unzip Worldtree: http://www.cognitiveai.org/dist/WorldtreeExplanationCorpusV2.1_Feb2020.zip
2. Update directory variables below
3. Run. New datasets in UQA format with explanations as labels will be in .../uqa_dir/worldtree_mc_expl and .../uqa_dir/worldtree_od_expl

"""

import os
import pandas as pd
import copy
import random

import utils
import text_processing

MAX_OUTPUT_TOKENS = 127 #max token size of total explanation text excl BOS & EOS tokens


worldtree_dir = '/home/thar011/data/worldtree/WorldtreeExplanationCorpusV2.1_Feb2020/'
explanation_dir = os.path.join(worldtree_dir, 'tablestore/v2.1/tables/')
question_dir = os.path.join(worldtree_dir, 'questions/')

uqa_dir = '/data/thar011/data/unifiedqa/worldtree_'
wt_mc_completion = 'mc_expl'
wt_od_completion = 'od_expl'
wt_mc_ans = 'mc_ans'
wt_od_ans = 'od_ans'
q_prefix= 'Add Explanation: '
selfsupervisedkey = '_selfsvised'

tokenizer = utils.load_model(model_name="facebook/bart-large", loadwhat='tokenizer_only')
random.seed(42)

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
    

def load_questions(question_file, fact_dict, tokenizer, verbose=False):
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
        row['explanation_sentences_raw'] = []
        row['explanation_roles'] = []
        row['explanation_parse1'] = []
        row['token_counts'] = []
        if type(row['explanation']) != str:
            print('No Explanation Provided:')
            print(row)
            print('SKIPPING...')
            continue
        else:    
            row['explanation_parse1'] = row['explanation'].split(' ')
            row['explanation_count'] = len(row['explanation_parse1'])
            for expl in row['explanation_parse1']:
                uid, role = expl.split('|')
                fact = fact_dict[uid]
                row['explanation_roles'].append(role)
                row['explanation_sentences_raw'].append(fact['sentence'])
                row['explanation_sentences'].append( text_processing.format_sentence(fact['sentence']) )
                row['token_counts'].append( len(utils.string_to_ids(row['explanation_sentences'][-1], 
                                                                    tokenizer, 
                                                                    norm_numbers=False, append_eos=False, prepend_bos=False))+1 )
        row['explanation_count'] = len(row['explanation_parse1'])
        
        # copy facts until MAX_OUTPUT_TOKENS reached, starting with those with CENTRAL role...
        currcount = 0
        explanation = []
        for i, expl in enumerate(row['explanation_sentences']):
            if row['explanation_roles'][i].strip().upper() == 'CENTRAL':
                if currcount + row['token_counts'][i] <= MAX_OUTPUT_TOKENS:
                    explanation.append(expl)
                    currcount += row['token_counts'][i]
                else:
                    break

        random.shuffle(explanation)  #shuffle CENTRAL following MACAW but don't shuffle non-central additions

        if currcount < MAX_OUTPUT_TOKENS:
            for i, expl in enumerate(row['explanation_sentences']):
                if row['explanation_roles'][i].strip().upper() != 'CENTRAL':
                    if currcount + row['token_counts'][i] <= MAX_OUTPUT_TOKENS:
                        explanation.append(expl)
                        currcount += row['token_counts'][i]
                    else:
                        break
        
        row['explanation_sentences_final'] = ' '.join(explanation)
        row['token_count_final'] = currcount

        # reformulate the question into UQA format...
        anskey = row['AnswerKey']
        q_mc = row['question'].split('(A)')
        if len(q_mc) != 2:
            q_mc = row['question'].split('(1)')
            if anskey == '1':
                anskey = 'A'
            elif anskey == '2':
                anskey = 'B'
            elif anskey == '3':
                anskey = 'C'
            elif anskey == '4':
                anskey = 'D'
            else:
                print("ERROR: Answer key is INVALID:")
                print(row)
        q = q_mc[0]
        if len(q_mc) == 2:
            q_mc[1] = q_mc[1].replace('(2)', '(B)', 1)
            q_mc[1] = q_mc[1].replace('(3)', '(C)', 1)
            q_mc[1] = q_mc[1].replace('(4)', '(D)', 1)
            mc = '(A)' + q_mc[1]
            ans = utils.find_mc_answer(mc, anskey)
            if ans == '':
                print('ANSWER NOT FOUND:')
                print(row)
        else:
            mc = ''
            print('NO MC OPTIONS FOUND:')
            print(row)
        row[wt_mc_completion] = utils.create_uqa_example(q_prefix + q, mc, row['explanation_sentences_final'])
        row[wt_od_completion] = utils.create_uqa_example(q_prefix + q, None, row['explanation_sentences_final'])
        row[wt_mc_ans] = utils.create_uqa_example(q, mc + '\\n' + row['explanation_sentences_final'], ans)
        row[wt_od_ans] = utils.create_uqa_example(q, row['explanation_sentences_final'], ans)
        outlist.append(row)
    return outlist


def save_single(split, outdir, ds_type, file):
    """ save a single dataset split """
    out = [s[ds_type] for s in split]
    outfile = os.path.join(outdir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))    
    return

def save_datasets(dev, test, train):
    """ save uqa-formatted dataset """
    for ds_type in [wt_mc_completion, wt_od_completion, wt_mc_ans, wt_od_ans]:
        outdir = uqa_dir + ds_type
        print(f'Saving dataset to {outdir} ...')
        os.makedirs(outdir, exist_ok=True)
        save_single(dev, outdir, ds_type, 'dev.tsv')
        save_single(test, outdir, ds_type, 'test.tsv')
        save_single(train, outdir, ds_type, 'train.tsv')
    print('Finished saving uqa-formatted explanation datasets!')
    return
        
def save_facts_dataset(out_dset, train_list, dev_list, devfile='dev.tsv'):
    """ Save self-supervised facts dataset
    """
    out_train = [utils.create_uqa_example(t, ' ', None, append_q_char='.') for t in train_list]
    out_dev = [utils.create_uqa_example(t, ' ', None, append_q_char='.') for t in dev_list]
    outdir = uqa_dir + out_dset
    print(f'Saving dataset to {outdir} ...')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(out_train))
    outfile = os.path.join(outdir, devfile)
    with open(outfile, 'w') as f:
        f.write(''.join(out_dev))    
    return

fact_dict = load_facts(explanation_dir)

questions_dev = load_questions(os.path.join(question_dir, 'questions.dev.tsv'), fact_dict, tokenizer)
questions_test = load_questions(os.path.join(question_dir, 'questions.test.tsv'), fact_dict, tokenizer)
questions_train = load_questions(os.path.join(question_dir, 'questions.train.tsv'), fact_dict, tokenizer)

save_datasets(questions_dev, questions_test, questions_train)

# build facts lists for self supervised datasets:
facts_dev =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_dev])))   #1839
facts_test =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_test]))) #4017
facts_train =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_train]))) #5198

facts_train_dev = list(set(facts_train+facts_dev)) #5889
facts_all = list(set(facts_train_dev+facts_test)) #7626

save_facts_dataset('testfactsonly'+selfsupervisedkey, facts_test, facts_dev, devfile='dev.tsv')
save_facts_dataset('all_dev_in_train'+selfsupervisedkey, facts_all, facts_dev, devfile='dev.tsv')
save_facts_dataset('dev_in_train_excl_test'+selfsupervisedkey, facts_train_dev, facts_dev, devfile='dev.tsv')


#TODO: FORMAT FOR GPT-J
#TODO: GENERATE PROMPT TEMPLATES WITH/WITHOUT TASK PROMPT  WITH/WITHOUT NUMBERING 
#TODO: FOR DIFFERENT K-SHOTS - GENERATE EG 100 and randomly sample from? Maybe can just load the dataset and use template on the fly!
#TODO: FOR GENERATING SINGLE FACT AT A TIME OR SETS?
#TODO: FOR GENERATING FROM DIFFT TEMPLATES FOR THE SAME QUESTION eg TEMPORAL, QUANTITY etc
#TODO: TAKE N MOST DIVERSE.
#TODO: USE EXISTING UQA MODEL as critic if judging by answer. - use this to compare supervised vs gpt-j generated versions
#TODO: Could score for plausibility and/or relevance - need to train a model to do this..


