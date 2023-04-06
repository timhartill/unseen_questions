#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:35:44 2021

@author: tim hartill

Import and convert WorldTree data into explanation datasets and self-supervised datasets of facts
Import and convert ARC Direct answer (ARC-DA), incorporating explantions from Worldtree

Zhengnan Xie, Sebastian Thiem, Jaycie Martin, Elizabeth Wainwright, Steven Marmorstein, and Peter Jansen. 2020. 
WorldTree V2: A corpus of science-domain structured explanations and inference patterns supporting multi-hop inference. 
In Proceedings of the 12th Language Resources and Evaluation Conference, pages 5456-5473

Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark. 2021. 
Think you have solved direct-answer question answering? Try ARC-DA, the direct-answer AI2 Reasoning Challenge. 
arXiv:2102.03315 [cs.CL].

1. Download and unzip Worldtree: http://www.cognitiveai.org/dist/WorldtreeExplanationCorpusV2.1_Feb2020.zip
1.1 Download and unzip ARC-DA: https://allenai.org/data/arc-da 
2. Update directory variables below
3. Run. New datasets in UQA format with explanations as labels will be in .../uqa_dir/worldtree_mc_expl and .../uqa_dir/worldtree_od_expl

ARC-DA output datsets are filtered to only those samples for which a matching explanation can be found in Worldtree
outputs:
    arc_da_expl_ans: q+explanation->a   # FILTERED to only those samples for which a matching explanation can be found in Worldtree
    arc_da_od_ans: q->a                 # NOT FILTERED - MATCHES ORIG ARCDA SAMPLES
    arc_da_od_expl: q->explanation      # FILTERED to only those samples for which a matching explanation can be found in Worldtree


Added rationale processing for rr model training:
    
    Output format:
    [ {'question': 'question text EXCLUDING MC options and preceding initial ctxt if any',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'worldtree',
       'pos_paras': [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]], 'mc_only': False}, ...],
       'neg_paras': [], #Same format as pos_paras but filled in later
       'mc_options':  '(A) banana (B) ...'  #key only present if multichoice options exist...
       'context': 'An initial para or other necessary context if exists'  #key only present if initial para exists...
       }, {...}, ..
     
    ]



"""

import os
import pandas as pd
import copy
import random
import numpy as np

import eval_metrics
import utils
import text_processing


MAX_OUTPUT_TOKENS = 127 #max token size of total explanation text excl BOS & EOS tokens

MAX_TRAIN_SAMPLES = 4

arcda_dir = '/home/thar011/data/arc-da/ARC-DA-v1.1/'
arcda_uqa_dir = '/data/thar011/data/unifiedqa/arc_da_'

worldtree_dir = '/home/thar011/data/worldtree/WorldtreeExplanationCorpusV2.1_Feb2020/'
explanation_dir = os.path.join(worldtree_dir, 'tablestore/v2.1/tables/')
question_dir = os.path.join(worldtree_dir, 'questions/')

uqa_dir = '/data/thar011/data/unifiedqa/worldtree_'
wt_mc_completion = 'mc_expl'
wt_od_completion = 'od_expl'
wt_mc_ans = 'mc_expl_ans'
wt_od_ans = 'od_ans'
wt_expl_ans = 'expl_ans'
q_prefix= 'Add Explanation: '
selfsupervisedkey = '_selfsvised'

tokenizer = utils.load_model(model_name="facebook/bart-large", loadwhat='tokenizer_only')


### Rationale processing file names and vars below:

# Note preceding/trailing spaces.. Should be the same as in utils.create_additional_pos_for_mc(..)
prepend_list = ['The answer must be ****. ', 'The answer is ****. ', 'The answer must be something like ****. ', 'The answer must be something that involves ****. ']
append_list = [' Thus, of the choices **** is the best answer.', ' Thus, of the choices, **** is the best answer.', ' Thus of the choices **** is the best answer.', ' Thus, of the choices, **** is the answer.', ' Thus, of the choices **** is the correct answer.',
               ' Of the choices, **** is the best answer.', ' Of the choices **** is the best answer.', ' Of the choices **** is the correct answer.', ' Of the choices, **** is the answer.',
               ' Of the above choices, **** is the best answer.', ' Of the above choices **** is the best answer.',
               ]


UQA_DIR = eval_metrics.UQA_DIR  # different to uqa_dir above...
WT_INPUT_DIR = os.path.join(UQA_DIR, 'worldtree_mc_expl_ans')  # For simplicity we reload the already processed worldtree explanations from the tsv formatted dataset 

# LLM neg rationales - augmentable means can add "Thus of the choices.." etc augmented samples, the others already have this text
file_rr_dev_negs_augmentable = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T15_MCFOCUS_WORLDTREE_DEV_all_onv3-02-20-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json', 
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T32v2_YN_WORLDTREE_DEV_onv6_sample-03-22-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T33v2_YN_WORLDTREE_DEV_onv6mod2_sample-03-22-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
               ]
file_rr_dev_negs = [
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T17_MCFOCUS_WORLDTREE_DEV_all_onv6-02-21-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                    '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T19_MCFOCUS_WORLDTREE_DEV_all_onv8-02-22-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
               ]

file_rr_dev_negs_nofilter = []  # unlike QASC, WT on v8 prompt sometimes leaks the answer so apply em filter as with other prompt results

file_rr_train_negs_augmentable = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T16_MCFOCUS_WORLDTREE_TRAIN_all_onv3-02-20-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T34v2_YN_WORLDTREE_TRAIN_onv6_sample-03-20-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T35v2_YN_WORLDTREE_TRAIN_onv6mod2_sample-03-22-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 ]
file_rr_train_negs = [
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T18_MCFOCUS_WORLDTREE_TRAIN_all_onv6-02-21-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                      '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T20_MCFOCUS_WORLDTREE_TRAIN_all_onv8-02-22-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 ]

file_rr_train_negs_nofilter = []

# the full final sample files with potentially multiple pos and negs including samples without negs:
rr_dev = worldtree_dir + 'worldtree_dev_rr_all_pos_neg.jsonl'
rr_train = worldtree_dir + 'worldtree_train_rr_all_pos_neg.jsonl'
# the full final sample files with potentially multiple pos and negs excluding samples without negs:
rr_dev_exclposonly = worldtree_dir + 'worldtree_dev_rr_all_pos_neg_exclposonly.jsonl'
rr_train_exclposonly = worldtree_dir + 'worldtree_train_rr_all_pos_neg_exclposonly.jsonl'




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
    """ save a single dataset split ignoring entries where a match to WT explanation wasnt found """
    out = [s[ds_type] for s in split if s[ds_type] is not None]
    out = utils.flatten(out)
    outfile = os.path.join(outdir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))    
    return

def save_datasets(dev, test, train, 
                  ds_list=[wt_mc_completion, wt_od_completion, wt_mc_ans, wt_od_ans],
                  dir_ = uqa_dir):
    """ save uqa-formatted dataset """
    for ds_type in ds_list:
        outdir = dir_ + ds_type
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


def process_arcda(arcda, wt_all, dset_type='train'):
    """ Add Worldtree explanation to arcda jsonl
    """
    no_match = 0
    for a in arcda:
        a['qid_wt'] = '_'.join(a['question_id'].split('_')[1:])
        a['wt_match'] = wt_all.get( a['qid_wt'] )
        q = a['question']
        ans = [aa.replace('\n', '') for aa in a['answers']]
        sel_indices = set(np.random.choice(len(ans), min(MAX_TRAIN_SAMPLES, len(ans)), replace=False))

        if a['wt_match'] is None:
            no_match += 1
            a['explanation_sentences_final'] = None
            a[wt_od_completion] = None
            a[wt_expl_ans] = None
        else:
            a['explanation_sentences_final'] = a['wt_match']['explanation_sentences_final']
            a[wt_od_completion] = utils.create_uqa_example(q_prefix + q, None, a['explanation_sentences_final'])
            if dset_type != 'train':
                a[wt_expl_ans] = utils.create_uqa_example(q, a['explanation_sentences_final'], ans)
            else:
                a[wt_expl_ans] = [utils.create_uqa_example(q, a['explanation_sentences_final'], ans_single) for i, ans_single in enumerate(ans) if i in sel_indices]
        if dset_type != 'train':
            a[wt_od_ans] = utils.create_uqa_example(q, None, ans)
        else:
            a[wt_od_ans] = [utils.create_uqa_example(q, None, ans_single) for i, ans_single in enumerate(ans) if i in sel_indices]
            
    print(f"Total count: {len(arcda)}  No match: {no_match}")
            
    return
    

###########################
# Output full open domain versions of arc datasets ie not filtered to samples with WT explanations
###########################

def output_arcda_full_od(split, file='train.tsv'):
    """ Output full open domain versions of arc datasets ie not filtered to samples with WT explanations
    
     - DUPLICATES arcda_od_ans but slight changes (only useful if training on ARCDA so not currently used):
            for train.tsv outputs a smaple for each answer not constrained to MAX_TRAIN_SAMPLES
            if any sample only has 1 answer, outputs a string answer rather that a list
    
    {   'question_id': 'ARCEZ_MCAS_1998_4_7',
        'tag': 'EASY-TEST',
        'question': 'What is the cause of most earthquakes?',
        'answers': ['plates shifting',
         'tectonic shifts',
         'two plates are rubbing against each other',
         'tectonic plates',
         "the movement of plates on Earth's crust"]}
    """    
    outlist = []
    for s in split:
        q = s['question']
        if file != 'train.tsv':
            if len(s['answers']) == 1:
                ans = s['answers'][0]
            else:
                ans = s['answers']
            outlist.append(utils.create_uqa_example(q, None, ans))
        else:
            for ans in s['answers']:
                outlist.append(utils.create_uqa_example(q, None, ans))
    out_dir = arcda_uqa_dir + 'unfiltered_od_ans'
    utils.save_uqa(outlist, out_dir, file)
    return
            


##########################
# Build expl and output Worldtree
##########################

fact_dict = load_facts(explanation_dir)

questions_dev = load_questions(os.path.join(question_dir, 'questions.dev.tsv'), fact_dict, tokenizer)
questions_test = load_questions(os.path.join(question_dir, 'questions.test.tsv'), fact_dict, tokenizer)
questions_train = load_questions(os.path.join(question_dir, 'questions.train.tsv'), fact_dict, tokenizer)

save_datasets(questions_dev, questions_test, questions_train)  # save worldtree datasets

# build facts lists for self supervised datasets:
facts_dev =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_dev])))   #1839
facts_test =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_test]))) #4017
facts_train =  list(set(utils.flatten([s['explanation_sentences'] for s in questions_train]))) #5198

facts_train_dev = list(set(facts_train+facts_dev)) #5889
facts_all = list(set(facts_train_dev+facts_test)) #7626

save_facts_dataset('testfactsonly'+selfsupervisedkey, facts_test, facts_dev, devfile='dev.tsv')
save_facts_dataset('all_dev_in_train'+selfsupervisedkey, facts_all, facts_dev, devfile='dev.tsv')
save_facts_dataset('dev_in_train_excl_test'+selfsupervisedkey, facts_train_dev, facts_dev, devfile='dev.tsv')

############################
# Output ARC-DA
############################

arcda_dev = utils.load_jsonl(os.path.join(arcda_dir, 'dev.jsonl'))
arcda_test = utils.load_jsonl(os.path.join(arcda_dir, 'test.jsonl'))
arcda_train = utils.load_jsonl(os.path.join(arcda_dir, 'train.jsonl'))

output_arcda_full_od(arcda_train, file='train.tsv')  # 4246
output_arcda_full_od(arcda_dev, file='dev.tsv')  # 338
output_arcda_full_od(arcda_test, file='test.tsv')  # 1397


wt_all = utils.build_dict_from_jsonl(questions_dev, key='QuestionID')
wt_all.update(utils.build_dict_from_jsonl(questions_test, key='QuestionID'))
wt_all.update(utils.build_dict_from_jsonl(questions_train, key='QuestionID'))

np.random.seed(42)
process_arcda(arcda_dev, wt_all, dset_type='dev')  # Total count: 338  No match: 119
process_arcda(arcda_test, wt_all, dset_type='dev')  # Total count: 1397  No match: 665
process_arcda(arcda_train, wt_all, dset_type='train')  # Total count: 1250  No match: 337

save_datasets(arcda_dev, arcda_test, arcda_train, dir_ = arcda_uqa_dir,
              ds_list=[wt_od_completion, wt_expl_ans, wt_od_ans])  # save arc-da datasets


########################################################
# Below is processing LLM rationales "rr formats"
########################################################

wt_dev = utils.load_uqa_supervised(os.path.join(WT_INPUT_DIR, 'dev.tsv'), ans_lower=False, return_parsed=True) # 496
wt_train = utils.load_uqa_supervised(os.path.join(WT_INPUT_DIR, 'train.tsv'), ans_lower=False, return_parsed=True)  #2206


dev_rr_format = [utils.create_rr_format(s['q_only'], s['context'], s['answer'],
                                        sentence_spans=None, _id=str(i), src='worldtree', append_q_char='?',
                                        mc_options=s['mc_options']) for i,s in enumerate(wt_dev)]
train_rr_format = [utils.create_rr_format(s['q_only'], s['context'], s['answer'],
                                        sentence_spans=None, _id=str(i), src='worldtree', append_q_char='?',
                                        mc_options=s['mc_options']) for i,s in enumerate(wt_train)]

random.seed(42)
utils.create_additional_rat_for_mc(dev_rr_format, include_prepend=0.7, include_append=0.9, include_both=0.35)
utils.create_additional_rat_for_mc(train_rr_format, include_prepend=0.7, include_append=0.9, include_both=0.35)

# load and filter augmentable negs
dev_rr_format = utils.load_merge_negs(dev_rr_format, file_rr_dev_negs_augmentable, overlap_method='em')  # 'f1' more restrictive but found em worked well and yields more samples
train_rr_format = utils.load_merge_negs(train_rr_format, file_rr_train_negs_augmentable, overlap_method='em')  # 'f1' more restrictive but found em worked well and yields more samples

#utils.saveas_jsonl(dev_rr_format, rr_dev)
#utils.saveas_jsonl(train_rr_format, rr_train)

#dev_rr_format = utils.load_jsonl(rr_dev) #496
#train_rr_format = utils.load_jsonl(rr_train) #2199

#augment
random.seed(42)
utils.create_additional_rat_for_mc(dev_rr_format, include_prepend=1.0, include_append=1.0, include_both=0.25, key='neg_paras')
utils.create_additional_rat_for_mc(train_rr_format, include_prepend=1.0, include_append=1.0, include_both=0.25, key='neg_paras')

#load and filter non-augmentable negs
dev_rr_format = utils.load_merge_negs(dev_rr_format, file_rr_dev_negs, overlap_method='em')  # 'f1' more restrictive but found em worked well and yields more samples
train_rr_format = utils.load_merge_negs(train_rr_format, file_rr_train_negs, overlap_method='em')  # 'f1' more restrictive but found em worked well and yields more samples

utils.saveas_jsonl(dev_rr_format, rr_dev)       #496
utils.saveas_jsonl(train_rr_format, rr_train)   #2199

utils.output_neg_tsv(dev_rr_format, os.path.join(UQA_DIR, 'worldtree_neg_expl_ans'), 'dev.tsv')
utils.output_neg_tsv(train_rr_format, os.path.join(UQA_DIR, 'worldtree_neg_expl_ans'), 'train.tsv')

# save final rr model creak training dataset - only output where negs exist which is all of them in this case but for consistency and debug..
utils.output_rr_where_negs_exist(dev_rr_format, outfile=rr_dev_exclposonly)   #496
utils.output_rr_where_negs_exist(train_rr_format, outfile=rr_train_exclposonly) #2199




