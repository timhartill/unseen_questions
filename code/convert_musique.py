#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:10:14 2021

@author: tim hartill

Convert Musique (ans) into uqa formatted datasets 

Notes:
1.  Concerting "ans" versions only, not "full" versions (which contain unanswerable questions where one+ of the decomp steps is unanswerable since no supporting para given)
    musique_ans_v0.1_dev.jsonl is constructed to not have answer overlap with musique_ans_v0.1_train.jsonl
    Therefore we treat musique_ans_v0.1_dev.jsonl as an OOD eval only dataset but we create a training dataset from the decomps + paras
    and construct a separate in-domain dev split from musique_ans_v0.1_train.jsonl
2. label format with decomp steps: "<s> the answer ## decomp step 1 #1:decomp ans ## decomp step 2 #2:decomp ans ## decomp step 3 #3:decomp ans </s>"


"""
import os
import numpy as np
from utils import load_jsonl, create_uqa_example, format_decomp_ans, load_model, string_to_ids
from text_processing import white_space_fix



UQA_DIR = '/data/thar011/data/unifiedqa/'

MU_DIR_IN = '/data/thar011/data/musique/musique_v0.1/'
MU_TRAIN_FILE = 'musique_ans_v0.1_train.jsonl'
MU_DEV_FILE = 'musique_ans_v0.1_dev.jsonl'


#MUFULL_TRAIN_FILE = 'musique_full_v0.1_train.jsonl'
#mufull_train = load_jsonl(os.path.join(MU_DIR_IN, MUFULL_TRAIN_FILE))

mu_train = load_jsonl(os.path.join(MU_DIR_IN, MU_TRAIN_FILE))
mu_dev = load_jsonl(os.path.join(MU_DIR_IN, MU_DEV_FILE))


tst = mu_train[0]
tst.keys()  # dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'])
tst['id'] # '2hop__482757_12019'
tst['paragraphs'][0].keys() # dict_keys(['idx', 'title', 'paragraph_text', 'is_supporting'])
tst['paragraphs'][0]['idx'] # 0
tst['paragraphs'][0]['paragraph_text']
tst['question'] # overall question
tst['answer'] # overall answer
tst['answer_aliases']

tst['question_decomposition']
tst['question_decomposition'][0].keys() # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
tst['question_decomposition'][0]['question']
tst['question_decomposition'][0]['answer']  # last answer same as tst['answer']
tst['question_decomposition'][0]['paragraph_support_idx']
tst['paragraphs'][ tst['question_decomposition'][0]['paragraph_support_idx'] ]['paragraph_text']
tst['paragraphs'][ tst['question_decomposition'][0]['paragraph_support_idx'] ]['title']


def retrieve_paras(mu_sample):
    """ Return list of paragraphs supporting each decomp step (1 para per decomp step so indices match)
    """
    paras = []
    for decomp in mu_sample['question_decomposition']:
        para_idx = decomp['paragraph_support_idx']
        if para_idx is not None:
            para = mu_sample['paragraphs'][para_idx]['paragraph_text'].strip()
            if para[-1] not in ['.', '?', '!']:
                para += '.'
            paras.append(para)
        else:
            paras.append('')  # for musique full, keep paras/decomps idxs aligned
    return paras

#tokenizer = load_model(loadwhat='tokenizer_only')
#tok_counts = []
#for i, mu_sample in enumerate(mu_train):
#    question = mu_sample['question']
#    paras = retrieve_paras(mu_sample)
#    context = white_space_fix(' '.join(paras))
#    input_string = create_uqa_example(question, context)    
#    ids = string_to_ids(input_string, tokenizer, verbose=False)
#    tok_counts.append( len(ids) )
#    if i % 1000 == 0:
#        print(f'Processed: {i}')
#tok_counts_np = np.array(tok_counts)
#print(f"count:{len(tok_counts)} max:{tok_counts_np.max()} mean:{tok_counts_np.mean()}") # count:19938 max:512 mean:299.1696759955863
#hittoklimit = np.where(tok_counts_np >= 512)
#hittoklimit[0].shape  # (1928,)  Approx 10% will be truncated..

#d_max = -1
#for i, mu_sample in enumerate(mu_train):
#    d_num = len(mu_sample['question_decomposition'])
#    if mu_sample['question_decomposition'][0]['question'].find('#9') != -1:
#        print(mu_sample)
#    if d_num > d_max:
#        d_max = d_num
#print(f"Max decompositions: {d_max}")


def process_musique(mu_data, dev_indices=[]):
    """ Retrieve, format and create new keys for outputting all the datasets, namely:
    decomp learning:
    musique_decomp_train: decomp q \\n para \t decomp ans                   train=decomps from mu train; dev=decomps from new dev (setting where qa dataset train can see it's facts but dev can't)
    musique_decomp_new_dev_in_train: decomp q \\n para \t decomp ans        train=decomps from mu train + decomps from new dev; dev=decomps from new dev (setting where qa dataset is allowed to have seen train & dev facts)
    musique_mu_dev_decomp: decomp q \\n para \t decomp ans                  train=decomps from musique dev; dev=same (dev is just to check how well decomps are learned)
    musique_decomp_all_dev_in_train decomp q \\n para \t decomp ans         train=decomps from mu train + decomps from new dev + decomps from musique dev; dev=decomps from musique dev
    
    qa learning:
    musique_qa: q \\n \t ans                                                train=mu train, dev=new dev from mu train
    musique_qa_paras: q \\n paras \t ans                                    train=mu train, dev=new dev from mu train
    musique_qa_decomp_ans: q \\n \t ans ## decomps+ans ## ..                train=mu train, dev=new dev from mu train
    musique_qa_paras_decomp_ans: q \\n paras \t ans  ## decomps+ans ## ..   train=mu train, dev=new dev from mu train
    musique_mu_dev_qa: q \\n \t ans                                                dev only = musique dev
    musique_mu_dev_qa_paras: q \\n paras \t ans                                    dev only = musique dev
    musique_mu_dev_qa_decomp_ans: q \\n \t ans ## decomps+ans ## ..                dev only = musique dev
    musique_mu_dev_qa_paras_decomp_ans: q \\n paras \t ans  ## decomps+ans ## ..   dev only = musique dev

    """
    for i, mu_sample in enumerate(mu_data):
        if dev_indices==[] or i in dev_indices:
            mu_sample['split'] = 'dev'
        else:    
            mu_sample['split'] = 'train'
        question = mu_sample['question'].strip()
        answer = mu_sample['answer'].strip()
        paras = retrieve_paras(mu_sample)
        mu_sample['context_paras'] = white_space_fix(' '.join(paras))
        decomp_ans_str = ''
        prior_answers = []
        for j, decomp_step in enumerate(mu_sample['question_decomposition']):   # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
            this_ans_str, prior_answers, subst_decomp = format_decomp_ans(decomp_step['question'], decomp_step['answer'], j, prior_answers)
            decomp_ans_str += this_ans_str
            decomp_step['question_subst'] = subst_decomp
            para_idx = decomp_step['paragraph_support_idx']
            if para_idx is not None:
                decomp_para = mu_sample['paragraphs'][para_idx]['paragraph_text'].strip()
                if decomp_para[-1] not in ['.', '?', '!']:
                    decomp_para += '.'
            else:
                decomp_para = ''
            decomp_step['context_para'] = decomp_para
            
        mu_sample['decomp_ans_str'] = answer + decomp_ans_str
        if i % 1000 == 0:
            print(f'Processed: {i}')
    return


def get_facts_datasets(mu_data):
    """ Output facts datasets:
    musique_decomp_train: decomp q \\n para \t decomp ans                   train=decomps from mu train; dev=decomps from new dev (setting where qa dataset train can see it's facts but dev can't)
    musique_decomp_new_dev_in_train: decomp q \\n para \t decomp ans        train=decomps from mu train + decomps from new dev; dev=decomps from new dev (setting where qa dataset is allowed to have seen train & dev facts)
    musique_mu_dev_decomp: decomp q \\n para \t decomp ans                  train=decomps from musique dev; dev=same (dev is just to check how well decomps are learned) 
    musique_decomp_all_dev_in_train decomp q \\n para \t decomp ans         train=decomps from mu train + decomps from new dev + decomps from musique dev; dev=decomps from musique dev
    """
    train_list = []
    dev_list = []
    for i, mu_sample in enumerate(mu_data):
        for j, decomp_step in enumerate(mu_sample['question_decomposition']):   # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx', 'question_subst', 'context_para'])
            sample = create_uqa_example(decomp_step['question_subst'], decomp_step['context_para'], decomp_step['answer'].strip() )    
            if mu_sample['split'] == 'dev':
                dev_list.append(sample)
            else:
                train_list.append(sample)
        if i % 1000 == 0:
            print(f'Processed: {i}')         
    return train_list, dev_list







num_q = len(mu_train)
dev_size = int(num_q*0.1)
np.random.seed(42)
dev_indices = np.random.choice(num_q, dev_size, replace=False)
process_musique(mu_train, dev_indices)
train_list, dev_list = get_facts_datasets(mu_train)
both_list = train_list + dev_list
print(f'Train: {len(train_list)}  Dev: {len(dev_list)}  Dev in train:{len(both_list)}')  # Train: 41990  Dev: 4623 Dev in train:46613

outdir = os.path.join(UQA_DIR, 'musique_decomp_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(train_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))

outdir = os.path.join(UQA_DIR, 'musique_decomp_new_dev_in_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(both_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))


process_musique(mu_dev, dev_indices=[])
train_list, dev_list = get_facts_datasets(mu_dev)
print(f'Train: {len(train_list)}  Dev: {len(dev_list)}')  # Train: 0  Dev: 6404

outdir = os.path.join(UQA_DIR, 'musique_mu_dev_decomp')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))

both_list = both_list + dev_list  #All facts both mu train + new dev + mu dev
print(len(both_list))  # 53017
outdir = os.path.join(UQA_DIR, 'musique_decomp_all_dev_in_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(both_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))






