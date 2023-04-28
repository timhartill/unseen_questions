#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:34:42 2023

@author: tim hartill

Add previously generated Iter contexts from non-unique answer (19938 sample) version of musique train to unique-answer (2057 sample) version of musique train

Note: cannot do this shortcut for llm_expl_ans as only 10K mu train samples have llm explanations generated for them

"""

import os


import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR

mu_od_unique_ans_dir = '/data/thar011/data/unifiedqa/musique_qa'  # train: 2057, dev:382

mu_iter_non_unique_ans_dir = '/data/thar011/data/unifiedqa/musique_qa_fullwiki_bs60' # train: 19556, dev: 382
mu_iter_maxp4_non_unique_ans_dir = '/data/thar011/data/unifiedqa/musique_qa_fullwiki_bs60_maxp4' # train: 19556, dev: 382

mu_unique_dev = utils.load_uqa_supervised(os.path.join(mu_od_unique_ans_dir, 'dev.tsv'), ans_lower=False, return_parsed=True)
mu_unique_train = utils.load_uqa_supervised(os.path.join(mu_od_unique_ans_dir, 'train.tsv'), ans_lower=False, return_parsed=True)

mu_unique_dev_dict = {s['q_only'].rstrip().rstrip('?!:. ').lstrip(): s for s in mu_unique_dev}  # 382 occasionally ending punctuation differences cause mismatches so strip all ending punctuation
mu_unique_train_dict = {s['q_only'].rstrip().rstrip('?!:. ').lstrip(): s for s in mu_unique_train}  # 2056 1 dup


def filter_to_unique(split, lookup):
    """ filter input split to only samples found in lookup dict 
    """
    foundset = set()
    outlist = []
    for s in split:
        q = s['q_only'].rstrip().rstrip('?!:. ').lstrip()
        if lookup.get(q) is not None and q not in foundset:
            outlist.append( utils.create_uqa_example(s['q_only'], s['context'], s['answer']) )
            foundset.add(q)
    print(f"Input samples:{len(split)}  Lookup samples:{len(lookup)}  Output samples:{len(outlist)}")
    return outlist


mu_iter_dev = utils.load_uqa_supervised(os.path.join(mu_iter_non_unique_ans_dir, 'dev.tsv'), ans_lower=False, return_parsed=True)
mu_iter_train = utils.load_uqa_supervised(os.path.join(mu_iter_non_unique_ans_dir, 'train.tsv'), ans_lower=False, return_parsed=True)

mu_iter_unique_dev = filter_to_unique(mu_iter_dev, mu_unique_dev_dict) # Input samples:382  Lookup samples:382  Output samples:382
mu_iter_unique_train = filter_to_unique(mu_iter_train, mu_unique_train_dict) # Input samples:19556  Lookup samples:2056  Output samples:2056

out_dset_dir = os.path.join(UQA_DIR, 'musique_qa_fullwiki_bs60_unique_answer')
utils.save_uqa(mu_iter_unique_dev, out_dset_dir, 'dev.tsv')
utils.save_uqa(mu_iter_unique_train, out_dset_dir, 'train.tsv')


# max 4 para version..
mu_iter_dev = utils.load_uqa_supervised(os.path.join(mu_iter_maxp4_non_unique_ans_dir, 'dev.tsv'), ans_lower=False, return_parsed=True)
mu_iter_train = utils.load_uqa_supervised(os.path.join(mu_iter_maxp4_non_unique_ans_dir, 'train.tsv'), ans_lower=False, return_parsed=True)

mu_iter_unique_dev = filter_to_unique(mu_iter_dev, mu_unique_dev_dict) # Input samples:382  Lookup samples:382  Output samples:382
mu_iter_unique_train = filter_to_unique(mu_iter_train, mu_unique_train_dict) # Input samples:19556  Lookup samples:2056  Output samples:2056

out_dset_dir = os.path.join(UQA_DIR, 'musique_qa_fullwiki_bs60_maxp4_unique_answer')
utils.save_uqa(mu_iter_unique_dev, out_dset_dir, 'dev.tsv')
utils.save_uqa(mu_iter_unique_train, out_dset_dir, 'train.tsv')


