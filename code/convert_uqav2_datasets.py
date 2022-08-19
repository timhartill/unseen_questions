#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 18:08:10 2022

@author: tim hartill

convert uqa v2 datasets and copy into uqa dir

essentially just renames files downloaded from uqa cloud bucket* 
also translates any (title) forms into title:
    
* https://console.cloud.google.com/storage/browser/unifiedqa?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false


Only copy the test set if it is labelled. Note pubmed 'test' renamed 'dev.tsv''

"""

import os
import json
import random

import eval_metrics
import utils

UQA_DIR = eval_metrics.UQA_DIR
IN_DIR = '/home/thar011/data/unifiedqa_v2_new_datasets/'

IN_DIR_DS = os.path.join(IN_DIR, 'csqa2')
OUT_DIR_DS = os.path.join(UQA_DIR, 'csqa2')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_csqa2_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_csqa2_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'qaconv')
OUT_DIR_DS = os.path.join(UQA_DIR, 'qaconv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_qaconv_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_qaconv_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_qaconv_test.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'test.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'quail')
OUT_DIR_DS = os.path.join(UQA_DIR, 'quail')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_quail_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_quail_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_quail_test.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'test.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'reclor')
OUT_DIR_DS = os.path.join(UQA_DIR, 'reclor')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_reclor_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_reclor_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'record_extractive')
OUT_DIR_DS = os.path.join(UQA_DIR, 'record_extractive')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_record_extractive_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_record_extractive_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'tweetqa')
OUT_DIR_DS = os.path.join(UQA_DIR, 'tweetqa')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_tweetqa_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_tweetqa_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')


def reformat_title(q):
    """ translate question sentence? (title_with_underscore) paragraph text.  -> translate question sentence? title with underscore: paragraph text.
    """
    firstleft = q.find('(')
    firstright = q.find(')')
    if firstleft == -1 or firstright == -1 or firstleft >= firstright:
        return q
    oldtitle = q[firstleft:firstright+1]
    newtitle = oldtitle.replace('_',' ').replace('(','').replace(')',':')
    return q.replace(oldtitle, newtitle, 1)
    

IN_DIR_DS = os.path.join(IN_DIR, 'adversarialqa_dbert')
OUT_DIR_DS = os.path.join(UQA_DIR, 'adversarialqa_dbert')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_dbert_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_dbert_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'adversarialqa_droberta')
OUT_DIR_DS = os.path.join(UQA_DIR, 'adversarialqa_droberta')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_droberta_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_droberta_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

IN_DIR_DS = os.path.join(IN_DIR, 'adversarialqa_dbidaf')
OUT_DIR_DS = os.path.join(UQA_DIR, 'adversarialqa_dbidaf')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_dbidaf_dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'dev.tsv')
qlist, alist = utils.load_uqa_supervised(os.path.join(IN_DIR_DS, 'data_adversarialqa_dbidaf_train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
qlist = [reformat_title(q) for q in qlist]
utils.save_uqa(utils.convert_q_a_to_uqalist(qlist, alist), OUT_DIR_DS, 'train.tsv')

# create combo of all three:
dev = []
train = []
for dset in ['adversarialqa_dbert', 'adversarialqa_droberta', 'adversarialqa_dbidaf']:
    indir = os.path.join(UQA_DIR, dset)
    qlist, alist = utils.load_uqa_supervised(os.path.join(indir, 'dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
    dev += utils.convert_q_a_to_uqalist(qlist, alist)
    qlist, alist = utils.load_uqa_supervised(os.path.join(indir, 'train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
    train += utils.convert_q_a_to_uqalist(qlist, alist)
OUT_DIR_DS = os.path.join(UQA_DIR, 'adversarialqa_all')
utils.save_uqa(dev, OUT_DIR_DS, 'dev.tsv')
utils.save_uqa(train, OUT_DIR_DS, 'train.tsv')
    
# convert title format for existing squad
for dset in ['squad1_1', 'squad2']:
    OUT_DIR_DS = os.path.join(UQA_DIR, dset + '_titlereformat')
    indir = os.path.join(UQA_DIR, dset)
    qlist, alist = utils.load_uqa_supervised(os.path.join(indir, 'dev.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
    qlist = [reformat_title(q) for q in qlist]
    dev = utils.convert_q_a_to_uqalist(qlist, alist)
    utils.save_uqa(dev, OUT_DIR_DS, 'dev.tsv')
    qlist, alist = utils.load_uqa_supervised(os.path.join(indir, 'train.tsv' ), ans_lower=False, verbose=True, return_parsed=False)
    qlist = [reformat_title(q) for q in qlist]
    train = utils.convert_q_a_to_uqalist(qlist, alist)
    utils.save_uqa(train, OUT_DIR_DS, 'train.tsv')







