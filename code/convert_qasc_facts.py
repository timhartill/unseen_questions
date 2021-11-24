#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:51:50 2021

@author: tim hartill

Convert QASC facts into explanation datasets

Note: making explanation the 2 gold facts, could alternatively have been the single gold fact but this usually has the answer in it as a single lookup.

"""
import os
import copy
import random

import utils
import text_processing

MAX_OUTPUT_TOKENS = 127 #max token size of total explanation text excl BOS & EOS tokens


QASC_DIR = '/home/thar011/data/qasc/QASC_Dataset/'

UQA_DIR = '/data/thar011/data/unifiedqa/qasc_'
MC_COMPLETION = 'mc_expl'
OD_COMPLETION = 'od_expl'
MC_ANS = 'mc_ans'
OD_ANS = 'od_ans'
Q_PREFIX = 'Add Explanation: '

tokenizer = utils.load_model(model_name="facebook/bart-large", loadwhat='tokenizer_only')

qasc_dev = utils.load_jsonl(os.path.join(QASC_DIR,'dev.jsonl'))
qasc_train = utils.load_jsonl(os.path.join(QASC_DIR,'train.jsonl'))

def create_explanation(split):
    """ Create explanation key """
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
        outdir = UQA_DIR + ds_type
        print(f'Saving dataset to {outdir} ...')
        os.makedirs(outdir, exist_ok=True)
        save_single(dev, outdir, ds_type, 'dev.tsv')
        save_single(train, outdir, ds_type, 'train.tsv')
    print('Finished saving uqa-formatted explanation datasets!')
    return

random.seed(42)
create_explanation(qasc_dev)
create_explanation(qasc_train)

save_datasets(qasc_dev, qasc_train)




