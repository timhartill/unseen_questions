#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:27:43 2021

@author: tim hartill

Preprocess CWWV and ATOMIC into cloze-style questions both with a specific span 
to be masked in training and without for fully self supervised training

eg for the first case  'The rain in [#Spain#] lies mainly on the plain.' will be preprocessed to 
'The rain in <mask> lies mainly on the plain.\\n ' with label: Spain

and for the second case the [# and #] will be stripped leaving standard self supervised task

"""
import json
import random
import os
import numpy as np


indir_cwwv = '/data/thar011/data/hykas-cskg/data/CWWV/'
indir_atomic = '/data/thar011/data/hykas-cskg/data/ATOMIC/'
uqa_dir = '/data/thar011/data/unifiedqa/'

selfsupervisedkey = "_selfsvised"   # dataset names ending in this will be processed as self supervised
force_ans_start = '[#'              # if self supervised, can force a specific mask by using eg 'The rain in [#Spain#] lies mainly on the plain.'
force_ans_end = '#]'

out_cwwv = 'cwwv_premask' + selfsupervisedkey
out_atomic = 'atomic_premask' + selfsupervisedkey
out_cskg = 'cskg_premask' + selfsupervisedkey

out_cwwv_ssvise = 'cwwv' + selfsupervisedkey
out_atomic_ssvise = 'atomic' + selfsupervisedkey
out_cskg_ssvise = 'cskg' + selfsupervisedkey

os.makedirs(os.path.join(uqa_dir, out_cwwv), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_atomic), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_cskg), exist_ok=True)

os.makedirs(os.path.join(uqa_dir, out_cwwv_ssvise), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_atomic_ssvise), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_cskg_ssvise), exist_ok=True)


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


def get_mask_subst(js):
    """ Get the actual answer for a [MASK] for cwwv
    """
    gt = ''
    for c in js['question']['choices']:
        if c['label'] == js['answerKey']:
            gt = c['text']
            break
    return gt


def replace_mask(text, ans):
    """ Replace [MASK] with [#answer#] """
    return text.replace('[MASK]', force_ans_start + ans + force_ans_end)


def create_uqa_input_cwwv(j):
    """ Create uqa-formatted input for a CWWV sample """
    ans = get_mask_subst(j)
    s = replace_mask(j['question']['stem'], ans)
    if len(s) <= 1:
        print(f"Unable to find question for: {j}")
    return f"{s}.\\n \n"


def create_uqa_input_atomic(a):
    """ Create uqa-formatted input for an atomic example """
    ans = a['candidates'][a['correct']]
    ans = ans[:len(ans)-1]  #strip fullstop
    sentences = a['context'].split('. ')
    h = sentences[0] + '.'
    h = h.replace('___', 'vvv')  # RACE sometimes uses underscores for different purpose
    s = replace_mask(sentences[1] + ' [MASK]', ans)
    if len(sentences) > 2:
        print(f"Sample: {a} context parsed into > 2 sentences.")
    if len(s) <= 1:
        print(f"Unable to find question for: {a}")  
    return f"{h} {s}.\\n \n"
    

def create_all(js, outfile, kgtype, include_count=-1):
    """ Create output file and output optionally restricting number of samples to include."""
    if include_count != -1:
        include_count = min(include_count, len(js))
        indices = np.arange(len(js))
        indices = set(np.random.permutation(indices)[:include_count])
    out = []
    for i, j in enumerate(js):
        if include_count == -1 or i in indices:
            if kgtype == 'cwwv':
                out.append( create_uqa_input_cwwv(j) ) 
            else:
                out.append( create_uqa_input_atomic(j) )
    random.shuffle(out)
    with open(outfile, 'w') as f:
        f.write(''.join(out))
    print(f"Completed output to {outfile}")
    
    out_ssvise = [o.replace(force_ans_start, '').replace(force_ans_end, '') for o in out]
    outfile_ssvise = outfile.replace('_premask', '')
    with open(outfile_ssvise, 'w') as f:
        f.write(''.join(out_ssvise))
    print(f"Completed output to {outfile_ssvise}")
    return out


random.seed(42)
np.random.seed(42)
cwwv_dev = load_jsonl( os.path.join(indir_cwwv, 'dev_random.jsonl') )
out_cwwv_dev = create_all(cwwv_dev, os.path.join(uqa_dir, out_cwwv, 'dev.tsv'), 'cwwv', include_count=2000 )
out_cwwv_dev_all = create_all(cwwv_dev, os.path.join(uqa_dir, out_cwwv, 'dev_all.tsv'), 'cwwv' )
cwwv_train = load_jsonl( os.path.join(indir_cwwv, 'train_random.jsonl') )
out_cwwv_train = create_all(cwwv_train, os.path.join(uqa_dir, out_cwwv, 'train.tsv'), 'cwwv' )

atomic_dev = load_jsonl(os.path.join(indir_atomic, 'dev_random.jsonl'))
out_atomic_dev = create_all(atomic_dev, os.path.join(uqa_dir, out_atomic, 'dev.tsv'), 'at', include_count=2000 )
out_atomic_dev_all = create_all(atomic_dev, os.path.join(uqa_dir, out_atomic, 'dev_all.tsv'), 'at' )
atomic_train = load_jsonl( os.path.join(indir_atomic, 'train_random.jsonl') )
out_atomic_train = create_all(atomic_train, os.path.join(uqa_dir, out_atomic, 'train.tsv'), 'at' )

out_cskg_dev = out_cwwv_dev + out_atomic_dev
outfile = os.path.join(uqa_dir, out_cskg, 'dev.tsv')
print(f"Outputting CSKG dev to {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(out_cskg_dev))
out_ssvise = [o.replace(force_ans_start, '').replace(force_ans_end, '') for o in out_cskg_dev]
outfile_ssvise = outfile.replace('_premask', '')
with open(outfile_ssvise, 'w') as f:
    f.write(''.join(out_ssvise))
print(f"Completed output to {outfile_ssvise}")


out_cskg_train = out_cwwv_train + out_atomic_train
random.shuffle(out_cskg_train)
outfile = os.path.join(uqa_dir, out_cskg, 'train.tsv')
print(f"Outputting CSKG train to {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(out_cskg_train))
out_ssvise = [o.replace(force_ans_start, '').replace(force_ans_end, '') for o in out_cskg_train]
outfile_ssvise = outfile.replace('_premask', '')
with open(outfile_ssvise, 'w') as f:
    f.write(''.join(out_ssvise))
print(f"Completed output to {outfile_ssvise}")    
    
print('Finished!')


