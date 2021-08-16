#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:27:43 2021

@author: tim hartill

Preprocess CWWV and ATOMIC

"""
import json
import random
import os
import numpy as np

import text_processing

indir_cwwv = '/data/thar011/data/hykas-cskg/data/CWWV/'
indir_atomic = '/data/thar011/data/hykas-cskg/data/ATOMIC/'
uqa_dir = '/data/thar011/data/unifiedqa/'

out_cwwv = 'cwwv'
out_atomic = 'atomic'
out_cskg = 'cskg'

os.makedirs(os.path.join(uqa_dir, out_cwwv), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_atomic), exist_ok=True)
os.makedirs(os.path.join(uqa_dir, out_cskg), exist_ok=True)


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


def replace_mask(text):
    """ Replace [MASK] with NOUNPHRASE """
    return text.replace('[MASK]', text_processing.NOUNPHRASE_LABEL)


def create_uqa_input_cwwv(j):
    """ Create uqa-formatted input for a CWWV sample """
    method = random.choice(['id', 'rq', 'wba'])
    s = replace_mask(j['question']['stem'])
    q = text_processing.make_into_question(s, method=method, mask_type=text_processing.NOUNPHRASE_LABEL) 
    if len(q) <= 1:
        print(f"Unable to find question for: {j} using method {method}")
    ans = get_mask_subst(j)
    choices = ''.join([' (' + c['label'] + ') ' + c['text'] for c in j['question']['choices'] ])
    return f"{q} \\n{choices}\t{ans}\n"


def create_uqa_input_atomic(a):
    """ Create uqa-formatted input for an atomic example """
    method = random.choice(['id', 'rq', 'wba'])
    sentences = a['context'].split('. ')
    h = sentences[0] + '.'
    h = h.replace('___', 'vvv')  # RACE sometimes uses underscores for different purpose
    s = replace_mask(sentences[1] + ' [MASK]')
    if len(sentences) > 2:
        print(f"Sample: {a} context parsed into > 2 sentences.")
    q = text_processing.make_into_question(s, method=method, mask_type=text_processing.NOUNPHRASE_LABEL)
    if len(q) <= 1:
        print(f"Unable to find question for: {a} using method {method}")  
    ans = a['candidates'][a['correct']]
    ans = ans[:len(ans)-1]  #strip fullstop
    choices = ''.join([' (' + chr(i+65) + ') ' + c[:len(c)-1] for i, c in enumerate(a['candidates']) ])
    return f"{q} \\n{choices} \\n {h}\t{ans}\n"
    

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

out_cskg_train = out_cwwv_train + out_atomic_train
random.shuffle(out_cskg_train)
outfile = os.path.join(uqa_dir, out_cskg, 'train.tsv')
print(f"Outputting CSKG train to {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(out_cskg_train))
print('Finished!')


""" Test fns
def recompose(js):
    # Recompose question by substituting correct answer for mask for cwwv
    ans = get_mask_subst(js)
    recomp = js['question']['stem'].replace('[MASK]', ans)
    return recomp


def get_relations_and_templates(js):
    # Return list of relations and templates in list of jsons for cwwv
    relations = set()
    templates = set()
    both = set()
    for j in js:
        t = j['question']['template']
        r = j['question']['relation']
        b = (r,t)
        relations.add(r)
        templates.add(t)
        both.add(b)
    return relations, templates, both


def mq(text):
    # wrapper for question making tests 
    text = replace_mask(text)
    print('id', text_processing.make_into_question(text, method='id', mask_type=text_processing.NOUNPHRASE_LABEL))
    print('nc', text_processing.make_into_question(text, method='nc', mask_type=text_processing.NOUNPHRASE_LABEL))
    print('rq', text_processing.make_into_question(text, method='rq', mask_type=text_processing.NOUNPHRASE_LABEL))
    print('wba', text_processing.make_into_question(text, method='wba', mask_type=text_processing.NOUNPHRASE_LABEL))
    print('awb', text_processing.make_into_question(text, method='awb', mask_type=text_processing.NOUNPHRASE_LABEL))


def test_atomic(a):
    sentences = a['context'].split('. ')
    h = sentences[0] + '. '
    h = h.replace('___', 'vvv')
    q = sentences[1] + ' [MASK]'
    print(f'H:{h}  Q:{q}')
    mq(q)
    return


print(cwwv[0])
text_processing.extract_and_print(recompose(cwwv[0]))

relations, templates, bothrt = get_relations_and_templates(cwwv)
#print('Relations:', relations)
#print('Templates:', templates)
print('(Relations, Templates):', bothrt)

mq(cwwv[0]['question']['stem'])
mq(cwwv[1]['question']['stem'])
mq(cwwv[2]['question']['stem'])
mq(cwwv[3]['question']['stem'])



print(atomic[0])
mq(atomic[0]['context'] + ' [MASK]')

"""
