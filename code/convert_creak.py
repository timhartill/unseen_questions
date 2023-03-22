#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:18:40 2022

Convert CREAK (https://github.com/yasumasaonoe/creak) to std formats:
    
    q->y/n
    q+seed entity paras->y/n
    q+expl->a

+ rationale reranker 'rr' training format:

    Output format:
    [ {'question': 'full question text incl MC options and preceding initial ctxt if any',
       'answers': ['answer1', ...],
       '_id': 'id string',
       'src': 'fever',
       pos_paras: [{'text': 'sentence 1. sentence 2. ..', "sentence_spans": [[0, 104], [104, 225], [225, 325]]}, ...],
       neg_paras: [], #filled in later
      },
     
    ]


@author: tim hartill


"""
import os
import json
import copy
import random
from tqdm import tqdm
from html import unescape

import eval_metrics
import utils


UQA_DIR = eval_metrics.UQA_DIR

addspecialtoksdict = eval_metrics.special_tokens_dict  # test tokenization length with ind. digit tokenization...
tokenizer = utils.load_model(model_name='facebook/bart-large', loadwhat='tokenizer_only', special_tokens_dict=addspecialtoksdict)        

max_toks = 512
added_bits = len(tokenizer.tokenize('<s>. \\n</s>'))
max_toks = max_toks - added_bits
print(f"Max num tokens for text after allowing for BOS, EOS etc: {max_toks}")


BQA_CORPUS = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl'

file_dev = '/home/thar011/data/creak/dev.json'
file_train = '/home/thar011/data/creak/train.json'
file_contrast_set = '/home/thar011/data/creak/contrast_set.json'

rr_dev = '/home/thar011/data/creak/creak_dev_rr_all_pos_neg.jsonl'
rr_train = '/home/thar011/data/creak/creak_train_rr_all_pos_neg.jsonl'

file_rr_dev_negs = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T24_YN_CREAK_DEV_onv6_sample-02-28-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json', 
               '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T25_YN_CREAK_DEV_onv6mod2_sample-03-01-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
               ]
file_rr_train_negs = ['/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T26_YN_CREAK_TRAIN_onv6_sample-03-01-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 '/large_data/thar011/out/mdr/logs/LLM_NEGRAT_T27_YN_CREAK_TRAIN_onv6mod2_sample-03-07-2023-LLM-bigscience-bloom-maxsmpls-1-randFalse/llm_samples_with_context.json',
                 ]


mdr_dev = '/home/thar011/data/creak/creak_dev_with_negs.json'
mdr_train = '/home/thar011/data/creak/creak_train_with_negs.json'
mdr_cs = '/home/thar011/data/creak/creak_cs_with_negs.json'


dev = utils.load_jsonl(file_dev)  #1371  # dict_keys(['ex_id', 'sentence', 'explanation', 'label', 'entity', 'en_wiki_pageid', 'entity_mention_loc'])
train = utils.load_jsonl(file_train)  #10176
cs = utils.load_jsonl(file_contrast_set)  #500  # No explanation or en_wiki_pageid but there is 'entity'

################################
# Rationales-orientfile_rr_dev_negsated code here (para-orientated below)
#################################

def make_expl_ans_format(split, out_dir, out_file):
    """ Output tsv formatted dataset q \n explanation \t yes/no
    AND return 'rationale reranker format' list
    """
    outlist = []
    for s in split:
        q = s['sentence'].strip()
        if q[-1] in ['.', '?', '!', ':', ';']:
            q = q[:-1]
        q = q + '?'
        if s['label'] == 'true':
            a = 'yes'
        else:
            a = 'no'
        c = s['explanation'].strip()
        if c[-1] not in ['.', '?', '!', ':', ';']:
            c = c + '.'
        outlist.append(utils.create_uqa_example(q, c, a))
    utils.save_uqa(outlist, out_dir, out_file)
    return        

make_expl_ans_format(dev, os.path.join(UQA_DIR, 'creak_expl_ans'), 'dev.tsv')
make_expl_ans_format(train, os.path.join(UQA_DIR, 'creak_expl_ans'), 'train.tsv')

dev_rr_format = [utils.create_rr_format(s['sentence'], s['explanation'], 'yes' if s['label'] == 'true' else 'no',
                                        sentence_spans=None, _id=s['ex_id'], src='creak', append_q_char='?') for s in dev]
#utils.saveas_jsonl(dev_rr_format, rr_dev)
train_rr_format = [utils.create_rr_format(s['sentence'], s['explanation'], 'yes' if s['label'] == 'true' else 'no',
                                        sentence_spans=None, _id=s['ex_id'], src='creak', append_q_char='?') for s in train]
utils.saveas_jsonl(train_rr_format, rr_train)

# merge routine to align pos and negs
dev_rr_format = utils.load_merge_negs(dev_rr_format, file_rr_dev_negs)
utils.saveas_jsonl(dev_rr_format, rr_dev)

train_rr_format = utils.load_merge_negs(train_rr_format, file_rr_train_negs)
utils.saveas_jsonl(train_rr_format, rr_train)


#TODO - merge into 1 pos + neg jsonl
#TODO - output in "1 pos + many negs format"

################################
# Rationales-orientated code above (para-orientated below)
#################################


# Add hyperlinked negative paras where title match in corpus found  WARNING: TAKES ~30 mins to load
docs = utils.load_jsonl(BQA_CORPUS)
titledict, dupdict = utils.build_title_idx(docs) # better to rebuild titledict as docs idxs changed after removal of docs with no paras.. 
id_dict = utils.build_idx_title(titledict)

def make_mdr_fmt(split, docs, titledict, id_dict, tokenizer, max_pos_toks = 200):
    """ Add mdr-formatted keys so can then make uqa-formatted examples using standard fn from mdr samples
    {'question': 'question', 'answers': [answers], 'src': 'creak', 'type': 'multi', '_id': sample['ex_id'],
     'pos_paras': pos_paras, 'neg_paras': neg_paras, 'bridge': bridge}
    """
    for s in tqdm(split):
        q = s['sentence'].strip()
        if q[-1] in ['.', '?', '!', ':', ';']:
            q = q[:-1]
        q = q + '?'
        s['question'] = q
        if s['label'] == 'true':
            s['answers'] = ['yes']
        else:
            s['answers'] = ['no']
        s['src'] = 'creak'
        s['type'] = 'multi'
        s['_id'] = s['ex_id']
        if s['en_wiki_pageid'] != 'n/a' and id_dict.get(s['en_wiki_pageid']) is not None:
            idx = id_dict[ s['en_wiki_pageid'] ]['idx']  #  {'title': 'Chinnar Wildlife Sanctuary', 'idx': 0}
        else:
            hlink, status, idx = utils.map_title_case(s['entity'], titledict)  # contrast set doesntt have wiki id 
        title = unescape(docs[idx]['title']).strip()
        s['bridge'] = [[title]]  # we will pack up to 3 paras from docs into single pos_paras entry
        para_merged = ''
        tok_count = 0
        for i, para in enumerate(docs[idx]['paras']):
            if i > 4:
                break
            this_para_tok_count = len(tokenizer.tokenize(para['text']))
            if para_merged == '':
                tok_count += this_para_tok_count
                para_merged = para['text'].strip()
            elif tok_count + this_para_tok_count < max_pos_toks+10:
                tok_count += this_para_tok_count
                para_merged += ' ' + para['text'].strip()
            else:
                break
        s['pos_paras'] = [{'title': title, 'text': para_merged}]
    return


make_mdr_fmt(dev, docs, titledict, id_dict, tokenizer, max_pos_toks = 200)
make_mdr_fmt(train, docs, titledict, id_dict, tokenizer, max_pos_toks = 200)
make_mdr_fmt(cs, docs, titledict, id_dict, tokenizer, max_pos_toks = 200)

utils.add_neg_paras(docs, titledict, dev, neg_key='neg_paras', top_up_with_rand=True)   # Status counts: total:1371 {'ok': 1368, 'nf': 0, 'sf': 3}
utils.add_neg_paras(docs, titledict, train, neg_key='neg_paras', top_up_with_rand=True) # Status counts: total:10176 {'ok': 10120, 'nf': 0, 'sf': 56}
utils.add_neg_paras(docs, titledict, cs, neg_key='neg_paras', top_up_with_rand=True)    # Status counts: total:500 {'ok': 397, 'nf': 2, 'sf': 101}
        
utils.saveas_jsonl(dev, mdr_dev)            # note: saves the existing, now extraneous keys also
utils.saveas_jsonl(train, mdr_train)            
utils.saveas_jsonl(cs, mdr_cs)      

dev = utils.load_jsonl(mdr_dev)      
train = utils.load_jsonl(mdr_train)  
cs = utils.load_jsonl(mdr_cs)      
    


# hard = q+gold+distractors->a
random.seed(42)
dev_out = utils.make_uqa_from_mdr_format(dev, tokenizer, max_toks, include_title_prob=0.9, include_all_sent_prob=1.1)
out_dir = os.path.join(UQA_DIR, "creak_hard")
print(f'Outputting to {out_dir}')
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, 'dev.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(dev_out))
    
train_out = utils.make_uqa_from_mdr_format(train, tokenizer, max_toks, include_title_prob=0.9, include_all_sent_prob=1.1)
outfile = os.path.join(out_dir, 'train.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(train_out))
print('Finished outputting creak_hard!')

cs_out = utils.make_uqa_from_mdr_format(cs, tokenizer, max_toks, include_title_prob=0.9, include_all_sent_prob=1.1)
out_dir = os.path.join(UQA_DIR, "creak_contrast_set_hard")
print(f'Outputting to {out_dir}')
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, 'dev.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(cs_out))
print('Finished outputting creak_contrast_set_hard!')


# open domain q->a
out_dir = os.path.join(UQA_DIR, "creak_od_ans")
dev_out = [utils.create_uqa_example(s['question'], '', s['answers'][0]) for s in dev]
utils.save_uqa(dev_out, out_dir, 'dev.tsv')
train_out = [utils.create_uqa_example(s['question'], '', s['answers'][0]) for s in train]
utils.save_uqa(train_out, out_dir, 'train.tsv')

out_dir = os.path.join(UQA_DIR, "creak_contrast_set_od_ans")
cs_out = [utils.create_uqa_example(s['question'], '', s['answers'][0]) for s in cs]
utils.save_uqa(cs_out, out_dir, 'dev.tsv')

# q + initial gold para -> a
out_dir = os.path.join(UQA_DIR, "creak_initial_context")
dev_out = [utils.create_uqa_example(s['question'], s['pos_paras'][0]['title'].strip()+': '+s['pos_paras'][0]['text'].strip(), s['answers'][0]) for s in dev]
utils.save_uqa(dev_out, out_dir, 'dev.tsv')
train_out = [utils.create_uqa_example(s['question'], s['pos_paras'][0]['title'].strip()+': '+s['pos_paras'][0]['text'].strip(), s['answers'][0]) for s in train]
utils.save_uqa(train_out, out_dir, 'train.tsv')

out_dir = os.path.join(UQA_DIR, "creak_contrast_set_initial_context")
cs_out = [utils.create_uqa_example(s['question'], s['pos_paras'][0]['title'].strip()+': '+s['pos_paras'][0]['text'].strip(), s['answers'][0]) for s in cs]
utils.save_uqa(cs_out, out_dir, 'dev.tsv')




