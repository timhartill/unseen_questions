#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:34:44 2022

Create wikipedia corpus for MLM pretraining

First edit/run convert_beerqa_wikipedia_paras.py to create the corpus file that is used by this program and several others.

@author: tim hartill


Note: takes ~30 mins to load docs..


"""

import os
import random
from html import unescape

from transformers import AutoTokenizer

import eval_metrics
import utils


BEER_WIKI_SAVE_WITHMERGES = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl'
OUT_DIR = '/data/thar011/data/unifiedqa/enwiki_20200801_selfsvised'
NUM_DEV = 1000

docs = utils.load_jsonl(BEER_WIKI_SAVE_WITHMERGES)

addspecialtoksdict = eval_metrics.special_tokens_dict  # test tokenization length with ind. digit tokenization...
tokenizer = utils.load_model(model_name='facebook/bart-large', loadwhat='tokenizer_only', special_tokens_dict=addspecialtoksdict)        

max_toks = 512
added_bits = len(tokenizer.tokenize('<s>. \\n</s>'))
max_toks = max_toks - added_bits
print(f"Max num tokens for text after allowing for BOS, EOS etc: {max_toks}")

def build_word_tok_ratio(docs, tokenizer, max_toks, n=10000):
    """ Build a token to word ratio empirically and calculate approx how many 
        words on average can fit in a context. 
        Note: Just informational. Doesn't guarantee that any sample won't exceed the max token count, just that an "average" sample won't.. 
    """
    stats = {'avg_wordcount': 0, 'avg_tokcount': 0}
    for i in range(n):
        docidx = random.randint(0, len(docs))
        doc = docs[docidx]
        text = ' '.join([t['text'] for t in doc['paras']])
        stats['avg_wordcount'] += len(text.split())
        stats['avg_tokcount'] += len(tokenizer.tokenize(text))
        if i % (n//4) == 0:
            print(f"Processed: {i}")
        
    stats['avg_wordcount'] /= n
    stats['avg_tokcount'] /= n
    stats['avg_ratio_macro_tokstowords'] = stats['avg_tokcount'] / stats['avg_wordcount']
    stats['max_wordcount'] = int(max_toks / stats['avg_ratio_macro_tokstowords'])
    print(f"n:{n} stats:{stats}") # n:10000 stats:{'avg_wordcount': 374.7298, 'avg_tokcount': 547.8532, 'avg_ratio_macro_tokstowords': 1.4619952830012453, 'max_wordcount': 346}
    return stats

#stats = build_word_tok_ratio(docs, tokenizer, max_toks, n=10000)


def make_samples(docs, tokenizer, max_toks):
    """ Create standard formatted samples with paras per doc packed in to roughly 512 toks.
        Note: short docs will be less than 512 toks. We dont pack more in to these to preserve diversity.
    """
    out_list = []
    for i, doc in enumerate(docs):
        text = ''
        tok_count = 0
        for para in doc['paras']:
            para_toks = tokenizer.tokenize(para['text'])
            if tok_count + len(para_toks) > max_toks:
                excess = max_toks - (tok_count+len(para_toks)+1) 
                para_toks = para_toks[:excess]
                para_truncated = tokenizer.decode(tokenizer.convert_tokens_to_ids(para_toks)) + '...'
                text += ' ' + para_truncated
                break
            else:
                tok_count += len(para_toks) + 1
                text += ' ' + para['text']
        out_list.append( utils.create_uqa_example(text.strip(), ' ', append_q_char='') )
        if i % 10000 == 0:
            print(f"Loaded {i} samples of {len(docs)}...")
    return out_list

out_list = make_samples(docs, tokenizer, max_toks)
print(f"Finished loading! Number of samples={len(out_list)}")
print('Shuffling...')
random.shuffle(out_list)
print(f"Separating {NUM_DEV} samples for dev..")
dev = out_list[:NUM_DEV]
train = out_list[NUM_DEV:]
print(f"Counts: Train:{len(train)} Dev:{len(dev)}")
print(f"Creating {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)
outfile = os.path.join(OUT_DIR, 'train.tsv')
print(f"Outputting train: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(train))
outfile = os.path.join(OUT_DIR, 'dev.tsv')
print(f"Outputting dev: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(dev))
print('Finished!')





