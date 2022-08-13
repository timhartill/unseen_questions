#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:29:23 2022

Build <No Answer> dataset from combined HPQA/Hover/FEVER examples.

Uses stategy built for  encode_context_stage2(..) in reader/reader_dataset.py

Input samples dataset created using combine_sentence_annot_datasets.py which in turn aggregates samples created
in convert_hotpot_sentence_annotations.py, convert_hover.py and convert_fever_sentence_annotations.py respectively.

@author: tim hartill

"""
import os
import random
from tqdm import tqdm

import eval_metrics
import utils
from text_processing import get_sentence_list, split_into_sentences, create_sentence_spans


SENT_DIR = '/home/thar011/data/sentences/'
SENT_TRAIN = os.path.join(SENT_DIR, 'sent_train.jsonl')
SENT_DEV = os.path.join(SENT_DIR, 'sent_dev.jsonl')

# Below for creating UQA-formatted hard examples - code to do so at end
UQA_DIR = eval_metrics.UQA_DIR

addspecialtoksdict = eval_metrics.special_tokens_dict  # test tokenization length with ind. digit tokenization...
tokenizer = utils.load_model(model_name='facebook/bart-large', loadwhat='tokenizer_only', special_tokens_dict=addspecialtoksdict)        

max_toks = 512
added_bits = len(tokenizer.tokenize('<s>. \\n</s>'))
max_toks = max_toks - added_bits
print(f"Max num tokens for text after allowing for BOS, EOS etc: {max_toks}")


sent_dev = utils.load_jsonl(SENT_DEV)     #26587
sent_train = utils.load_jsonl(SENT_TRAIN) #239276


def standardize(split):
    for sample in tqdm(split):
        utils.consistent_bridge_format(sample)  # all samples now dict_keys(['question', 'answers', 'src', 'type', '_id', 'bridge', 'num_hops', 'pos_paras', 'neg_paras', 'sp_gold', 'para_idxs'])
        sample['para_idxs'] = utils.get_para_idxs(sample["pos_paras"])
        if sample['answers'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
            sample['answers'][0] = 'yes'
        elif sample['answers'][0] in ["REFUTES", "NOT_SUPPORTED"]:
            sample['answers'][0] = 'no'
        elif sample['answers'][0] == 'NOT ENOUGH INFO':  #Unused
            sample['answers'][0] = '<No Answer>'
    return

standardize(sent_dev)
standardize(sent_train)


random.seed(42)
dev_out = utils.make_unanswerable_uqa_from_mdr_format(sent_dev, tokenizer, max_toks, include_title_prob=0.65, include_all_sent_prob=0.5)
out_dir = os.path.join(UQA_DIR, "noanswer_hpqa_fever_hover_hard")
print(f'Outputting to {out_dir}')
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, 'dev.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(dev_out))
    
train_out = utils.make_unanswerable_uqa_from_mdr_format(sent_train, tokenizer, max_toks, include_title_prob=0.65, include_all_sent_prob=0.5)
outfile = os.path.join(out_dir, 'train.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(train_out))
print('Finished outputting noanswer_hpqa_fever_hover_hard!')



