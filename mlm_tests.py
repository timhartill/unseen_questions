#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:59:58 2021

@author: tim hartill

Masking Objective Tests


"""
import numpy as np
import copy
import string

from transformers import AutoTokenizer, AutoModelForPreTraining

m = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(m)

txt1 = "the rain is 123.25% heavier in Spain than NZ."
txt2 = txt1 + " the antimatter component has largess in banana land."




# this part occurs in preprocessing ie requires access to either the original string or the tokenised input 

def get_word_starts(toks, specialchar = 'Ġ'):
    """ Get the beginning of each word in a list of tokenised text
        Return list of word beginning indices into toks
    """
    return np.array([0] + [i for (i,t) in enumerate(toks) if t[0]==specialchar or t[0] in string.punctuation])    


# this part occurs on-the-fly during training in the dataset object

def get_spans(tok_idxs, toks_to_mask=0.15, avg_span_len=3, sd=0.75):
    """ Calculate number and length of spans for given input seq length
    """
    num_toks = len(tok_idxs)
    num_spans = int( (num_toks * toks_to_mask) / avg_span_len) + 1
    span_lengths = np.random.normal(avg_span_len, scale=sd, size=num_spans).round().astype('int')
    span_lengths = np.clip(span_lengths, 1, avg_span_len+4)    
    return span_lengths


def merge_intervals(in_list):
    in_list.sort(key=lambda interval: interval[0])
    merged = [in_list[0]]
    for current in in_list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def mask_words(tok_idxs, span_lengths, word_starts, mask_token):
    """ Given a list of token indices, an array of spans and a list of word start indices
        return a masked version of toks plus the list of masked spans 
    """
    num_toks = len(tok_idxs)
    num_words = len(word_starts)
    replace_spans = []
    for length in span_lengths:
        print(f'Processing length: {length}')
        span_start_idx = np.random.choice(num_words)
        span_start = word_starts[span_start_idx]
        print(f'Start: {span_start}  tok: {tok_idxs[span_start]}  length:{length}')
        if span_start + length > num_toks:
            length = num_toks - span_start
            print(f'Length past eos, truncating length to {length}')
        else:
            for next_wordstart in word_starts[span_start_idx+1:]:
                print(f"Finding word boundary. Checking {next_wordstart} {tok_idxs[next_wordstart]}")
                if next_wordstart >= span_start+length:
                    length = next_wordstart - span_start
                    print(f"Found! New length: {length}  last token in span: {tok_idxs[span_start+length]}")
                    break
        span_end = span_start + length
        replace_spans.append( [span_start, span_end]  )
    replace_spans = merge_intervals(replace_spans)  # aggregate overlaps
    print(replace_spans)
    replaced_toks = []
    tmp_tok_idxs = tok_idxs.copy()
    for replace_span in replace_spans:
        replaced_toks.append( tok_idxs[replace_span[0]:replace_span[1]] )
        first = True
        for i in range(replace_span[0], replace_span[1]):
            if first:
                tmp_tok_idxs[i] = mask_token
                first = False
            else:
                tmp_tok_idxs[i] = -999999
    print(replaced_toks)
    new_tok_idxs = []
    for tok in tmp_tok_idxs:
        if tok != -999999:
            new_tok_idxs.append(tok)
    print(new_tok_idxs)
    return new_tok_idxs
    


toktxt = tokenizer.tokenize(txt2)
word_starts = get_word_starts(toktxt)

tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)

span_lengths = get_spans(tok_idxs)
new_tok_idxs = mask_words(tok_idxs, span_lengths, word_starts, mask_token=tokenizer.mask_token_id)






