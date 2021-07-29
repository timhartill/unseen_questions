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

import spacy
nlp = spacy.load("en_core_web_sm")

m = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(m)

qasc_corpus = []
with open("/data/thar011/data/unifiedqa/qasc_facts_selfsvised/dev.txt", "r") as f:
                for line in f:
                    qasc_corpus.append(tokenizer.bos_token + ' ' + line)
                    
wiki_corpus = []
with open("/data/thar011/data/unifiedqa/enwiki-20200511_selfsvised/dev.txt", "r") as f:
                for line in f:
                    wiki_corpus.append(tokenizer.bos_token + ' ' + line)



txt1 = tokenizer.bos_token + " " + "the rain is 123.25% heavier in Spain than NZ."
txt2 = txt1 + " the antimatter component has largess in banana land."




# this part occurs in preprocessing ie requires access to either the original string or the tokenised input 

def get_word_starts(toks, specialchar = 'Ġ', bos_token='<s>'):
    """ Get the beginning of each word in a list of tokenised text
        Return list of word beginning indices into toks
    """
    word_starts = [i for (i,t) in enumerate(toks) if t[0]==specialchar or t[0] in string.punctuation]
    if toks[0] == bos_token: # don't want to mask a bos token 
        word_starts.pop(0)   
    if word_starts[0] != 1:  # first non bos token is always a word start
        word_starts = [1] + word_starts
    return word_starts

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


def mask_words(tok_idxs, span_lengths, word_starts, mask_token, verbose = False):
    """ Given a list of token indices, an array of spans and a list of word start indices
        return a masked version of toks plus the list of masked spans 
    """
    num_toks = len(tok_idxs)
    num_words = len(word_starts)
    replace_spans = []
    for length in span_lengths:
        if verbose: print(f'Processing length: {length}')
        span_start_idx = np.random.choice(num_words)
        span_start = word_starts[span_start_idx]
        if verbose: print(f'Start: {span_start}  tok: {tok_idxs[span_start]}  length:{length}')
        if span_start + length > num_toks:
            length = num_toks - span_start
            if verbose: print(f'Length past eos, truncating length to {length}')
        else:
            for next_wordstart in word_starts[span_start_idx+1:]:
                if verbose: print(f"Finding word boundary. Checking {next_wordstart} {tok_idxs[next_wordstart]}")
                if next_wordstart >= span_start+length:
                    length = next_wordstart - span_start
                    if verbose: print(f"Found! New length: {length}  last token in span: {tok_idxs[span_start+length]}")
                    break
        span_end = span_start + length
        replace_spans.append( [span_start, span_end]  )
    replace_spans = merge_intervals(replace_spans)  # aggregate overlaps
    if verbose: print('replace_spans', replace_spans)
    replaced_toks = []
    tmp_tok_idxs = tok_idxs.copy()
    masked_tok_count = 0
    for replace_span in replace_spans:
        replaced_toks.append( tok_idxs[replace_span[0]:replace_span[1]] )
        first = True
        for i in range(replace_span[0], replace_span[1]):
            masked_tok_count += 1
            if first:
                tmp_tok_idxs[i] = mask_token
                first = False
            else:
                tmp_tok_idxs[i] = -999999
    if verbose: print(f'masked tok_idxs: {tmp_tok_idxs}')
    if verbose: print('replace_toks', replaced_toks)
    new_tok_idxs = []
    for tok in tmp_tok_idxs:
        if tok != -999999:
            new_tok_idxs.append(tok)
    if verbose: print(f"new_tok_idxs: {new_tok_idxs}")
    ratio = masked_tok_count/len(tok_idxs)
    if verbose: print(f"masked token ratio: {ratio}")
    return new_tok_idxs, replaced_toks, ratio
    


#TODO might have to do .lower after tokenizing in order to allow spacy better opportunity to find named entities...

txt3 = tokenizer.bos_token + "the rain is 123.25% heavier in Spain than NZ."
txt4 = "the rain is 123.25% heavier in Spain than NZ."


toktxt = tokenizer.tokenize(txt1)

toktxt = tokenizer.tokenize(txt2)

toktxt = tokenizer.tokenize(txt3)
toktxt = tokenizer.tokenize(txt4)


word_starts = get_word_starts(toktxt)  # in reality add 1 to word starts for the extra bos token thats added later in manual_batch_encode

print(f"toktxt: {toktxt}")
print(f"word_starts: {word_starts}")
tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
print(f"tok_idxs: {tok_idxs}")

span_lengths = get_spans(tok_idxs, toks_to_mask=0.10, avg_span_len=3)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
print(f"span_lengths: {span_lengths}")
new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, span_lengths, word_starts, mask_token=tokenizer.mask_token_id)
print(f"New toks: {tokenizer.decode(new_tok_idxs)}")
print(f"Masked: {replaced_toks}")

ratios = []
num_spans = []
new_idxs = []
replaced = []
for line in qasc_corpus:
    toktxt = tokenizer.tokenize(line)
    word_starts = get_word_starts(toktxt)
    tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
    span_lengths = get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
    new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, span_lengths, word_starts, mask_token=tokenizer.mask_token_id)
    ratios.append(ratio)
    num_spans.append(len(replaced_toks))
    new_idxs.append(new_tok_idxs)
    replaced.append(replaced_toks)
ratios = np.array(ratios)
num_spans = np.array(num_spans)
print(f"Number: {ratios.shape[0]}  Mean masked toks ratio: {ratios.mean():.2f}  Max masked toks:{ratios.max():.2f}  Min masked toks:{ratios.min():.2f}")
print(f"Mean spans: {num_spans.mean():.2f}  Max spans:{num_spans.max():.2f}  Min spans:{num_spans.min():.2f}")


ratios = []
num_spans = []
new_idxs = []
replaced = []
for line in wiki_corpus:
    toktxt = tokenizer.tokenize(line)
    word_starts = get_word_starts(toktxt)
    tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
    span_lengths = get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
    new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, span_lengths, word_starts, mask_token=tokenizer.mask_token_id)
    ratios.append(ratio)
    num_spans.append(len(replaced_toks))
    new_idxs.append(new_tok_idxs)
    replaced.append(replaced_toks)
ratios = np.array(ratios)
num_spans = np.array(num_spans)
print(f"Number: {ratios.shape[0]}  Mean masked toks ratio: {ratios.mean():.2f}  Max masked toks:{ratios.max():.2f}  Min masked toks:{ratios.min():.2f}")
print(f"Mean spans: {num_spans.mean():.2f}  Max spans:{num_spans.max():.2f}  Min spans:{num_spans.min():.2f}")

print(wiki_corpus[0])
print(tokenizer.decode(new_idxs[0]))
print(replaced[0])

print(wiki_corpus[10])
print(tokenizer.decode(new_idxs[10]))
print(replaced[10])


# SSM tests
txt=tokenizer.bos_token + " " + "john smith is a nice person. john smith from apple is looking at buying u.k. startup for $1 billion in July 2020 or perhaps 1/6/23 or failing that 2024\nHello world.\nApples are good fruit to eat\nAre new zealand fruit or australian vegetables better for you? Astronomers look for the bright stars that orbit dark partners in the same way. The North Star can be used to find your way if you're lost in the dark. The north star can be used to find your way if you're lost in the dark"


def ner(instr, verbose=False):
    """ Perform named entity recognition on text and return a list of named entities, numbers, dates etc
    """
    ner_list = []
    doc = nlp(instr.strip('\\n \n'))    
    for ent in doc.ents:
        if verbose: print(ent.text, '"' + ent.text_with_ws + '"', ent.start_char, ent.end_char, ent.label_)
        ner_list.append( {'txt_with_ws': ent.text_with_ws, 'start':ent.start_char, 'end': ent.end_char, 'type': ent.label_} )
    return ner_list


def find_tok_idx(toks, start, end):
    """ Convert start/end indices in original text to token indices in tokenised text list
    """
    tok_start = -1
    tok_end = -1
    curr_str_idx_start = 0
    for i, tok in enumerate(toks):
        curr_str_idx_end = curr_str_idx_start + len(tok.replace('Â','').replace('Ä','').replace('Å',''))
        if start >= curr_str_idx_start and start <= curr_str_idx_end:
            tok_start = i
        if end >= curr_str_idx_start and end <= curr_str_idx_end:
            tok_end = i+1
            break
        curr_str_idx_start = curr_str_idx_end
    return tok_start, tok_end


def find_sub_list(sublst1, sublst2, lst):
    """ Return start/end indices of all occurences of sublist in list
        Note: Can't tell whether the tokens match with/out a preceding space so must try both ways
    """
    results=[]
    sll=len(sublst1)
    for ind in (i for i,e in enumerate(lst) if e==sublst1[0]):
        if lst[ind:ind+sll]==sublst1:
            results.append((ind,ind+sll))
    sll=len(sublst2)
    for ind in (i for i,e in enumerate(lst) if e==sublst2[0]):
        if lst[ind:ind+sll]==sublst2:
            results.append((ind,ind+sll))  
    results = list(set(results))
    return results  


def map_ners(toks, ner, tokenizer, verbose = False):
    """ Map list of NERs previously identified on raw text to token ids
    """
    unique_ner = list(set([ w['txt_with_ws'].strip(string.punctuation+' ').strip() for w in ner ]))
    tok_map = []
    final_ner = []
    for n in unique_ner:
        ner_txt_tok = tokenizer.tokenize(n)
        ner_txt_tok2 = tokenizer.tokenize(' ' + n)
        found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, toks)
        if verbose: 
            print(f"Orig: {n}") 
            for tok_start, tok_end in found_list:
                print(f"tokens: {wiki0[tok_start:tok_end]}")
            if len(found_list) == 0:
                print(f"{i}NOT FOUND: {n} 1:{ner_txt_tok} 2:{ner_txt_tok2}")
        if len(found_list) > 0:
            tok_map.append(found_list)
            final_ner.append(n)
    return final_ner, tok_map


ners = []
for line in qasc_corpus:
    ners.append( ner( line.strip('\\n \n') ) )  #spacy thinks land.\\n and \\n in general is a person..
num_ners = [len(n) for n in ners]
num_ners_np = np.array(num_ners)
print(f" Num: {len(num_ners)}  Mean:{num_ners_np.mean():.2f}  Max:{num_ners_np.max():.2f}  Min:{num_ners_np.min():.2f}")


wiki_ners = []
for line in wiki_corpus:
    wiki_ners.append( ner( line.strip('\\n \n') ) )  #spacy thinks land.\\n and \\n in general is a person..
num_ners = [len(n) for n in wiki_ners]
num_ners_np = np.array(num_ners)
print(f" Num: {len(num_ners)}  Mean:{num_ners_np.mean():.2f}  Max:{num_ners_np.max():.2f}  Min:{num_ners_np.min():.2f}")


def test_ner(c, ner, i, verbose = False):
    wiki0 = tokenizer.tokenize(c[i].lower())
    #if verbose: print(c[i].lower())
    #if verbose: print(wiki0)
    unique_ner = set([ w['txt_with_ws'].lower().strip(string.punctuation+' ').strip() for w in ner[i] ])
    for n in unique_ner:
        ner_txt_tok = tokenizer.tokenize(n)
        ner_txt_tok2 = tokenizer.tokenize(' ' + n)
        #if verbose: print(f"{ner_txt} tokenised: 1:{ner_txt_tok}  2: {ner_txt_tok2}")
        found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, wiki0)
        if verbose: 
            print(f"Orig: {n}")  #  Start:{n['start']}  End:{n['end']}")
            #print(f"Tok idx found: {found_list}")
            for tok_start, tok_end in found_list:
                print(f"tokens: {wiki0[tok_start:tok_end]}")
        if len(found_list) == 0:
            print(f"{i}NOT FOUND: {n} 1:{ner_txt_tok} 2:{ner_txt_tok2}")

test_ner(wiki_corpus, wiki_ners, 2, verbose=True)

for i in range(len(wiki_corpus)):
    test_ner(wiki_corpus, wiki_ners, i)

for i in range(len(qasc_corpus)):
    test_ner(qasc_corpus, ners, i)


wiki0 = tokenizer.tokenize("Bibekananda Agarwala " + wiki_corpus[1]+" Bibekananda Agarwala.")
print(wiki_corpus[1])
print(wiki0)
wiki0rejoined = ''.join(wiki0)
print(wiki0rejoined)
print(f"len wiki0: {len(wiki_corpus[1])} len wiki0rejoined: {len(wiki0rejoined)}")
for n in ners[1]:
    ner_txt = n['txt_with_ws'].strip()
    ner_txt_tok = tokenizer.tokenize(ner_txt)
    ner_txt_tok2 = tokenizer.tokenize(' ' + ner_txt)
    print(f"{ner_txt} tokenised: 1:{ner_txt_tok}  2: {ner_txt_tok2}")
    found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, wiki0)
    print(f"Orig: {n['txt_with_ws']}  Start:{n['start']}  End:{n['end']}")
    print(f"Tok idx found: {found_list}")
    #print(f"wiki_corpus: #{wiki_corpus[1][n['start']:n['end']]}#")
    #print(f"wiki0rejoined: #{wiki0rejoined.replace('Â','').replace('Ä','')[n['start']:n['end']]}#")  ## Not exactly identical § tokenizes to Â§ so strip Â
    #tok_start, tok_end = find_tok_idx(wiki0, n['start'], n['end'])
    #print(f"tok start: {tok_start}  tok end: {tok_end}")
    for tok_start, tok_end in found_list:
        print(f"tokens: {wiki0[tok_start:tok_end]}")






