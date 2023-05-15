#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:37:36 2023

@author: tim hartill

LLama tests

"""
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

lpath2 = '/data/thar011/data_bai2/ckpts/llama13b'

tokenizer = AutoTokenizer.from_pretrained(lpath2)  #works

def tokenize_input(tokenizer, text, max_seq_len=1024):  # got oom on seq len of ~1600
    """ Tokenise input
    if text = str, output is [1, #toks]
    if text = list of n strings, output is [#strings, #maxtoks] but since we arent setting padding=True this will error out
    Note: don't use text= list since need the attention masks to ignore padding - without these eg beam search will consider padding and return poor results...
    """
    input_ids = tokenizer(text, return_tensors="pt", max_length=max_seq_len, truncation=True).input_ids
    return input_ids.cuda()

def generate_simple(model, tokenizer, input_ids, do_sample=True, num_beams=1, 
                    max_new_tokens=128, num_return_sequences=1, temperature=0.7, top_k=0, top_p=0.92):
    """ Simple generation routine, takes in input ids [[0,432, 123, 2]] and generates outputs
    # NOTE: topk=50, temperature=0.7 errored out
    #    greedy: model.generate(input_ids, max_length=50) 
    #    beam: model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True,num_return_sequences=2,no_repeat_ngram_size=2)
    #    sample: model.generate(input_ids, do_sample=True, max_length=50, top_k=0, temperature=0.7) # the lower the temp the greater the chance of picking high prob words
    #    topk: model.generate(input_ids, do_sample=True, max_length=50, top_k=50) # only sample from the top 50 words by prob each time
    #    nucleus: model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0) # (also called topP) choose from the top words whose collective prob exceeds p so lower p = fewer but higher prob words to choose from
    #    combo: model.generate(input_ids,do_sample=True, max_length=50, top_k=50, top_p=0.95, num_return_sequences=3)
    """
    start_decode = input_ids.shape[1]
    if not do_sample:
        generated_ids = model.generate(input_ids, num_beams=num_beams, min_length=1, 
                                       max_new_tokens=max_new_tokens, early_stopping=True, 
                                       num_return_sequences=num_return_sequences,
                                       return_dict_in_generate=True)
    else:
        generated_ids = model.generate(input_ids, do_sample=True, 
                                       max_new_tokens=max_new_tokens, 
                                       top_k=top_k, top_p=top_p, temperature=temperature,
                                       num_return_sequences=num_return_sequences,
                                       return_dict_in_generate=True)
    
    return tokenizer.batch_decode(generated_ids['sequences'][:, start_decode:], skip_special_tokens=True)


tst = tokenize_input(tokenizer, "Did Aristotle use a laptop? ")  #works

#works. Up to approx 51GB GPU
model = AutoModelForCausalLM.from_pretrained(lpath2) #works
model.to(torch.device("cuda"))  #works, takes 50GB
out = generate_simple(model, tokenizer, input_ids=tst, do_sample=True, num_beams=1, max_new_tokens=128, num_return_sequences=1, temperature=0.7, top_k=0, top_p=0.92)

model = AutoModelForCausalLM.from_pretrained(lpath2, device_map='auto', max_memory={0:'65GB'})  #works 51GB
#works:
out = generate_simple(model, tokenizer, input_ids=tst, do_sample=True, num_beams=1, max_new_tokens=128, num_return_sequences=1, temperature=0.7, top_k=0, top_p=0.92)

model = AutoModelForCausalLM.from_pretrained(lpath2,device_map='auto', load_in_8bit=True, max_memory={0:'65GB'}) #works 14.1GB GPU !
#works: 14.5GB GPU!
out = generate_simple(model, tokenizer, input_ids=tst, do_sample=True, num_beams=1, max_new_tokens=128, num_return_sequences=1, temperature=0.7, top_k=0, top_p=0.92)


