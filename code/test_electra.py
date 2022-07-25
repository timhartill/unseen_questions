#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:23:33 2022

@author: thar011
"""

from transformers import (AdamW, AutoConfig, AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)


#ADDITIONAL_SPECIAL_TOKENS = {'YES': '[unused0]', # 1
#                             'NO': '[unused1]',  # 2
#                             'SOP': '[unused2]', # 3
#                             'NONE': '[unused3]'}
ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']                           
model = 'google/electra-large-discriminator'

config = AutoConfig.from_pretrained(model)

#tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
sents = ['Sent one.', ' Sent 2.', ' Sent 3.']
q = 'The question?'
item = {'question': q, 'passages':[{'title':'title 1', 'sents':sents}, {'title':'title 2', 'sents':sents}]}

context = 'yes no [SEP] title 1 [unused1] Sent one. [unused1] Sent 2. [unused1] Sent 3. [SEP] title 2 [unused1] Sent one. [unused1] Sent 2. [unused1] Sent 3.'

q_toks = tokenizer.tokenize(item["question"])[:70] # ['the', 'question', '?']  - '?' would have been stripped alreay

c_toks = tokenizer.tokenize(context)

encodings = tokenizer.encode_plus(q_toks, text_pair=c_toks, max_length=200, return_tensors="pt", 
                                  is_split_into_words=True)
#{'input_ids': tensor([[ 101, 1996, 3160, 1029,  102, 2748, 2053,  102, 2516, 1015,    2, 2741,
#         2028, 1012,    2, 2741, 1016, 1012,    2, 2741, 1017, 1012,  102, 2516,
#         1016,    2, 2741, 2028, 1012,    2, 2741, 1016, 1012,    2, 2741, 1017, 1012,  102]]), 
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

tokenizer.vocab('[unused4]')  #5  : the 'unused' tokens are in the electra vocab already but need to add with additional_special_tokens to tokenise to them...they are not in the roberta vocab..

vocab_toks = list(tokenizer.vocab.keys()) # 30522

unused_toks = [v for v in vocab_toks if v.startswith('[unused')]  #994


special_tokens_dict = {'additional_special_tokens':['0','1','2', '3', '4', '5', '6', '7', '8', '9']}

# if try to add '[unused0]' into special_tokens_dict along with '0', then the '0' supercedes '[unused0]' and tokenizes as '[', 'unused', '0', ']'
# so keep '[unused0]' in tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
# and add the digits separately:
num_new = tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.tokenize('[unused0]1234567890') # ['[unused0]', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# note: if num_new > 0 (not the case here but eg if add '<totallynewtok>') then need to do: model.resize_token_embeddings(len(tokenizer))

