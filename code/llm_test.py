#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:52:45 2022

@author: tim hartill

adapted from https://gist.github.com/younesbelkada/073f0b7902cbed2cbff662996a74162e

Test Large LM

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
#model_name = 'facebook/opt-66b'
model_name = 'bigscience/bloom'

print(f"MODEL: {model_name}")

text = """
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. 
How many punches did he throw?\n
A: Let’s think step by step.\n"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

print(f"MODEL: {model_name}. Loaded and tokenizer and tokenised inputs..")


free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{free_in_GB-2}GB'

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

print(f"MODEL: {model_name}. n_gpus:{n_gpus}  max_memory:{max_memory}")

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
  model_name, 
  device_map='auto', 
  load_in_8bit=True, 
  max_memory=max_memory
)

print(f"Loaded model {model_name}!")

generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(f"FROM {model_name} Generate: GREEDY: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_length=MAX_NEW_TOKENS, early_stopping=True,)
print(f"FROM {model_name} Generate: BEAM=4: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

generated_ids = model.generate(input_ids, do_sample=True, max_length=MAX_NEW_TOKENS, top_k=50, top_p=0.95, num_return_sequences=1)
print(f"FROM {model_name} Generate: SAMPLE: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

print("FINISHED!")
