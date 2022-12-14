#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:52:45 2022

@author: tim hartill

Large LM Inference

"""
import os
import logging
from datetime import date
import string


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from mdr_config import llm_args






if __name__ == '__main__':
    args = llm_args()
    if args.max_memory < 0:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-6}GB'
    else:
        max_memory = f'{args.max_memory}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    samples = None
    num_already_processed = 0
    if args.resume_dir is not None and args.resume_dir.strip() != '':
        args.output_dir = args.resume_dir
        resume_file = os.path.join(args.output_dir, 'samples_with_context_llm.jsonl')
        if os.path.exists(resume_file):
            samples = utils.load_jsonl(resume_file)
            num_already_processed = -1
            for i, sample in enumerate(samples):
                if sample['llm_retrieved'] == []:  # no retrieved paras in dense_retrieved key = unprocessed sample
                    num_already_processed = i
                    break
            if num_already_processed == -1:
                num_already_processed = len(samples)  # all samples already processed, loading so can output using alternative context-building params
        else:
            print(f"Processed samples file not found: {resume_file}. Did you intend to set --resume_dir?")
            assert os.path.exists(resume_file)
            
    else:
        date_curr = date.today().strftime("%m-%d-%Y")
        mstrip = ''.join(c if c not in string.punctuation else '-' for c in args.model_name )
        run_name = f"{args.prefix}-{date_curr}-LLM-{mstrip}-"
        args.output_dir = os.path.join(args.output_dir, run_name)
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)    
    logger.info(f"Output log eval_log.txt will be written to: {args.output_dir}")

    logger.info(f"MODEL: {args.model_name}. n_gpus:{n_gpus}  max_memory:{max_memory}")
    
    #MAX_NEW_TOKENS = 128
    #model_name = 'facebook/opt-66b'
    #model_name = 'bigscience/bloom'
    
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info(f"MODEL: {args.model_name}. Loaded tokenizer. Now loading model..")
      
    model = AutoModelForCausalLM.from_pretrained(
      args.model_name, 
      device_map='auto', 
      load_in_8bit=True, 
      max_memory=max_memory
    )
    
    logger.info(f"Loaded model {args.model_name}!")

    text = """
    Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. 
    How many punches did he throw?\n
    A: Let’s think step by step.\n"""
    input_ids = tokenizer(text, return_tensors="pt").input_ids    
    input_ids = input_ids.cuda()

    
    generated_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    logger.info(f"FROM {args.model_name} Generate: GREEDY: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    
    generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_new_tokens=args.max_new_tokens, early_stopping=True,)
    logger.info(f"FROM {args.model_name} Generate: BEAM=4: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    
    generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=1)
    logger.info(f"FROM {args.model_name} Generate: SAMPLE: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    
    logger.info("FINISHED!")
