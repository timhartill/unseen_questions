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
import language_modelling
import eval_metrics
from mdr_config import llm_args



# names of datasets to generate explanatory info for. Train datasets are assumed to have 'train.tsv' and 'dev.tsv':
TRAIN_SETS = ['creak_od_ans','csqa2', 
              'hpqa_od_ans', 'hover_od_ans', 'musique_qa', 'nq_open_od_ans', 
              'tatqa', 
              'qasc', 'arc_easy', 'arc_hard']

EVAL_SETS_DEV = ['commonsenseqa', 'drop', 'musique_mu_dev_odv2', 'strategy_qa_bigbench_od_ans']
EVAL_SETS_TEST = ['arc_da_od_ans', 'iirc_initial_context']

TEMPLATES = ['generic_kojima_22_0shot_stepbystep.txt',     #generic zero shot COT
             'generic_csqa2_liu_22_modified.txt',          # modified from liu 2022 'generate some knowledge about the input'
             'generic_csqa2_weicot_noanswer_modified.txt', # modified from liu 2022 to be cot without "So the answer is.."
             'generic_csqa2_weicot_modified.txt',          # modified from liu 2022 to be cot with "So the answer is.."
             'generic_csqa_weicot_from_li_22_noanswer_modified.txt', # modified from Li 2002  to be cot with ans options but without "So the answer is.."
             'generic_csqa_weicot_from_li_22_modified.txt', # modified from Li 2002  to be cot with ans options and with "So the answer is.."
             'generic_csqa_weicot_from_li_22_noanswer_noanschoices_modified.txt', # modified from Li 2002 to be cot without ans options & without "So the answer is.."
            ]


def make_llm_query(sample):
    """ Make sample components into LLM query
    # Question text only?" or 
    # "Context text.\nQuestion text?" or "Context text. Question text?" or
    # "Question text? Answer Choices:  (A) ground (B) bathroom (C) forest (D) countryside (E) rural area" or
    # "Context text. Question text?  Answer Choices:  (A) ground (B) bathroom (C) forest (D) countryside (E) rural area" or
    # "Question text is Answer Choices: ..."
    """
    query = sample['q_only'].strip()
    query = query.replace('.?', '?') # csqa has some questions like Are all apples red.?
    if query[-1] != '?':
        query += '?'
    if sample['mc_options'] != '':
        query += ' Answer Choices: ' + sample['mc_options']
    if sample['context'] != '':
        context = sample['context'].strip()
        if context[-1] != '.':
            context += '.'
        query = context + ' ' + query 
    return query



def load_files(ds_set, file_name):
    """ Load files for each dataset in a set
    """
    out_set = {}
    for ds in ds_set:
        path = os.path.join(eval_metrics.UQA_DIR, ds, file_name)
        print(f'Loading input file: {path}...')
        ds_in = utils.load_uqa_supervised(path, ans_lower=False, return_parsed=True)
        for sample in ds_in:
            sample['llm_query'] = make_llm_query(sample)
        out_set[ds] = {'path': path, 'split': file_name[:-4],
                       'data': ds_in}
    return out_set
    

def tokenize_input(tokenizer, text):
    """ Tokenise input
    if text = str, output is [1, #toks]
    if text = list of n strings, output is [#strings, #maxtoks] but since we arent setting padding=True this will error out
    Note: don't use text= list since need the attention masks to ignore padding - without these eg beam search will consider padding and return poor results...
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return input_ids.cuda()



def generate_simple(args, model, tokenizer, input_ids):
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
    if not args.do_sample:
        generated_ids = model.generate(input_ids, num_beams=args.num_beams, min_length=1, 
                                       max_new_tokens=args.max_new_tokens, early_stopping=True, 
                                       num_return_sequences=args.num_return_sequences)
    else:
        generated_ids = model.generate(input_ids, do_sample=True, 
                                       max_new_tokens=args.max_new_tokens, 
                                       top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                                       num_return_sequences=args.num_return_sequences)
    
    return tokenizer.batch_decode(generated_ids[:, start_decode:], skip_special_tokens=True)  # ['rationale 1.', 'rationale 2', ...]


def generate_all(args, logger, model, tokenizer, ds_set, templates):
    """ Generate rationales for all datasets
    """
    for ds in ds_set:
        curr_ds = ds_set[ds]
        logger.info(f'Generating rationales for dataset: {ds}  split:{curr_ds["split"]}')
        for i, sample in enumerate(curr_ds['data']):
            sample['rationales'] = {}
            for j, template in enumerate(templates):
                prompt = language_modelling.fill_prompt_template(template, query=sample['llm_query'])
                if args.debug:
                    logger.info('--------------------------------------')
                    logger.info(f"QUERY {i} TEMPLATE {j} ANSWER {sample['answer']}: {prompt}")
                input_ids = tokenize_input(tokenizer, prompt)
                rationales = generate_simple(args, model, tokenizer, input_ids)
                if args.debug:
                    logger.info(f"RATIONALE(S) Q:{i} T:{j}: {rationales}")
                    logger.info('--------------------------------------')
                sample['rationales'][j] = rationales 
            if i == args.max_samples-1:
                break
    return


if __name__ == '__main__':
    args = llm_args()
    if args.max_memory < 0:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB-4}GB'
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
    
    if args.debug:
        logger.info('Debug mode: outputting to log only.')
    logger.info('Max samples per dataset to output: {args.max_samples}')
    
    template_paths = [os.path.join(eval_metrics.UQA_DIR, 'prompts', t) for t in TEMPLATES]
    templates = language_modelling.load_templates(template_paths)
    logger.info('Template ids and paths:')
    for i, t in enumerate(template_paths):
        logger.info(f"Template {i} {t}")
    logger.info('Template ids and full formats:')
    for i, t in enumerate(templates):
        logger.info(f"{i}###{t}###")
        
    if args.generate_train:
        train_dict = load_files(TRAIN_SETS, 'train.tsv')
        generate_all(args, logger, model, tokenizer, train_dict, templates=templates)
    
    if args.generate_dev:
        dev_dict = load_files(TRAIN_SETS, 'dev.tsv')

    
    if args.generate_eval:
        eval_dev_dict = load_files(EVAL_SETS_DEV, 'dev.tsv')
        eval_test_dict = load_files(EVAL_SETS_TEST, 'test.tsv')
        eval_dict = {**eval_dev_dict, **eval_test_dict}
        generate_all(args, logger, model, tokenizer, eval_dict, templates=templates)

    
    logger.info('Finished!')



'''
    #SUMMARY: Can get BS 2 on 3 gpus with max 128 new toks
    # Beam search seems better than greedy and sample
    # BUT when have bs > 1 beam doesnt work any better than greedy!
    # Safest: Always beam, bs 1 and # ret seqs 1 
    #TODO try with real data
    #TODO add templates
    # param = template set per dataset. generate 1 output/sample/template and store in separate keys - output scores??
    # for eval - run x templates for each, use reranker to select best,  for train - template set has 1 (best) template...
    # try galactica before settling on a model

    text = """
    Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. 
    How many punches did he throw?\n
    A: Let’s think step by step.\n"""
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()

    logger.info(f"Input shape: {input_ids.shape}")  # [1, 41]
    
    generated_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
    logger.info(f"GREEDY generated ids shape: {generated_ids.shape}")   # [1, 169]
    logger.info(f"FROM {args.model_name} Generate: GREEDY: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")
    
    generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_new_tokens=args.max_new_tokens, early_stopping=True,)
    logger.info(f"BEAM=4 generated ids shape: {generated_ids.shape}")  # [1, 169]
    
    logger.info(f"FROM {args.model_name} Generate: BEAM=4: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")
    
    generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=1)
    logger.info(f"SAMPLE generated ids shape: {generated_ids.shape}")   # [1, 169]

    logger.info(f"FROM {args.model_name} Generate: SAMPLE: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")


    logger.info("BS 1 NUM_RETURN_SEQS=3 ###########")
    
    
    generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_new_tokens=args.max_new_tokens, early_stopping=True, num_return_sequences=3)
    logger.info(f"BEAM=4 generated ids shape: {generated_ids.shape}")  # [3, 169]
    
    logger.info(f"FROM {args.model_name} Generate: BEAM=4: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")
    
    generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=3)
    logger.info(f"SAMPLE generated ids shape: {generated_ids.shape}")  # [3, 169]

    logger.info(f"FROM {args.model_name} Generate: SAMPLE: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")

    logger.info("FINISHED NUM_RETURN_SEQS=3 ###########")


    logger.info("BATCH SIZE 2 tests NUM_RET_SEQS=1 *****")

    text2 = """
    Q: On average Joe throws 5 punches per minute. A fight lasts 6 rounds of 2 minutes. 
    How many punches did he throw?\n
    A: Let’s think step by step.\n"""

    input_ids = tokenizer([text, text2], return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.cuda()

    logger.info(f"Input shape: {input_ids.shape}") # [2, 41]
    
    generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_new_tokens=args.max_new_tokens, early_stopping=True, num_return_sequences=1)
    logger.info(f"BEAM=4 generated ids shape: {generated_ids.shape}")   # [2, 169]
    
    logger.info(f"FROM {args.model_name} Generate: BEAM=4: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")
    
    generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=1)
    logger.info(f"SAMPLE generated ids shape: {generated_ids.shape}")   # [2, 169]

    logger.info(f"FROM {args.model_name} Generate: SAMPLE: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")

    
    logger.info("BATCH SIZE 2 tests NUM_RET_SEQS=3 *****")
    logger.info(f"Input shape: {input_ids.shape}") # [2, 41]
    
    generated_ids = model.generate(input_ids, num_beams=4, min_length=1, max_new_tokens=args.max_new_tokens, early_stopping=True, num_return_sequences=3)
    logger.info(f"BEAM=4 generated ids shape: {generated_ids.shape}")  # [6, 169]
    
    logger.info(f"FROM {args.model_name} Generate: BEAM=4: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")
    
    generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=args.max_new_tokens, top_k=50, top_p=0.95, num_return_sequences=3)
    logger.info(f"SAMPLE generated ids shape: {generated_ids.shape}")  # [6, 169]

    logger.info(f"FROM {args.model_name} Generate: SAMPLE: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")


    
    logger.info("FINISHED!")
'''