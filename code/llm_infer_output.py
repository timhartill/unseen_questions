#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:52:45 2022

@author: tim hartill

Large LM Inference

Called in two modes - eval mode: Processes a set of datasets for eval/prompt search without outputting tsv dataset(s) and not resumable
                      single: if outputting a single dataset. resumable and outputs tsv dataset


"""
import os
import logging
from datetime import date
import string
import numpy as np
import json
import random


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
import language_modelling
import eval_metrics
from mdr_config import llm_args



# names of datasets to generate explanatory info for. Train datasets are assumed to have 'train.tsv' and 'dev.tsv':
TRAIN_SETS = [#'creak_od_ans','csqa2',  'hover_od_ans', ]
              'hpqa_od_ans', 'musique_qa_full', 'nq_open_od_ans', ]
              #'tatqa', 
              #'qasc', 'arc_easy', 'arc_hard']

EVAL_SETS_DEV = ['musique_mu_dev_odv2']  #['commonsenseqa', 'strategy_qa_bigbench_od_ans'] #['commonsenseqa', 'strategy_qa_bigbench_od_ans', 'musique_mu_dev_odv2', 'drop']  # 
EVAL_SETS_TEST = ['arc_da_od_ans', 'iirc_initial_context']

#TEMPLATES = ['generic_kojima_22_0shot_stepbystep.txt',     #generic zero shot COT
#             'generic_csqa2_liu_22_modified.txt',          # modified from liu 2022 'generate some knowledge about the input'
#             'generic_csqa2_weicot_noanswer_modified.txt', # modified from liu 2022 to be cot without "So the answer is.."
#             'generic_csqa2_weicot_modified.txt',          # modified from liu 2022 to be cot with "So the answer is.."
#             'generic_csqa_weicot_from_li_22_noanswer_modified.txt', # modified from Li 2002  to be cot with ans options but without "So the answer is.."
#             'generic_csqa_weicot_from_li_22_modified.txt', # modified from Li 2002  to be cot with ans options and with "So the answer is.."
#             'generic_csqa_weicot_from_li_22_noanswer_noanschoices_modified.txt', # modified from Li 2002 to be cot without ans options & without "So the answer is.."
#            ]


#TEMPLATES = ['generic_csqa2_weicot_modified.txt',          # modified from liu 2022 to be cot with "So the answer is.."
#             'generic_csqa2_weicot_modified_withinstruction.txt', # modified from liu 2022 to be instruction + cot with "So the answer is.."
#             'generic_hpqa_weicot_modified.txt',            # cot with "So the answer is" created from hpqa train by me
#             'generic_csqa_weicot_from_li_22_modified.txt', # modified from Li 2002  to be cot with ans options and with "So the answer is.." eg "So the answer is book (C)."
#             'generic_csqa_weicot_from_li_22_anschoices_choicetextonly.txt',  # modified from Li 2002 to be cot with ans options and with "So the answer is.." without the answer key eg "So the answer is book."
#             'generic_csqa2_csqa_weicot_modified.txt',  # cot combo of csqa2 and csqa, the latter with ans choices w/o keys
#             'generic_hpqa_csqa2_weicot_modified.txt',  # cot combo of hpqa + csqa2
#             'generic_hpqa_csqa2_csqa_weicot_modified.txt',  # cot combo of hpqa + csqa2 + csqa
#            ]

#TEMPLATES = ['generic_csmadeup_weicot_anschoices_choicetextonly.txt', # my made up mc template with answer as option text only. Also changing to options separated by \nl
#             'generic_csmadeup_weicot_anschoices_choicekeyonly.txt', # my made up mc template with answer as option key only. Also changing to options separated by \nl
#    ]


#TEMPLATES = ['generic_csmadeup_weicot_anschoices_answeronly.txt', # made up mc. answer only without rationales
#             'generic_csmadeup_weicot_anschoices_choicetextonlysqastyle.txt',  # made up mc. sqa-style deductive rationale rather than 'the answer must..' style
#    ]

#TEMPLATES = ['generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2v1.txt', # made up mc + 1 each hpqa csqa2
#             'generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2v2.txt',  # made up mc + 2 each hpqa csqa2
#    ]

#TEMPLATES = ['generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv3.txt',  # made up mc + 2 each hpqa csqa2 + instruction
#             'generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt',  # made up mc + 2 each hpqa csqa2 + 1 more made up + instruction
#             'generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionremovedv5.txt',  # made up mc + 2 each hpqa csqa2 + 1 more made up without instruction
#             'generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionreorderedv6.txt',  # v4 reordered made up mc + 2 each hpqa csqa2 + 1 more made up without instruction
#    ]

#TEMPLATES = ['generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt',  # made up mc + 2 each hpqa csqa2 + 1 more made up + instruction
#    ]

#TEMPLATES = ['generic_csqa2_ynmadeup_weicot.txt',  # modified from liu 2022 csqa2 plus some hpqa and other made up to be cot with "So the answer is.."
#             'generic_csqa2_ynmadeup_weicot_withinstruction.txt', # modified from liu 2022 csqa2 plus some hpqa and other made up to be cot with "So the answer is.." plus an instruction
#    ]

#TEMPLATES = ['generic_csqa2_ynmadeup_weicot_answeronly.txt',  # modified from liu 2022 csqa2 plus some hpqa and other made up to be answer only"
#             'generic_csqa2_ynmadeup_weicot_answeronly_withinstruction.txt', # modified from liu 2022 csqa2 plus some hpqa and other made up to be answer only plus an instruction
#             'generic_csqa2_weicot_modified_answeronly.txt',  # answer only version of generic_csqa2_weicot_modified.txt
#             'generic_csqa2_ynmadeup_weicot_withinstructionv2.txt',  # generic_csqa2_ynmadeup_weicot_withinstruction.txt with 2 extra csqa2 examples
#    ]

#TEMPLATES = ['generic_csqa2_ynmadeup_weicot_withinstructionv3.txt',  # generic_csqa2_ynmadeup_weicot_withinstructionv2.txt minus the factual hpqa examples
#             'generic_csqa2_ynmadeup_weicot_withinstructionv4.txt',  # generic_csqa2_ynmadeup_weicot_withinstructionv3.txt minus the 8x5 area sample
#    ]

#TEMPLATES = ['generic_spanmadeup_hpqa_csqa2_weicot.txt',  # generic span + y/n prompt from csqa2, hpqa plus some made up
#             'generic_spanmadeup_hpqa_csqa2_weicot_withinstruction.txt',  # generic span + y/n prompt from csqa2, hpqa plus some made up + instruction
#    ]

TEMPLATES = ['generic_spanmadeup_hpqa_csqa2_answeronly.txt',  # baseline answer only prompt for spanyn with same examples as generic_spanmadeup_hpqa_csqa2_weicot.txt
             'generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt',  # generic span + y/n prompt from csqa2, hpqa plus some made up + instruction + 2 musique train examples
             'generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_mu_iirclikev3.txt',  # generic span + y/n prompt from csqa2, hpqa plus some made up + instruction + 2 musique train examples + 2 iirc-like madeup examples
    ]




ANSWER_PREFIX = 'So the answer is'


def make_llm_query(sample):
    """ Make sample components into LLM query
    # Question text only?" or 
    # "Context text.\nQuestion text?" or "Context text. Question text?" or
    # "[Context text. ]Question text?\nAnswer Choices: (A) ground\n(B)bathroom\n(C) forest\n(D) countryside\n(E) rural area\n" or
    # "Question text is Answer Choices: ..."
    """
    query = sample['q_only'].strip()
    query = query.replace('.?', '?') # csqa has some questions like Are all apples red.?
    if query[-1] != '?':
        query += '?'
    if sample['mc_options'] != '':
        query += '\nAnswer Choices:\n' + sample['mc_options'].replace(' (','\n(')
    if sample['context'] != '':
        context = sample['context'].strip()
        if context[-1] != '.':
            context += '.'
        query = context + ' ' + query 
    return query



def load_files(args, logger, ds_set, file_name):
    """ Load files for each dataset in a set
    Output: {'dataset_1': {'path: 'file/path/dev.tsv', 'split': 'dev', data': [{'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}, ...]},
             'dataset_2': {'path: 'file/path/dev.tsv', 'split': 'dev', data': [{'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}, ...]},
             ...
            }
    """
    out_set = {}
    for ds in ds_set:
        path = os.path.join(eval_metrics.UQA_DIR, ds, file_name)
        logger.info(f'Loading input file: {path}...')
        ds_in = utils.load_uqa_supervised(path, ans_lower=False, return_parsed=True)
        logger.info(f"{ds} {file_name} Sample Count: {len(ds_in)}")
        if args.rand_order:
            logger.info('Randomised sample order.')
            random.shuffle(ds_in)
        if args.max_samples > 0:
            logger.info(f'Truncating to max {args.max_samples} samples.')
            ds_in = ds_in[:args.max_samples]
        for i, sample in enumerate(ds_in):
            sample['llm_query'] = make_llm_query(sample)
        out_set[ds] = {'path': path, 'split': file_name[:-4],
                       'data': ds_in}
    return out_set
    

def tokenize_input(tokenizer, text, max_seq_len=1024):  # got oom on seq len of ~1600
    """ Tokenise input
    if text = str, output is [1, #toks]
    if text = list of n strings, output is [#strings, #maxtoks] but since we arent setting padding=True this will error out
    Note: don't use text= list since need the attention masks to ignore padding - without these eg beam search will consider padding and return poor results...
    """
    input_ids = tokenizer(text, return_tensors="pt", max_length=max_seq_len, truncation=True).input_ids
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
                                       num_return_sequences=args.num_return_sequences,
                                       return_dict_in_generate=True)
    else:
        generated_ids = model.generate(input_ids, do_sample=True, 
                                       max_new_tokens=args.max_new_tokens, 
                                       top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                                       num_return_sequences=args.num_return_sequences,
                                       return_dict_in_generate=True)
    
    return tokenizer.batch_decode(generated_ids['sequences'][:, start_decode:], skip_special_tokens=True)  # ['rationale 1.', 'rationale 2', ...]


def split_rationale(rationales, sample):
    """ Input: rationales' = list of rationale strings, potentially containing \n and 'So the answer is ...'. 
                For greedy decode: 1 rationale in list
        output = [ {'raw': 'raw rationale', 
                    'nl_trunc': 'rationale truncated at 1st \n and answer potentially removed', 
                    'answer': 'predicted answer or ""'}, ...] 
            to reconstruct everything before nl: output['nl_trunc'] + ' So the answer is ' + output['answer'] + '.'
            
    sample input format: {'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}
    """
    outlist = []
    for r in rationales:
        out = {'nl_trunc': '', 'answer': '', 'raw': r}
        nl_idx = r.find('\n')
        if nl_idx == -1:
            nl_trunc = r.strip()
        else:
            nl_trunc = r[:nl_idx].strip()
        ans_idx = nl_trunc.lower().rfind(ANSWER_PREFIX.lower()) 
        if ans_idx == -1:
            out['nl_trunc'] = nl_trunc
        else:
            out['nl_trunc'] = nl_trunc[:ans_idx].strip()
            answer = nl_trunc[ans_idx+len(ANSWER_PREFIX):].strip()
            if answer != '' and answer[-1] == '.':
                answer = answer[:-1].strip()
            # can get answers like 'abc', '(C) abc', 'abc (C), 'C' or '(C)':
            if len(answer) <= 3 and (answer.find('(') != -1 or answer in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):  #  eg 'C' or '(C)'
                answer = utils.find_mc_answer(sample['mc_options'], answer.strip('(').strip(')'))
            else:
                opt_idx= answer.find('(')
                if opt_idx != -1:                   # eg 'abc (C)' or '(C) abc'
                    answer = answer[:opt_idx] + answer[opt_idx+3:]
            out['answer'] = answer.strip()
        if out['nl_trunc'] != '' and out['nl_trunc'][-1] not in ['.','?','!',':',';']:
            out['nl_trunc'] = out['nl_trunc'] + '.'
        outlist.append(out)
    return outlist
    


def generate_all(args, logger, model, tokenizer, ds_set, templates, num_already_processed=0):
    """ Generate rationales for all datasets. For greedy decode there is 1 rationale generated per prompt template
    sample output format:
    {'dataset_1': {'path': 'file/path/dev.tsv', 'split': 'dev', 'metric': 'SS', 'ans_score_llm': 0.99, 
                    'data': [
                        {'question': 'full q input txt',
                         'answer': 'ans txt', #or ['ans text 1', 'ans text 2']
                         'q_only': 'q only',
                         'mc_options': '(A) opta (B) optb (C) optc',
                         'context': 'context',
                         'rationales': {0: [{'nl_trunc': 'rationale truncated at 1st nl and answer potentially removed',
                                               'answer': 'answer [with option txt only if mc]',
                                               'raw': 'as generated minus query',
                                               'metric': 'F1 or SS or EM',
                                               'ans_score': 0.99}],
                                        1: [{'nl_trunc': 'rationale truncated at 1st nl and answer potentially removed',
                                               'answer': 'answer [with option txt only if mc]',
                                               'raw': 'as generated minus query',
                                               'metric': 'F1 or SS or EM',
                                               'ans_score': 0.99}],
                                       }
                         } ],
     'dataset_2': {...}, ... }
     }
    """
    for k, ds in enumerate(ds_set):
        curr_ds = ds_set[ds]
        logger.info(f'Generating rationales for dataset: {ds}  split:{curr_ds["split"]}')
        metric, metric_fn = eval_metrics.get_dataset_metric(ds)
        for i, sample in enumerate(curr_ds['data']):
            if  args.resume_dir is not None and args.resume_dir.strip() != '' and i < num_already_processed:  # only resumable in single mode
                continue
            sample['rationales'] = {}
            if args.debug and i <= 2:
                logger.info('--------------------------------------')
                logger.info(f"DS: {ds} Q#:{i} Q:{sample['llm_query']}  ANSWER {sample['answer']}")
            for j, template in enumerate(templates):  # only ever 1 template in single mode
                jkey = str(j)
                prompt = language_modelling.fill_prompt_template(template, query=sample['llm_query'])
                input_ids = tokenize_input(tokenizer, prompt, args.max_seq_len_in)
                rationales = generate_simple(args, model, tokenizer, input_ids)   #[rationale] stripped of query but including text after nl and answer if any
                rationales_processed = split_rationale(rationales, sample)
                for r in rationales_processed:
                    r['metric'] = metric
                    if r['answer'] == '':  # if couldnt extract answer, use nl_trunc as answer as this sometimes happens when it just generates an answer instead of a rationale
                        ans = r['nl_trunc']
                    else:
                        ans = r['answer']
                    if metric == 'SS':
                        score = metric_fn(ans, sample['answer'], sample['mc_options'])
                    else:
                        score = metric_fn(ans, sample['answer'])
                    r['ans_score'] = float(score)
                sample['rationales'][jkey] = rationales_processed
                if args.debug and i <= 2:
                    logger.info(f"RATIONALE Q:{i} T:{jkey}: R:{sample['rationales'][jkey][0]['nl_trunc']} A:{sample['rationales'][jkey][0]['answer']} GOLD:{sample['answer']} {sample['rationales'][jkey][0]['metric']}:{sample['rationales'][jkey][0]['ans_score']}")
                    logger.info('--------------------------------------')
            if i % 5 == 0 and i != 0:
                logger.info(f"Processed: {i+1} samples..")
            if args.template_file.strip() != '' and i != 0 and i % 500 == 0:
                logger.info(f"Processed {i+1+num_already_processed} of {len(curr_ds['data'])} samples. Saving partially completed file..")
                json.dump(ds_set, open(os.path.join(args.output_dir, 'llm_samples_with_context.json'), 'w'))
                logger.info(f"Saved partially completed json file to {os.path.join(args.output_dir, 'llm_samples_with_context.json')}")
            if i == args.max_samples-1:
                logger.info(f"Stopped at {i+1} samples..")
                #break
        logger.info(f"DS SUMMARY {metric}: {ds}")
        for j, t_file in enumerate(TEMPLATES):
            jkey = str(j)
            scores = []
            for i, sample in enumerate(curr_ds['data']):
                for r in sample['rationales'][jkey]:
                    scores.append(r['ans_score'])
                if i == args.max_samples-1:
                    break
            mean = np.mean(scores)
            if np.isnan(mean):
                mean = -1.0
            mean = float(mean)
            logger.info(f"T:{jkey} {metric}:{mean} ({t_file})")
            curr_ds['metric'] = metric
            curr_ds['ans_score_llm'] = mean
    return


if __name__ == '__main__':
    random.seed(42)
    args = llm_args()
    if args.max_memory < 0:
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        max_memory = f'{free_in_GB - args.max_memory_buffer}GB'
    else:
        max_memory = f'{args.max_memory}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    pred_dict = None
    num_already_processed = 0
    if args.resume_dir is not None and args.resume_dir.strip() != '':
        predict_file = args.predict_file.strip()
        ds, split_file = os.path.split(predict_file)
        ds = os.path.split(ds)[-1]
        args.output_dir = args.resume_dir
        resume_file = os.path.join(args.output_dir, 'llm_samples_with_context.json')
        if os.path.exists(resume_file):
            pred_dict = json.load(open(resume_file))
            num_already_processed = -1
            for i, sample in enumerate(pred_dict[ds]['data']):
                if sample.get('rationales') is None:  # no key = unprocessed sample
                    num_already_processed = i
                    break
            if num_already_processed == -1:
                num_already_processed = len(pred_dict[ds]['data'])  # all samples already processed
        else:
            print(f"Processed samples file not found: {resume_file}. Did you intend to set --resume_dir?")
            assert os.path.exists(resume_file)
            
    else:  # not resuming
        date_curr = date.today().strftime("%m-%d-%Y")
        mstrip = ''.join(c if c not in string.punctuation else '-' for c in args.model_name )
        run_name = f"{args.prefix}-{date_curr}-LLM-{mstrip}-maxsmpls{args.max_samples}-rand{args.rand_order}"
        args.output_dir = os.path.join(args.output_dir, run_name)
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(f"Output log eval_log.txt will be written to: {args.output_dir}")
    if args.template_file.strip() != '':
        logger.info(f"SINGLE mode enabled. Using prompt template:{args.template_file}")
        TEMPLATES = [args.template_file.strip()]
        predict_file = args.predict_file.strip()
        if pred_dict is not None:
            logger.info(f"Resuming job. Number already processed: {num_already_processed} of {len(pred_dict[ds]['data'])}.")

    else:
        logger.info(f"EVAL mode. Train: {args.generate_train} dev: {args.generate_dev} ds:{TRAIN_SETS}")
        logger.info(f" eval: {args.generate_eval} ds dev:{EVAL_SETS_DEV}  ds test:{EVAL_SETS_TEST}")
        predict_file = ''

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
        logger.info('Debug mode: outputting 1st 3 rationales to log.')
    logger.info(f'Max samples per dataset to output: {args.max_samples}')
    
    
    template_paths = [os.path.join(eval_metrics.UQA_DIR, 'prompts', t) for t in TEMPLATES]  # load prompt files
    templates = language_modelling.load_templates(template_paths)
    logger.info('Template ids and paths:')
    for i, t in enumerate(template_paths):
        logger.info(f"Template {i} {t}")
    logger.info('Template ids and full formats:')
    for i, t in enumerate(templates):
        logger.info(f"T:{i}###{t}###")
    
    if args.template_file.strip() != '':  # run rationale generation for entire input file using single template 
        ds, split_file = os.path.split(predict_file)
        ds = os.path.split(ds)[-1]
        if pred_dict is None:
            pred_dict = load_files(args, logger, [ds], split_file)
        generate_all(args, logger, model, tokenizer, pred_dict, templates=templates, num_already_processed=num_already_processed)
        json.dump(pred_dict, open(os.path.join(args.output_dir, 'llm_samples_with_context.json'), 'w'))
        logger.info(f"Saved fully processed file as {os.path.join(args.output_dir, 'llm_samples_with_context.json')}")
        outlist = []
        outlist_with_ans = []
        for sample in pred_dict[ds]['data']:
            q = sample['q_only'].strip()
            a = sample['answer'][0] if len(sample['answer']) <= 1 else sample['answer']
            rationale = sample['rationales']['0'][0]['nl_trunc']
            rat_ans = sample['rationales']['0'][0]['answer']
            c = sample['context'].strip()
            if c != '':  # if there is an initial context make new context that + 'Further Explanation: ' + llm rationale so can distinguish it later if needed
                if c [-1] not in ['.', '?', '!', ':', ';']:
                    c += '.'
                rationale = c + ' Further Explanation: ' + rationale
            rationale_with_ans = rationale
            if rat_ans != '':
                rationale_with_ans += ' So the answer is ' + rat_ans
            outlist.append( utils.create_uqa_example(q, 
                                                     utils.create_uqa_context(sample['mc_options'], rationale),
                                                     a, append_q_char='?') 
                          )
            outlist_with_ans.append( utils.create_uqa_example(q, 
                                                     utils.create_uqa_context(sample['mc_options'], rationale_with_ans),
                                                     a, append_q_char='?') 
                          )
        out_dir, out_file = os.path.split(args.output_dataset)    
        utils.save_uqa(outlist, out_dir, out_file)
        logger.info(f"Saved into {args.output_dataset}")
        utils.save_uqa(outlist_with_ans, out_dir + '_with_llm_ans', out_file)
        logger.info(f"Saved {out_dir}_with_llm_ans version also..")
            
    else:  # eval mode
        if args.generate_train:
            train_dict = load_files(args, logger, TRAIN_SETS, 'train.tsv')
            generate_all(args, logger, model, tokenizer, train_dict, templates=templates)
            json.dump(train_dict, open(os.path.join(args.output_dir, 'llm_samples_with_context_train.json'), 'w'))
        
        if args.generate_dev:
            dev_dict = load_files(args, logger, TRAIN_SETS, 'dev.tsv')
            generate_all(args, logger, model, tokenizer, dev_dict, templates=templates)
            json.dump(dev_dict, open(os.path.join(args.output_dir, 'llm_samples_with_context_dev.json'), 'w'))
        
        if args.generate_eval:
            eval_dev_dict = load_files(args, logger, EVAL_SETS_DEV, 'dev.tsv')
            eval_test_dict = load_files(args, logger, EVAL_SETS_TEST, 'test.tsv')
            eval_dict = {**eval_dev_dict, **eval_test_dict}
            generate_all(args, logger, model, tokenizer, eval_dict, templates=templates)
            json.dump(eval_dict, open(os.path.join(args.output_dir, 'llm_samples_with_context_eval.json'), 'w'))
    
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