#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:29:18 2023

@author: tim hartill

Create new tsv formatted samples by concatenating Initial Context (if any) + LLM-retrieved rationale + Iterator-retrieved context 

"""

import os
import argparse


import utils
import eval_metrics

VALID_FILES = ['dev.tsv', 'test.tsv', 'train.tsv']
UQA_DIR = eval_metrics.UQA_DIR


def main():
    """ If there was an initial given context the LLM version will have a context of form: "initial context. Further Explanation: llm rationale"
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_dataset", default="", type=str, help="Name of dataset output from LLM prompting. UQA_DIR will be prepended and each file dev|train|test.tsv from orig_dataset will be appended.")
    parser.add_argument("--iter_dataset", default="", type=str, help="Name of dataset output from Iterator. UQA_DIR will be prepended and each file dev|train|test.tsv from orig_dataset will be appended.")
    parser.add_argument("--output_dataset", default="", type=str, help="Output dataset name. UQA_DIR will be prepended and each file dev|train|test.tsv from iter_dataset will be appended.")

    args = parser.parse_args()
    """ Test options
    args.orig_dataset = 'creak_od_ans'
    args.iter_dataset = 'creak_fullwiki_bs150_implrel'
    args.output_dataset = 'creak_fullwiki_bs150_implrel_origq'
    """
    
    origdir = os.path.join(UQA_DIR, args.llm_dataset)  # treat as original ie tsv files in this dataset are considered matchable
    origfiles = utils.list_files_pattern(origdir, '*.tsv')
    origfiles = [f for f in origfiles if f in VALID_FILES]
    print(f"LLM Question Files from {origdir} : {origfiles}")
    
    iterdir = os.path.join(UQA_DIR, args.iter_dataset)
    iterfiles = utils.list_files_pattern(iterdir, '*.tsv')
    print(f"New context Files from {iterdir} : {iterfiles}")    
    
    outdir = os.path.join(UQA_DIR, args.output_dataset)
    print(f"Output to: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    
    for file in origfiles:
        infile = os.path.join(origdir, file)
        origdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        infile = os.path.join(iterdir, file)
        iterdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        
        assert len(iterdset) == len(origdset)
        
        out_list = []
        for origsample, itersample in zip(origdset, iterdset ):
            orig_context = origsample['context']  # 'llm rationale' or 'initial context. Further Explanation: llm rationale'
            idx = orig_context.find(' Further Explanation: ')
            if idx != -1:
                initial_context = orig_context[:idx].strip()
                rationale = orig_context[idx:].strip()
            else:
                initial_context = ''
                rationale = 'Further Explanation: ' + orig_context.strip()
            iter_context = itersample['context']
            iter_context = iter_context[len(initial_context):].strip()  # remove initial context if any from retrieved sample
            if initial_context != '' and initial_context[-1] not in ['.', '!', '?', ':', ';']:
                initial_context += '.'
            if rationale[-1] not in ['.', '!', '?', ':', ';']:
                rationale += '.'
            if iter_context[-1] not in ['.', '!', '?', ':', ';']:
                iter_context += '.'
            
            new_context = (initial_context + ' ' + rationale + ' ' + iter_context).strip()
            
            out_list.append( utils.create_uqa_example(origsample['q_only'], 
                                                      utils.create_uqa_context(origsample['mc_options'], new_context), 
                                                      origsample['answer']) )
        
        utils.save_uqa(out_list, outdir, file)

    
    
    
if __name__ == '__main__':
    main()