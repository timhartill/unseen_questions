#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:17:48 2023

@author: tim hartill

Extract LLM-generated rational from tsv for of form q \\n gold/init/retrieved papra(s) Further Explanation: llm generated rat.


"""

import os
import argparse


import utils
import eval_metrics

VALID_FILES = ['dev.tsv', 'test.tsv', 'train.tsv']
UQA_DIR = eval_metrics.UQA_DIR
LLM_RAT_KEY = ' Further Explanation: '



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_dataset", default="", type=str, help="Name of dataset output from LLM prompting. UQA_DIR will be prepended and each file dev|train|test.tsv from orig_dataset will be appended.")
    parser.add_argument("--output_dataset", default="", type=str, help="Output dataset name. UQA_DIR will be prepended and each file dev|train|test.tsv from iter_dataset will be appended.")

    args = parser.parse_args()
    """ Test options
    args.llm_dataset = 'drop_llm_expl'
    args.output_dataset = 'drop_llm_expl_only'
    """

    origdir = os.path.join(UQA_DIR, args.llm_dataset)  # treat llm as original ie tsv files in this dataset are considered matchable
    origfiles = utils.list_files_pattern(origdir, '*.tsv')
    origfiles = [f for f in origfiles if f in VALID_FILES]
    print(f"LLM Question Files from {origdir} : {origfiles}")

    outdir = os.path.join(UQA_DIR, args.output_dataset)
    print(f"Output to: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    out_list = []
    for file in origfiles:
        print(f"Processing: {file} ...")
        infile = os.path.join(origdir, file)
        origdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        print(f"Sample count: LLM: {len(origdset)}")
        for i, origsample in enumerate(origdset):
            orig_context = origsample['context']  # 'llm rationale' or 'initial/gold context. Further Explanation: llm rationale'
            idx = orig_context.find(LLM_RAT_KEY)
            if idx != -1:
                initial_context = orig_context[:idx].strip()
                rationale = orig_context[idx+len(LLM_RAT_KEY):].strip()
            else:
                initial_context = ''
                rationale = ''  #orig_context.strip()
                print(f"WARNING: row {i} '{LLM_RAT_KEY}' not found. Outputting '' as rationale: {origsample['question']}")
            if rationale != '' and rationale[-1] not in ['.', '!', '?', ':', ';']:
                rationale += '.'
            
            out_list.append( utils.create_uqa_example(origsample['q_only'], 
                                                      utils.create_uqa_context(origsample['mc_options'], rationale), 
                                                      origsample['answer']) )
        
        utils.save_uqa(out_list, outdir, file)
    print("Finished!")
    
    return



if __name__ == '__main__':
    main()


