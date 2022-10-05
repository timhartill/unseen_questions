#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 18:13:49 2022

@author: tim hartill

Add 'Yes or no - ' to beginning of each sample in dataset

Used for StrategyQA datasets built with retrieved contexts

"""

import os
import argparse
from tqdm import tqdm

import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepend_phrase", default="Yes or no - ", type=str, help="Original dataset name. format: datasetname. UQA_DIR will be prepended.")        
    parser.add_argument("--orig_dataset", default="datasetname", type=str, help="Original dataset name. format: datasetname. UQA_DIR will be prepended.")
    parser.add_argument("--output_dataset", default="", type=str, help="Output dataset name. UQA_DIR will be prepended and each file dev|train|test.tsv from iter_dataset will be appended.")

    args = parser.parse_args()

    """ Test options
    args.orig_dataset = 'strategy_qa_bigbench_fullwiki_bs150_implrel'
    args.output_dataset = 'strategy_qa_bigbench_fullwiki_bs150_implrel_yn'
    """
    
    origdir = os.path.join(UQA_DIR, args.orig_dataset)
    origfiles = utils.list_files_pattern(origdir, '*.tsv')
    origfiles = [f for f in origfiles if f in ['dev.tsv', 'train.tsv', 'test.tsv']]  # skip any tsv files not in standard set
    print(f"Original Question Files from {origdir} : {origfiles}")
    
    outdir = os.path.join(UQA_DIR, args.output_dataset)
    print(f"Output to: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    for file in origfiles:
        infile = os.path.join(origdir, file)
        origdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True)

        out_list = []
        for origsample in tqdm(origdset):
            out_list.append( utils.create_uqa_example(args.prepend_phrase + origsample['q_only'], 
                                                      utils.create_uqa_context(origsample['mc_options'], origsample['context']), 
                                                      origsample['answer']) )
        
        utils.save_uqa(out_list, outdir, file)
    return




if __name__=='__main__':
    main()




