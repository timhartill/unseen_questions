#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:41:45 2022

The Iterator creates a new dataset with the question reformatted to include implicit relations. 
This script takes that dataset and creates a new one with the original question plus the Iterator-created context

if the output_dataset name already exists, any existing tsv files matching tsv names in iter_dataset (& orig_dataset) will be overwritten
otherwise any new ones will be added

@author: tim hartill


Usage: edit and run bash addorigq_bart_s12_v1_add_impl_rels_creak_ods_ans.sh

"""

import os
import argparse
from tqdm import tqdm

import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig_dataset", default="datasetname", help="Original dataset name. format: datasetname. UQA_DIR will be prepended.")
    parser.add_argument("--iter_dataset", default="", type=str, help="Name of dataset output from Iterator. UQA_DIR will be prepended and each file dev|train|test.tsv from orig_dataset will be appended.")
    parser.add_argument("--output_dataset", default="", type=str, help="Output dataset name. UQA_DIR will be prepended and each file dev|train|test.tsv from iter_dataset will be appended.")

    args = parser.parse_args()
    """ Test options
    args.orig_dataset = 'creak_od_ans'
    args.iter_dataset = 'creak_fullwiki_bs150_implrel'
    args.output_dataset = 'creak_fullwiki_bs150_implrel_origq'
    """
    
    origdir = os.path.join(UQA_DIR, args.orig_dataset)
    origfiles = utils.list_files_pattern(origdir, '*.tsv')
    print(f"Original Question Files from {origdir} : {origfiles}")
    
    iterdir = os.path.join(UQA_DIR, args.iter_dataset)
    iterfiles = utils.list_files_pattern(iterdir, '*.tsv')
    print(f"New context Files from {iterdir} : {iterfiles}")    
    
    outdir = os.path.join(UQA_DIR, args.output_dataset)
    print(f"Output to: {outdir}")
    os.makedirs(outdir, exist_ok=True)

   
    for file in iterfiles:
        infile = os.path.join(iterdir, file)
        iterdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        infile = os.path.join(origdir, file)
        origdset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
        
        assert len(iterdset) == len(origdset)
        
        out_list = []
        for itersample, origsample in zip(iterdset, origdset):
            out_list.append( utils.create_uqa_example(origsample['q_only'], 
                                                      utils.create_uqa_context(origsample['mc_options'], itersample['context']), 
                                                      origsample['answer']) )
        
        utils.save_uqa(out_list, outdir, file)


if __name__=='__main__':
    main()
 

