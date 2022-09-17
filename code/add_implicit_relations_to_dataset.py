#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:41:45 2022

Generate implicit relations using already trained model, update samples and output to new directory

@author: tim hartill


Usage: edit and run bash addimpl_bart_s12_v1_add_impl_rels_creak_ods_ans.sh

"""

import os
import argparse
from tqdm import tqdm

import utils
import eval_metrics

UQA_DIR = eval_metrics.UQA_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict_dataset", default="datasetname", help="format: datasetname. UQA_DIR will be prepended.")
    parser.add_argument("--output_dataset", default="", type=str, help="output dataset name. UQA_DIR will be prepended and each file dev|train|test.tsv from predict_file will be appended.")
    parser.add_argument("--model", type=str, default="facebook/bart-large", help="The model to predict from.")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint to load.")
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=130)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true')
    parser.add_argument("--do_lowercase", action='store_true')
    parser.add_argument("--predict_batch_size", default=500, type=int, help="Batch size for evaluation.")

    args = parser.parse_args()
    """ Test options
    args.append_another_bos=True
    args.do_lowercase=True
    args.checkpoint = '/large_data/thar011/out/mdr/logs/UQA_s10_v2_implicitrelations_dsfix_ws50/best-model.pt'
    args.predict_dataset = 'creak_od_ans'
    args.output_dataset = 'creak_od_ans_impl_rels'
    
    """
    
    indir = os.path.join(UQA_DIR, args.predict_dataset)
    files = utils.list_files_pattern(indir, '*.tsv')
    print(f"Input from {indir} : {files}")
    
    outdir = os.path.join(UQA_DIR, args.output_dataset)
    print(f"Output to: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    tokenizer, model = utils.load_model(model_name=args.model, checkpoint=args.checkpoint)
    
    for file in files:
        infile = os.path.join(indir, file)    
        dset = utils.load_uqa_supervised(infile, ans_lower=False, verbose=True, return_parsed=True) # [{'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
    
        for b_start in tqdm( range(0, len(dset), args.predict_batch_size) ):
            b_end = b_start + args.predict_batch_size
            batch = []
            for s in dset[b_start:b_end]:
                batch.append(s['q_only'])
            
            res = utils.run_model(batch, model, tokenizer, indiv_digits=False, norm_numbers=False, skip_special_tokens=True,
                                  num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, 
                                  max_length=args.max_output_length, lower=args.do_lowercase, 
                                  max_input_length=args.max_input_length, output_scores=False, return_dict_in_generate=False)
            
            for i, s in enumerate(dset[b_start:b_end]):
                rstr = res[i].strip()  # eg 'oxygen therapy: countries used; countries: prohibitions;'
                s['impl_rels'] = rstr
                if rstr[-1] == ';':
                    rstr = rstr[:-1]
                rlist = rstr.split(';')
                out_list = []
                for r in rlist:
                    ent_rel_list = r.strip().split(':')
                    if len(ent_rel_list) != 2:
                        if r.strip() != '':
                            out_list.append(r.strip())
                    else:
                        out_list.append('for ' + ent_rel_list[0].strip() + ' - ' + ent_rel_list[1].strip())  # Considering for aristotle - year of death and for laptop - year invented.
                if len(out_list) > 0:
                    out_str = 'Considering ' + ' and '.join(out_list) + '.'
                else:
                    out_str = ''
                s['rel_sent'] = out_str
                s['q_out'] = s['rel_sent'] + ' ' + s['q_only'].strip()
                
        out_list = [utils.create_uqa_example(s['q_out'], utils.create_uqa_context(s['mc_options'], s['context']), s['answer']) for s in dset ]
        utils.save_uqa(out_list, outdir, file)


if __name__=='__main__':
    main()
 

