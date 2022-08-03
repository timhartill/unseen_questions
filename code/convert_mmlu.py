#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:51:52 2021

@author: tim hartill

MMLU datasets from https://openreview.net/pdf?id=d7KBjmI3GmQ

Convert MMLU datasets into UnifiedQA format

Update indir, outdir and unifiedqadir before running...

"""

import os
import pandas as pd
import re 
import shutil

indir = '/data/thar011/data/mmlu/data/test/'
dsets = os.listdir(indir)
outdir = '/data/thar011/data/mmlu/data/unifiedqa_format_test/'
os.makedirs(outdir, exist_ok=True)
unifiedqadir = '/data/thar011/data/unifiedqa/'



def replace_multiple_underscore(instr): #not used
    return re.sub(r'(_)\1+', r'\1', instr)  # replace multiple _ with single _

def replace_chars(instr): 
    outstr = instr.replace("’", "'")
    return outstr.replace('“', '"').replace('”','"').replace("\t", " ").replace("\n", "")

def convert_dset(df, dsname, outdir):
    dsname = os.path.splitext(dsname)[0]
    fout = open(f"{outdir}{dsname}.tsv", "w")

    for j, row in enumerate(df.itertuples(index=False)):
        question = replace_chars(row[0])
        numcols = len(row)
        answer_char = row[numcols-1]
        answer_idx = int(chr(ord(answer_char) - 17))
        answer_text = 'ERROR ANSWER NOT FOUND'
        choices_text = ''
        for i, col in enumerate(row[1:numcols-1]):
            if answer_idx == i:
                answer_text = replace_chars(col)
            choices_text = choices_text + ' (' + chr(ord('A')+i) + ') ' + replace_chars(col)
        fout.write(f"{question.strip()} \\n {choices_text.strip()}\t{answer_text.strip()}\n")  
        #if j > 14: break
    fout.close()
    
def copy_dset(dsname, outdir, unifiedqadir):
    unifiedqa_name = os.path.join(unifiedqadir, 'mmlu_'+os.path.splitext(dsname)[0])
    print(f"Creating directory: {unifiedqa_name}")
    os.makedirs(unifiedqa_name, exist_ok=True) 
    src = os.path.join(outdir, dsname)
    dest = os.path.join(unifiedqa_name, 'test.tsv')   
    print(f"Copying {src} to {dest} ...")
    shutil.copyfile(src, dest)
        
    


#df = pd.read_csv(indir+dsets[0], header=None)  # test
#convert_dset(df, dsets[0], outdir)

for dset in dsets:
    print(f'Processing: {dset} into {outdir}')
    df = pd.read_csv(indir+dset, header=None)  
    convert_dset(df, dset, outdir)
print('Finished!')   

#Copy the mmlu .tsv files into the unifiedQA subdir with form ../unifiedqa/mmlu_dataset_name/test.tsv 
dsets = os.listdir(outdir)
#copy_dset(dsets[0], outdir, unifiedqadir)   # test

for dset in dsets:
    copy_dset(dset, outdir, unifiedqadir)    
print('Finished!')

mmlu = os.listdir(unifiedqadir)
mmlu = [m for m in mmlu if m[0:5] == 'mmlu_']  #filter for mmlu datasets


