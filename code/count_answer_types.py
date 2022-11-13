#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:43:27 2022

@author: tim hartill

counts by answer type

input file format:
    
{'ans_type': 'SPAN',
 'pkey': '0',
 'qid': '23801627-ff77-4597-8d24-1c99e2452082',
 'tsv_question': 'What is the company paid on a cost-plus type contract?',
 'tsv_answer': 'our allowable incurred costs plus a profit which can be fixed or variable depending on the contract’s fee arrangement up to predetermined funding levels determined by the customer',
 'ans_from': 'text'}

"""

import os
import json
from collections import Counter

import eval_metrics

atypedir = os.path.join(eval_metrics.UQA_DIR, 'answer_types')
atypefiles = list(eval_metrics.answer_type_map.keys())
atypedict = {}
atypecounts = {}
for file in atypefiles:
    fullpath = os.path.join(atypedir, file)
    print(f"Loading answer types file: {fullpath}")
    atypedict[file] = [json.loads(line.strip()) for line in open(fullpath).readlines()]
    atypecounts[file] = dict(Counter([at['ans_type'] for at in atypedict[file]]))
    
print(atypecounts)
outfile = os.path.join(eval_metrics.UQA_DIR, 'answer_types', 'counts.json')
print(f"Saving to {outfile}")
json.dump(atypecounts, open(outfile, 'w'))    
