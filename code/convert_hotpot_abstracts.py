#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:35:27 2022

@author: tim hartill

Convert HotpotQA abstracts into a jsonl file from the MDR output file to the input format for encoding
(avoiding fooling around with the original hpqa multiple bz2 file format) 

output:
[ {'title': 'the title', 'text': 'the text'} ]

"""

import os
import utils
import json


#not used INDIR_BASE = '/data/thar011/gitrepos/multihop_dense_retrieval/data/hpqa_raw_tim/enwiki-20171001-pages-meta-current-withlinks-abstracts'
OUTDIR = '/data/thar011/gitrepos/multihop_dense_retrieval/data/hpqa_raw_tim'


mdr_hpqa = utils.loadas_json('/data/thar011/gitrepos/multihop_dense_retrieval/data/hotpot_index/wiki_id2doc.json')

mdr_out = [{'title': v['title'], 'text': v['text']} for v in mdr_hpqa.values() if v['text'].strip() != ''] #strips 4 blanks

utils.saveas_jsonl(mdr_out, os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl'))


#test
#datatest = [json.loads(l) for l in open(os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl')).readlines()]
