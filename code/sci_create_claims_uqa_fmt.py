#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:06:02 2022

@author: thar011
"""

import utils
from text_processing import create_sentence_spans, split_into_sentences



claimfile = '/home/thar011/data/SCI/claims.txt'
questions = utils.loadas_txt(claimfile)
outfile = '/data/thar011/data/unifiedqa/claims_test/test.tsv'
utils.create_uqa_from_list(questions, outfile, answers=None, ans_default='NO ANS PROVIDED')


scicorpusfile = '/home/thar011/data/SCI/corpus.jsonl'
scicorpusfileout = '/home/thar011/data/SCI/sci_corpus_with_sent_spans.jsonl'

sci_corpus = utils.load_jsonl(scicorpusfile)  #dict_keys(['doc_id', 'title', 'abstract', 'metadata', 'scifact_orig'])

#v = sci_corpus[0]
#create_sentence_spans(v['abstract'])
#' '.join(v['abstract'])


sci_abstracts_out = [{'title': v['title'], 'text': ' '.join(v['abstract']), 
                      'sentence_spans': create_sentence_spans(v['abstract'])} for v in sci_corpus] # 500,000
utils.saveas_jsonl(sci_abstracts_out, scicorpusfileout)

#sci_abstracts_out[0]

