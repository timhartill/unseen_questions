#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:06:02 2022

@author: thar011
"""

import utils


claimfile = '/home/thar011/data/SCI/claims.txt'
questions = utils.loadas_txt(claimfile)
outfile = '/data/thar011/data/unifiedqa/claims_test/test.tsv'
utils.create_uqa_from_list(questions, outfile, answers=None, ans_default='NO ANS PROVIDED')
