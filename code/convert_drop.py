#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:37:07 2022

Extract answer types from original DROP files

@author: tim hartill


"""

import os
import json

import eval_metrics
import utils

devfile = '/home/thar011/data/drop/drop_dataset/drop_dataset_dev.json'
trainfile = '/home/thar011/data/drop/drop_dataset/drop_dataset_train.json'


dev = json.load(open(devfile))
train = json.load(open(trainfile))


