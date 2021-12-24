#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:57:41 2021

@author: tim hartill

Convert the 'standard' open domain versions of Natural Questions and TriviaQA 
from the FiD repository which in turn mirror Lee 2019:
    NQ Counts: Dev:8757  Test:3610  Train:79168
    TQA Counts: Dev:8837  Test:11313  Train:78785

@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2007.01282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

- first git clone https://github.com/facebookresearch/FiD.git 
- then run get-data.sh
- then edit and run this script

FiD json format:
[{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}]

    For each dataset outputs:
        DS_od_ans - q-> a where a = #!# formatted list for dev/test and individual answers for train. 

"""

import os
import copy
import random
import numpy as np

import utils
import text_processing


NQ_DIR = '/home/thar011/data/git/FiD/open_domain_data/NQ/'
TQA_DIR = '/home/thar011/data/git/FiD/open_domain_data/TQA/'

NQ_OUT = '/data/thar011/data/unifiedqa/nq_open_'
TQA_OUT = '/data/thar011/data/unifiedqa/tqa_open_'

MAX_TRAIN_SAMPLES = 4 # max number of individual training samples to output if |answers| > than this

OD_ANS = 'od_ans'


def load_json(in_dir):
    dev = utils.loadas_json( os.path.join(in_dir, 'dev.json') )
    test = utils.loadas_json( os.path.join(in_dir, 'test.json') )
    train = utils.loadas_json( os.path.join(in_dir, 'train.json') )
    return dev, test, train


def process_od(ds, dset_type='train'):
    """ Add key OD_ANS to ds jsonl - a list of single instances if dset_type == 'train' else a single instance with list of answers
    """
    for i, a in enumerate(ds):
        if i != 0 and i % 25000 == 0:
            print(f"Processed: {i}")

        q = a['question']
        ans = [aa.replace('\n', '') for aa in a['answers']]
        sel_indices = set(np.random.choice(len(ans), min(MAX_TRAIN_SAMPLES, len(ans)), replace=False))

        if dset_type != 'train':
            a[OD_ANS] = utils.create_uqa_example(q, None, ans)
        else:
            a[OD_ANS] = [utils.create_uqa_example(q, None, ans_single) for i, ans_single in enumerate(ans) if i in sel_indices]       
    return


def process_set(dev, test, train):
    process_od(dev, dset_type='dev')
    process_od(test, dset_type='dev')
    process_od(train, dset_type='train')
    return


def save_single(split, outdir, ds_type, file):
    """ save a single dataset split """
    out = [s[ds_type] for s in split if s[ds_type] is not None]
    out = utils.flatten(out)
    outfile = os.path.join(outdir, file)
    print(f'Saving {outfile} ...')
    with open(outfile, 'w') as f:
        f.write(''.join(out))    
    return


def save_datasets(dev, test, train, dir_,
                  ds_list=[OD_ANS]):
    """ save uqa-formatted dataset """
    for ds_type in ds_list:
        outdir = dir_ + ds_type
        print(f'Saving dataset to {outdir} ...')
        os.makedirs(outdir, exist_ok=True)
        save_single(dev, outdir, ds_type, 'dev.tsv')
        save_single(test, outdir, ds_type, 'test.tsv')
        save_single(train, outdir, ds_type, 'train.tsv')
    print('Finished saving uqa-formatted datasets!')
    return


nq_dev, nq_test, nq_train = load_json(NQ_DIR)
print(f"NQ Counts: Dev:{len(nq_dev)}  Test:{len(nq_test)}  Train:{len(nq_train)}" )  #NQ Counts: Dev:8757  Test:3610  Train:79168
tqa_dev, tqa_test, tqa_train = load_json(TQA_DIR)
print(f"TQA Counts: Dev:{len(tqa_dev)}  Test:{len(tqa_test)}  Train:{len(tqa_train)}" )  #TQA Counts: Dev:8837  Test:11313  Train:78785

np.random.seed(42)
process_set(nq_dev, nq_test, nq_train)
process_set(tqa_dev, tqa_test, tqa_train)

save_datasets(nq_dev, nq_test, nq_train, dir_=NQ_OUT, ds_list=[OD_ANS])
save_datasets(tqa_dev, tqa_test, tqa_train, dir_=TQA_OUT, ds_list=[OD_ANS])







