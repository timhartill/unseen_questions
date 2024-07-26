#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:49:04 2024

@author: tim hartill


S2ORC file preprocessing

Metadata file row format:
    
{'paper_id': '77490191',
 'title': 'Nature of the reflex excitation of the spinal motor neurons in children with isiopathic scoliosis',
 'authors': [{'first': 'Popov', 'middle': [], 'last': 'Sv', 'suffix': ''},
  {'first': 'Gaĭvoronskiĭ', 'middle': [], 'last': 'Gi', 'suffix': ''}],
 'abstract': None,
 'year': 1977,
 'arxiv_id': None,
 'acl_id': None,
 'pmc_id': None,
 'pubmed_id': None,
 'doi': None,
 'venue': None,
 'journal': 'Zhurnal Nevropatologii I Psikhiatrii Imeni S S Korsakova',
 'mag_id': '2437424934',
 'mag_field_of_study': ['Medicine'],
 'outbound_citations': [],
 'inbound_citations': [],
 'has_outbound_citations': False,
 'has_inbound_citations': False,
 'has_pdf_parse': False,
 's2_url': 'https://api.semanticscholar.org/CorpusID:77490191'}




"""
import os
import gzip
import json

BASE_METADATA = '/media/tim/dl1storage/unseen_questions/S2ORC_test_samples/metadata/'
BASE_PDFPARSE = '/media/tim/dl1storage/unseen_questions/S2ORC_test_samples/pdf_parses/'

FILE_TEMPLATE_META='metadata_0.jsonl.gz'
FILE_TEMPLATE_PDF='pdf_parses_0.jsonl.gz'

OUTPUT_DIR = '/media/tim/dl1storage/unseen_questions/S2ORC_test_samples/out/'

def load_gzip(filename):
    """ Load gzipped jsonl file, return as list of json
    Note alternative: 
        with gzip.open(sample_meta) as f:
          for i in f:
            i = json.loads(i.decode('utf-8'))
            #do something..
    """    
    print(f"Reading {filename}...")
    f = gzip.open(filename)
    lines = f.readlines()
    f.close()
    print("Converting to json list ...")
    lines = [ json.loads(i.decode('utf-8')) for i in lines ] 
    print(f"Finished converting {len(lines)} rows.")
    return lines

sample_meta = os.path.join(BASE_METADATA, FILE_TEMPLATE_META)
sample_pdf = os.path.join(BASE_PDFPARSE, FILE_TEMPLATE_PDF)

meta_0 = load_gzip(sample_meta)
sample_0 = load_gzip(sample_pdf)

#########################################
from datasets import get_dataset_split_names
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("allenai/peS2o")
ds_builder.info.description
ds_builder.info.dataset_size  # 38972211
ds_builder.info.download_size # 87129236480
ds_builder.info.features  
"""{'added': Value(dtype='string', id=None),
 'created': Value(dtype='string', id=None),
 'id': Value(dtype='string', id=None),
 'source': Value(dtype='string', id=None),
 'text': Value(dtype='string', id=None),
 'version': Value(dtype='string', id=None)}
"""

get_dataset_split_names("allenai/peS2o")  # ['train', 'validation']

from datasets import load_dataset
ds = load_dataset("allenai/peS2o", "v2")  # dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST).
ds.cache_files
ds.num_columns
ds.num_rows
ds.column_names
ds["train"][0:3]["text"]

#ds = load_dataset("rotten_tomatoes")
ds['validation'][0] # {'text': 'compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .', 'label': 1}
type(ds['validation'][0])  # dict
ds.num_rows  # {'train': 38811179, 'validation': 161032}
ds.column_names # {'train': ['added', 'created', 'id', 'source', 'text', 'version'], 'validation': ['added', 'created', 'id', 'source', 'text', 'version']}

t_list = [r for r in ds['train']]
v_list = [r for r in ds['validation']]

saveas_jsonl(v_list, "peS2o_validation.jsonl")
saveas_jsonl(t_list, "peS2o_train.jsonl")

