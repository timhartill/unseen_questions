#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:50:16 2021

@author: tim hartill

Index the DPR/FiD chunked Wikipedia dump in ElasticSearch

"""

import argparse
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import smart_open
from tqdm import tqdm

def get_esclient(host, port):
    return Elasticsearch(hosts=[{"host": args.host, "port": args.port}], retries=3, timeout=60)


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(
        description='Add paragraphs from chunked Wikipedia tsv file to an Elasticsearch index.')
    parser.add_argument('--host', default="localhost",help='Elastic Search hostname')
    parser.add_argument('--port', default=9200, help='Port number')
    parser.add_argument('--file', default="/home/thar011/data/git/FiD/open_domain_data/psgs_w100.tsv", 
                        help='Path of file to index, e.g. /path/to/my_corpus.json')
    parser.add_argument('--index', default="wiki_psgs_100", 
                        help='Name of index to create')
    
    args = parser.parse_args()

    # Get Index Name
    index_name = args.index

    # Document Type constant
    TYPE = "paragraph"

    # Get an ElasticSearch client
    es = get_esclient(args.host, args.port)

    mapping = '''
    {
      "settings": {
        "index": {
          "number_of_shards": 5,
        }
      }
      "mappings": {
        "paragraph": {
          "dynamic": "false",
          "properties": {
            "paraId": {
               "type": "keyword"
            },  
            "title": {
              "analyzer": "snowball",
              "type": "text"
            },
            "text": {
              "analyzer": "snowball",
              "type": "text",
              "fields": {
                "raw": {
                  "type": "keyword"
                }
              }
            }
          }
        }
      }
    }'''


    # Function that constructs a json body to add each line of the file to index
    def make_documents(f):
        doc_id = 0
        headerline=True
        for l in tqdm(f):
            if headerline:
                headerline=False
                continue
            l_split = l.strip().split('\t')
            if len(l_split) != 3:
                print(f"Skipping {doc_id} due to error processing row: {l_split}")
                continue
            doc = {
                '_op_type': 'create',
                '_index': index_name,
                '_type': TYPE,
                '_id': doc_id,
                '_source': {
                    'paraId': l_split[0],
                    'title': l_split[2],
                    'text': l_split[1]
                }
            }
            doc_id += 1
            yield (doc)


    # Create an index, ignore if it exists already
    try:
        res = es.indices.create(index=index_name, ignore=400, body=mapping)

        # Bulk-insert documents into index
        with smart_open.open(args.file, "r") as f:
            res = bulk(es, make_documents(f))
            doc_count = res[0]

        print("Index {0} is ready. Added {1} documents.".format(index_name, doc_count))

    except Exception as inst:
        print(inst)



