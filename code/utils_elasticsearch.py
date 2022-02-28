#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:03:19 2022

@author: tim hartill

Elasticsearch utils

"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from tqdm import tqdm
from typing import Dict, List, Tuple
import re
import json

# from https://github.com/zycdev/AISO/blob/f5637028ad2cdbba88bdaf3d6bf26cd3859e673f/retriever.py#L20
core_title_pattern = re.compile(r'([^()]+[^\s()])(?:\s*\(.+\))?')
def filter_core_title(x):
    return core_title_pattern.match(x).group(1) if core_title_pattern.match(x) else x


TYPE = "paragraph"  #doc type

settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "en_analyzer": {  
                    "type": "standard",  # does not stem
                    "stopwords": "_english_"
                },
                "simple_bigram_analyzer": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "shingle", "asciifolding"] # "shingle" w/o params adds bigrams to the unigrams. "asciifolding" converts non-ascii chars to ascii equivalents 
                },
                "bigram_analyzer": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "shingle", "asciifolding"]
                }
            }
        }
    },
    "mappings": { TYPE: {
        "properties": {
            "para_id": {   # '1234_2'
                "type": "keyword"
            },
            "para_idx": {  # 2
                "type": "integer"
            },
            "doc_id": {    # '1234'
                "type": "keyword"
            },
            "title": {
                "type": "text",
                "similarity": "BM25",
                "analyzer": "simple",
                "fields": { # index in 2 extra different ways:
                    "exact": { #access as "title.exact"
                        "type": "keyword"
                    },
                    "bigram": {
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "simple_bigram_analyzer",
                    }
                }
            },
            "title_unescaped": {  #Note GR = title_unescape
                "type": "text",
                "similarity": "BM25",
                "analyzer": "simple",
                "fields": {
                    "exact": {
                        "type": "keyword"
                    },
                    "bigram": {
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "simple_bigram_analyzer",
                    }
                }
            },
            "text": {
                "type": "text",
                "similarity": "BM25",
                "analyzer": "en_analyzer",
                "fields": {
                    "bigram": {
                        "type": "text",
                        "analyzer": "bigram_analyzer"
                    }
                }
            },
            "for_hotpot": {
                "type": "boolean"
            },
            "for_squad": {
                "type": "boolean"
            },
#            "hyperlinks": {
#                "type": "object",   #json object. read from here with json.dumps(..). 
#                "enabled": False    # don't parse/index
#            }
        }
    }}
}



def get_esclient(host = 'localhost', port=9200, retries=3, timeout=60):
    return Elasticsearch(hosts=[{"host": host, "port": port}], retries=retries, timeout=timeout)


def ping_client(client):
    return(client.info())


def create_index(client, mapping, index_name, force_reindex=False):
    if client.indices.exists(index=index_name) and force_reindex:
        print(f'Deleting existing index {index_name}...')
        client.indices.delete(index=index_name)
    if not client.indices.exists(index=index_name):
        print(f'Creating index {index_name}...')  
        if type(mapping) != str:
            mapping = json.dumps(mapping)
        res = client.indices.create(index=index_name, ignore=400, body=mapping)
        print(res)
    else:
        print(f'Index {index_name} already exists. Set force_reindex=True to overwrite existing index.')
    return True


def get_chunk(docs, n):
    """Yield successive n-sized chunks from docs"""
    for i in range(0, len(docs), n):
        yield docs[i:i + n]


def add_to_index(client, some_docs):
    """ batch = [ { "_index": index_name, "_type": TYPE, "_id": doc_id, "_source": new_doc} ]
    e.g. new_doc = { "doc_id": para['doc_id'],  "url": para['url'],  "title": para['title'],
                     "title_unescaped": unescape(para['title']), "para_id": para['para_id'],
                     "text": para['text'], "hyperlinks": para['hyperlinks']
                   }
    """
    res = bulk(client, some_docs)
    #assert not 'errors' in res, res
    return len(some_docs)


def index_by_chunk(client, docs, chunksize=50):
    for some_docs in tqdm(get_chunk(docs, chunksize), total=(len(docs) + chunksize - 1) // chunksize):
        add_count = add_to_index(client, some_docs)
    print(f"Finished! {len(docs)} items indexed into {docs[0]['_index']}")
    return


def list_indexes(client):
    return client.indices.get_alias(index="*")  # list indices

def index_stats(client, index_name=None):
    return client.indices.stats(index = index_name) # dict of index stats including doc count

def create_query(query: str, fields: List = None,
               must_not: Dict = None, filter_dic: Dict = None, offset: int = 0, size: int = 50) -> Dict:
    """ Create ES query. Adapted from https://github.com/zycdev/AISO/blob/f5637028ad2cdbba88bdaf3d6bf26cd3859e673f/retriever.py
    """
    if fields is None:
        fields = ["title^1.25", "title_unescaped^1.25", "text",
                  "title.bigram^1.25", "title_unescaped.bigram^1.25", "text.bigram"]
    dsl = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": fields
                    }
                }
            }
        },
        "from": offset,  # number of hits to skip, defaulting to 0
        "size": size     # number of hits to return
    }
    if must_not is not None:
        dsl['query']['bool']['must_not'] = must_not
    if filter_dic:
        dsl['query']['bool']['filter'] = filter_dic  # e.g. {"term": {"for_hotpot": True}}
    return dsl


def exec_query(client, index_name, dsl, **kwargs):
    """ Return data from single query with format like:
    [{'_index': 'wiki_psgs_100',
     '_id': '0',
     '_score': 14.331728,
     '_ignored': ['text.keyword'],
     '_source': {'para_id': '1234_1', ... ,
      'title': 'Aaron',
      'text': '"Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from"'}
     ...}, ..]
    """
    return client.search(dsl, index_name, **kwargs)['hits']['hits']


# adapted from https://github.com/zycdev/AISO/blob/f5637028ad2cdbba88bdaf3d6bf26cd3859e673f/retriever.py#L20
def search(client, index_name, query: str, n_rerank: int = 10, fields: List = None,
           must_not: Dict = None, filter_dic: Dict = None, n_retrieval: int = 50, **kwargs) -> List[Dict]:
    n_retrieval = max(n_rerank, n_retrieval)
    dsl = create_query(query, fields, must_not, filter_dic, size=n_retrieval)
    hits = exec_query(client, index_name, dsl, **kwargs)   #[hit for hit in self.es.search(dsl, self.index_name, **kwargs)['hits']['hits']]
    if n_rerank > 0:
        hits = rerank_with_query(query, hits)[:n_rerank]

    return hits


# from https://github.com/zycdev/AISO/blob/f5637028ad2cdbba88bdaf3d6bf26cd3859e673f/retriever.py#L20
def rerank_with_query(query: str, hits: List[Dict]):
    def score_boost(hit: Dict, q: str):
        title = hit['_source']['title_unescaped']
        core_title = filter_core_title(title)
        q1 = q[4:] if q.startswith('The ') or q.startswith('the ') else q

        score = hit['_score']
        if title in [q, q1]:
            score *= 1.5
        elif title.lower() in [q.lower(), q1.lower()]:
            score *= 1.2
        elif title.lower() in q:
            score *= 1.1
        elif core_title in [q, q1]:
            score *= 1.2
        elif core_title.lower() in [q.lower(), q1.lower()]:
            score *= 1.1
        elif core_title.lower() in q.lower():
            score *= 1.05
        hit['_score'] = score

        return hit

    return sorted([score_boost(hit, query) for hit in hits], key=lambda hit: -hit['_score'])



