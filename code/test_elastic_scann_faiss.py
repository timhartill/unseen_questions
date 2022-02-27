#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:49:03 2021

@author: thar011

SCANN tests

From https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb

"""

import numpy as np
import h5py
import os
import requests
import tempfile
import time

import scann
import faiss
from elasticsearch import Elasticsearch

############### Elasticsearch ###############
# start elasticsearch as daemon: https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html
#    cd /home/thar011/elasticsearch/elasticsearch-7.16.2
#    screen
#    ./bin/elasticsearch -d -p pid
#    to kill: pkill -F pid
# query indices in browser: http://localhost:9200/_cat/indices?v
# query index settings in browser: http://localhost:9200/wiki_psgs_100/_settings

client = Elasticsearch("http://localhost:9200", retries=3, timeout=60)
resp = client.info()
print(resp)

client.indices.get_alias(index="*")  # list indices
search_object = {'match': {'paraId': '1'}}
search_object = {"bool": {"must": [
                                      { "match": { "title":   "Aaron"        }}, # "and"
                                      { "match": { "text": "brother" }}  ] } }
search_object = {"bool": {"must": [
                                      { "multi_match": { "fields": ["title", "text"], 
                                                        "query": "Tempelsman Mushimba", "operator": "and"}}, # "and" here means must match both full (not partial) words "Tempelsman Mushimba" (in any word order) not "Tempelsman" or "Mushimba" alone
                                      { "match": { "text": "brother Ujuma"}}  ] } } #and "text" must match either/both brother or Ujuma 

"""
#6.7 query form demo of multifields:
{
  "mappings": {
    "properties": {
      "text": { 
        "type": "text", #  uses the standard analyzer
        "fields": {
          "english": { 
            "type":     "text",
            "analyzer": "english"  # text.english field uses the english analyzer which stems
          }
        }
      }
    }
  }
}

{
  "query": {
    "multi_match": {
      "query": "quick brown foxes",
      "fields": [ 
        "text",
        "text.english"
      ],
      "type": "most_fields"  # Query both the text and text.english fields and combine the scores. Default = best_fields which takes the max field hit as the score
    }
  }
}
"""    

res = client.search(index="wiki_psgs_100", query=search_object, size=20)
res.keys() # dict_keys(['took', 'timed_out', '_shards', 'hits'])
res['hits']['hits'][0] 
"""
{'_index': 'wiki_psgs_100',
 '_type': 'paragraph',
 '_id': '0',
 '_score': 14.331728,
 '_ignored': ['text.keyword'],
 '_source': {'paraId': '1',
  'title': 'Aaron',
  'text': '"Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from"'}}
"""
res['hits']['hits'][0]['_source']['text']

len(res['hits']['hits'])
###########################################

glovedir = '/home/thar011/data/glove-100-angular'
loc = os.path.join(glovedir, "glove.hdf5")

response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
with open(loc, 'wb') as f:
    f.write(response.content)

glove_h5py = h5py.File(loc, "r")

list(glove_h5py.keys()) # ['distances', 'neighbors', 'test', 'train']

dataset = glove_h5py['train']
queries = glove_h5py['test']
print(dataset.shape) # h5py._hl.dataset.Dataset (1183514, 100)
print(queries.shape) # h5py._hl.dataset.Dataset (10000, 100)
queries_np = np.array(queries)
dataset_np = np.array(dataset)


d = dataset.shape[1] # vector dim
k = 10  # num neighbours to return

normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]  # normalise to make emb trained for cosine similarity into inner product
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README
# normalized_dataset: np array (1183514, 100)

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = scann.scann_ops_pybind.builder(normalized_dataset, k, "dot_product").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()

searcher_nonorm = scann.scann_ops_pybind.builder(dataset_np, k, "squared_l2").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()


        
def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size        
        
# this will search the top 100 of the 2000 leaves, and compute
# the exact dot products of the top 100 candidates from asymmetric
# hashing to get the final top 10 candidates.
start = time.time()
neighbors, distances = searcher.search_batched(queries_np)
end = time.time()

# we are given top 100 neighbors in the ground truth, so select top 10
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k]))  # Recall: 0.89965 Time: 1.9511089324951172
print("Time:", end - start)        
        

# increasing the leaves to search increases recall at the cost of speed
start = time.time()
neighbors, distances = searcher.search_batched(queries_np, leaves_to_search=150)
end = time.time()

print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.92339 Time: 2.447828531265259
print("Time:", end - start) 

# increasing reordering (the exact scoring of top AH candidates) has a similar effect.
start = time.time()
neighbors, distances = searcher.search_batched(queries_np, leaves_to_search=150, pre_reorder_num_neighbors=250)
end = time.time()

print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.93143 Time: 3.1643025875091553
print("Time:", end - start)

# we can also dynamically configure the number of neighbors returned
# currently returns 10 as configued in ScannBuilder()
neighbors, distances = searcher.search_batched(queries_np)
print(neighbors.shape, distances.shape)  # (10000, 10) (10000, 10)

# now returns 20
neighbors, distances = searcher.search_batched(queries_np, final_num_neighbors=20)
print(neighbors.shape, distances.shape) # (10000, 20) (10000, 20)

# we have been exclusively calling batch search so far; the single-query call has the same API
start = time.time()
neighbors, distances = searcher.search(queries_np[0], final_num_neighbors=5)
end = time.time()

print(neighbors) # [ 97478 846101 671078 727732 544474]
print(distances) # [2.5518737 2.539792  2.5383418 2.5097368 2.4656374]
print("Latency (ms):", 1000*(end - start)) # Latency (ms): 1.1560916900634766


# https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md - more decriptiom of params to tweak

# save to disk - directory to save into must exist
searcher.serialize('/data/thar011/gitrepos/faiss/tutorial/python/scann_save')

#load from disk:
s2 = scann.scann_ops_pybind.load_searcher('/data/thar011/gitrepos/faiss/tutorial/python/scann_save')

# Unnormalised comparison:
start = time.time()
neighbors, distances = searcher_nonorm.search_batched(queries_np, leaves_to_search=150, pre_reorder_num_neighbors=250)
end = time.time()

print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.53956 Time: 3.3468635082244873 # recall 0.14 if using "dot product"
print("Time:", end - start)    

#######
# compare faiss
# https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
#######

nlist = 1000  # number of voroni cells 1000 takes forever to train/add

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # by default it performs inner-product search

assert not index.is_trained
index.train(dataset_np)  # must do before adding any data to the index.
assert index.is_trained
index.add(dataset_np)                  # add may be a bit slower as well. Note separate train then add steps


index.nprobe = 256              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.55175 Time: 19.244289875030518
print("Time:", end - start) 

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.5468 Time: 8.135462045669556
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.50183 Time: 0.8248088359832764
print("Time:", end - start)


nlist = 100  # number of voroni cells

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # by default it performs inner-product search

assert not index.is_trained
index.train(dataset_np)  # must do before adding any data to the index.
assert index.is_trained
index.add(dataset_np)                  # add may be a bit slower as well. Note separate train then add steps


index.nprobe = 256              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.55262 Time: 46.2905387878418
print("Time:", end - start) 

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.55262 Time: 46.244651317596436
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.53352 Time: 8.106829643249512
print("Time:", end - start)


nlist = 100  # number of voroni cells

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # by default it performs inner-product search

assert not index.is_trained
index.train(dataset_np)  # must do before adding any data to the index.
assert index.is_trained
index.add(dataset_np)                  # add may be a bit slower as well. Note separate train then add steps


index.nprobe = 256              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.12316 Time: 47.269572734832764
print("Time:", end - start) 

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.12316 Time: 48.3382682800293
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.14607 Time: 5.294668436050415
print("Time:", end - start)


##### try normalised ds - much better

nlist = 100  # number of voroni cells

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # by default it performs inner-product search

assert not index.is_trained
index.train(normalized_dataset)  # must do before adding any data to the index.
assert index.is_trained
index.add(normalized_dataset)                  # add may be a bit slower as well. Note separate train then add steps

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.99998 Time: 48.3382682800293
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.88243 Time: 5.294668436050415
print("Time:", end - start)


nlist = 100  # number of voroni cells

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # by default it performs inner-product search
assert not index.is_trained
index.train(normalized_dataset)  # must do before adding any data to the index.
assert index.is_trained
index.add(normalized_dataset) 

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 1.0 Time: 48.3382682800293
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.90238 Time: 5.294668436050415
print("Time:", end - start)


nlist = 1000  # number of voroni cells

quantizer = faiss.IndexFlatL2(d)  # the index to the cell
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # by default it performs inner-product search
assert not index.is_trained
index.train(normalized_dataset)  # must do before adding any data to the index.
assert index.is_trained
index.add(normalized_dataset) 

index.nprobe = 100              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.95709 Time: 4.838177680969238
print("Time:", end - start)

index.nprobe = 10              # default nprobe is 1, try a few more. If approximate search with IndexIVFFlat returns suboptimal results, we can improve accuracy by increasing the search scope. We do this by increasing the nprobe attribute value — which defines how many nearby cells to search
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.77508 Time: 0.5030732154846191
print("Time:", end - start)



# HNSW test - used for DPR but they do DOT->L2 conversion on embeddings and add extra dim..
# see https://github.com/facebookresearch/DPR/blob/main/dpr/indexer/faiss_indexers.py
index = faiss.IndexHNSWFlat(d, 64) # dpr M=512
#index.hnsw.efSearch = 128 #<- DPR value, default 16
#index.hnsw.efConstruction = 200 #<- DPR value, default 40
index.is_trained # True
index.add(dataset_np)  # takes forever

start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.34819 Time: 0.17808914184570312
print("Time:", end - start)

# try normalised
index = faiss.IndexHNSWFlat(d, 64) # dpr M=512
index.is_trained # True
index.add(normalized_dataset)  

#index.hnsw.efSearch = 128 #<- DPR value, default 16
#index.hnsw.efConstruction = 200 #<- DPR value, default 40
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.61544 Time: 0.14808914184570312
print("Time:", end - start)

index.hnsw.efSearch = 128 #<- DPR value, default 16
index.hnsw.efConstruction = 200 #<- DPR value, default 40
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.87965 Time: 0.810128927230835
print("Time:", end - start)


index = faiss.IndexHNSWFlat(d, 128) # dpr M=512
index.add(normalized_dataset)  

index.hnsw.efSearch = 128 #<- DPR value, default 16
index.hnsw.efConstruction = 200 #<- DPR value, default 40
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.8975 Time: 0.9523766040802002
print("Time:", end - start)


index = faiss.IndexHNSWFlat(d, 256) # dpr M=512
index.add(normalized_dataset)  

index.hnsw.efSearch = 128 #<- DPR value, default 16
index.hnsw.efConstruction = 200 #<- DPR value, default 40
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.90781 Time: 1.1120312213897705
print("Time:", end - start)


# BEST = DPR config:
index = faiss.IndexHNSWFlat(d, 512) # dpr M=512
index.add(normalized_dataset)  # takes forever with m=512

index.hnsw.efSearch = 128 #<- DPR value, default 16
index.hnsw.efConstruction = 200 #<- DPR value, default 40
start = time.time()
distances, neighbors = index.search(queries_np, k)     # actual search was D, I       
end = time.time()
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:, :k])) # Recall: 0.91415 Time: 1.217567682266235
print("Time:", end - start)






