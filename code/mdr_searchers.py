#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:52:39 2022

@author: tim hartill


dense retrieval, stage 1 and stage 2 search classes 


"""

import argparse
import collections
import json
import logging
import os
from os import path
import time
from html import unescape  # To avoid ambiguity, unescape all titles

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr_config import eval_args
from mdr.retrieval.models.mhop_retriever import RobertaRetriever_var
from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer

from utils import encode_text, load_saved, move_to_cuda, return_filtered_list


def get_gpu_resources_faiss(n_gpu, gpu_start=0, gpu_end=-1, tempmem=0):
    """ return vectors of device ids and resources useful for faiss gpu_multiple
        https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
        https://github.com/belvo/faiss/blob/master/benchs/bench_gpu_1bn.py contains example of multigpu sharded index
    """ 
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()  #faiss.Int32Vector()  #TJH was IntVector but deprecation warning
    if gpu_end == -1:
        gpu_end = n_gpu
    for i in range(gpu_start, gpu_end):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i]) #vres.push_back(res)  # TJH was vres.push_back(gpu_resources[i])
    return vres, vdev


def convert_hnsw_query(query_vectors):
    """ Convert query vectors from DotProduct space to L2 space as H
    """
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors


class DenseSearcher():
    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger

        logger.info("Loading trained dense retrieval encoder model  {args.model_name}...")
        self.bert_config = AutoConfig.from_pretrained(args.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.simple_tokenizer = SimpleTokenizer()  #TODO needed in eval?        
        self.model = RobertaRetriever_var(self.bert_config, args)
        self.model = load_saved(self.model, args.init_checkpoint, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict        
        if args.gpu_model:
            device0 = torch.device(type='cuda', index=0)
            self.model.to(device0)
        self.model.eval()

        logger.info("Loading Dense index...")
        d = 768
        if args.hnsw:
            index_path = os.path.join(os.path.split(args.index_path)[0], "index_hnsw.index")
            if os.path.exists(index_path):
                logger.info(f"Reading HNSW index from {index_path} ...")
                index = faiss.read_index(index_path)
            else:
                logger.info(f"Creating HNSW index from {args.index_path}. WARNING this will take a long time ...")
                xb = np.load(args.index_path).astype('float32')  
                logger.info("Building HNSW index ...")
                m = 512  # the number of links per vector - higher is more accurate but uses more RAM. memory usage is (d * 4 + m * 2 * 4) bytes per vector (d=d+1 in this case) ~258GB
                index = faiss.IndexHNSWFlat(d + 1, m) # HNSW only supports L2 so conversions below are to convert dot sim space -> L2
                index.hnsw.efSearch = 128  # speed-accuracy tradeoff: depth of layers explored during search. No impact on build or mem usage. Can be set anytime before searching.
                index.hnsw.efConstruction = 200  # higher=better recall at lower m. No impact on mem usage but on build speed: depth of layers explored during index construction.
                phi = 0
                for i, vector in enumerate(xb):
                    norms = (vector ** 2).sum()
                    phi = max(phi, norms)
                logger.info(f'HNSWF DotProduct -> L2 space phi={phi}')
                buffer_size = 20000000  #1000000000  #50000
                n = len(xb)
                logger.info(f"Indexing {n} vectors with buffer size {buffer_size}...")
                index.verbose = True
                for i in range(0, n, buffer_size):
                    vectors = [np.reshape(t, (1, -1)) for t in xb[i:i + buffer_size]]
                    norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
                    aux_dims = [np.sqrt(phi - norm) for norm in norms]
                    hnsw_vectors = [np.hstack((doc_vector, aux_dims[idx].reshape(-1, 1))) for idx, doc_vector in enumerate(vectors)]
                    hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)  # could save hnsw vectors to disk here then delete xb, vectors, norms, aux_dims
                    del aux_dims
                    del norms
                    del vectors
                    if i+buffer_size >= n:
                        del xb
                    logger.info(f"Finished preprocessing vectors for {i} to {i+buffer_size}. Adding to index ...")
                    index.add(hnsw_vectors)
                    logger.info(f"Finished adding vectors to index for {i} to {i+buffer_size}. ")
                    del hnsw_vectors    
                index.verbose = False
                if args.save_index:
                    logger.info(f"Saving HNSW index to {index_path} ...")
                    faiss.write_index(index, index_path)
        else: # not hnsw
            # SIDE NOTE: if vectors had been encoded for cosine sim objective (eg sentence-transformers) 
            # can use faiss.normalize_L2(xb) (does this in-place) before index.add to perform L2 normalization on the database s.t very vector has same magnitude (sum of the squares always = 1) and cosine similarity becomes indistinguishable from dot product
            # in this case must also do faiss.normalize_L2(xq) on the search query..            
            xb = np.load(args.index_path).astype('float32')  
            logger.info("Building Flat index ...")
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            if args.gpu_faiss:
                logger.info(f"Moving index to {n_gpu} GPU(s) ...")
                if n_gpu == 1:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index) #TJH was 6 which would take 7 gpus but fails if < 7 available.
                else:
                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True
                    vres, vdev = get_gpu_resources_faiss(n_gpu, gpu_start=0, gpu_end=-1)
                    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
            del xb
        logger.info('Dense Index loaded.')
        self.index = index
        
        logger.info(f"Loading corpus mapped to dense index from {args.corpus_dict}...")
        id2doc = json.load(open(args.corpus_dict))
        self.evidence_key = 'title'
        if isinstance(id2doc["0"], list):
            if len(id2doc["0"]) == 2 or not str(id2doc["0"][2]).replace('_', '').isnumeric():
                id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
            else:
                id2doc = {k: {"title":v[0], "text": v[1], "para_id": v[2]} for k, v in id2doc.items()}
                self.evidence_key = 'para_id'
        logger.info(f"Evidence key field: {self.evidence_key}")        
        # title2text = {v[0]:v[1] for v in id2doc.values()}
        logger.info(f"Corpus size {len(id2doc)}")
        self.id2doc = id2doc
        
    def encode_query_input(q, qcontext):
        """ Encode a question and optional para for dense encoder input
        """
        pass
    
    



if __name__ == '__main__':
    args = eval_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)    
    n_gpu = torch.cuda.device_count()
    logger.info(f"Visible gpus: {n_gpu}")
    
        
    if args.gpu_faiss and n_gpu > 1:  #Note: FAISS freezes at index_cpu_to_gpu_multiple if gpu_resources is not a list of res's with global scope, hence defining here..
        tempmem = 0
        logger.info(f"Preparing FAISS resources for {n_gpu} GPUs")
        gpu_resources = []
        for i in range(n_gpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

    dense_searcher = DenseSearcher(args, logger)







