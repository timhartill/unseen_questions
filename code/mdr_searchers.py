#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:52:39 2022

@author: tim hartill


dense retrieval, stage 1 and stage 2 search classes 

args.prefix = 'TESTITER'
args.predict_file = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val.json'
args.index_path = '/large_data/thar011/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/index.npy'
args.corpus_dict = '/large_data/thar011/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/id2doc.json'
args.model_name = 'roberta-base'
args.init_checkpoint = '/large_data/thar011/out/mdr/logs/hpqa_sent_annots_test1-04-18-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt'
args.gpu_model=True
args.gpu_faiss=True
args.hnsw = True
args.beam_size = 4      # num paras to return from retriever each hop
args.topk = 9           # max num sents to return from stage 1
args.topk_stage2 = 5    # max num sents to return from stage 2
args.max_hops = 2       # max num hops
args.fp16=True
args.max_q_len = 70
args.max_q_sp_len = 400  # retriever max input len

"""

import argparse
import collections
import json
import logging
import os
from os import path
import time
import copy
from datetime import date
from html import unescape  # To avoid ambiguity, unescape all titles

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr_config import eval_args
from mdr.retrieval.models.mhop_retriever import RobertaRetriever_var
from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer

from reader.reader_model import Stage1Model


from utils import encode_text, load_saved, move_to_cuda, return_filtered_list, aggregate_sents

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']



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

        logger.info(f"Loading trained dense retrieval encoder model  {args.model_name}...")
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
            if len(id2doc["0"]) == 3 or not str(id2doc["0"][2]).replace('_', '').isnumeric():
                self.id2doc = {k: {"title":v[0], "text": v[1], "sentence_spans":v[2]} for k, v in id2doc.items()}
            else:
                self.id2doc = {k: {"title":v[0], "text": v[1], "sentence_spans":v[2], "para_id": v[3]} for k, v in id2doc.items()}
                self.evidence_key = 'para_id'
        logger.info(f"Evidence key field: {self.evidence_key}")
        # title2text = {v[0]:v[1] for v in id2doc.values()}
        logger.info(f"Corpus size {len(self.id2doc)}")
        return
        
    def encode_input(self, sample):
        """ Encode a question (and title:sents for hops > 0) for dense encoder input
        sample['s2'] = output from stage 2 = currently selected sents: [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:..}, ..]
        sample['s2'] = [{'title': 'Ed Wood', 'sentence': 'Edward Davis Wood Jr. (October 10, 1924\xa0– December 10, 1978) was an American filmmaker, actor, writer, producer, and director.', 'score':1.0, 'idx': 1787155}]
        eg for sample['s2'] = [{'title':'title_a', 'sentence':'Sent 1', 'score':1.0},{'title':'title_b', 'sentence':'Sent 2', 'score':1.0},{'title':'title_a', 'sentence':' ', 'score':1.0}, {'title':'title_a', 'sentence':'Sent 3', 'score':1.0}, {'title':'title_c', 'sentence':'Sent c1', 'score':1.0}]        
        returns tokenised version of '<s>Were Scott Derrickson and Ed Wood of the same nationality</s></s>title_a:  Sent 1. Sent 3. title_b:  Sent 2. title_c:  Sent c1.</s>'
        """        
        q_sents = aggregate_sents(sample['s2'], title_sep = ':')  # aggregate sents for same title
        if len(q_sents) == 0:
            q_sents = None
        m_input = encode_text(self.tokenizer, sample['question'], text_pair=q_sents, max_input_length=self.args.max_q_sp_len, 
                              truncation=True, padding=False, return_tensors="pt") 
        return m_input
    
    def search(self, sample):
        """ retrieve top beam_size paras
        """
        s2_idxs = set([s['idx'] for s in sample['s2']])  # dont retrieve id2doc keys that are already in s2 output
        dense_query = self.encode_input(sample)  # #create mdr query
        if self.args.gpu_model:
            batch_q_sp_encodes = move_to_cuda(dict(dense_query))

        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            q_sp_embeds = self.model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], 
                                              batch_q_sp_encodes.get("token_type_ids", None), include_stop=False)            
        q_sp_embeds = q_sp_embeds.cpu().contiguous().detach().numpy()  # [1, hs]
        if self.args.hnsw:
            q_sp_embeds = convert_hnsw_query(q_sp_embeds)
        D, I = self.index.search(q_sp_embeds, self.args.beam_size)  # D,I = [1, #beams]
        if self.args.hnsw:
            D = -D  #HNSW smaller is better
        if sample['dense_retrieved'] != []:
            sample['dense_retrieved_hist'].append(sample['dense_retrieved'])
        sample['dense_retrieved'] = []
        for i, idx in enumerate(I[0]):
            if idx not in s2_idxs:
                sample['dense_retrieved'].append( self.id2doc[str(idx)] ) 
                sample['dense_retrieved'][-1]['idx'] = idx
                sample['dense_retrieved'][-1]['score'] = float(D[0, i])
        return
        


class stage1_reranker():
    """ Stage 1 reranker model
    """    
    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger

        logger.info(f"Loading trained stage 1 reranker model  {args.model_name_stage1}...")
        self.bert_config = AutoConfig.from_pretrained(args.model_name_stage1)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_stage1, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
        self.model = Stage1Model(self.bert_config, args)
        self.model = load_saved(self.model, args.init_checkpoint_stage1, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict
        if args.gpu_model:
            device0 = torch.device(type='cuda', index=0)
            self.model.to(device0)
        self.model.eval()
        return

    def encode_input(self, sample):
        """ Encode a question, optional title | sents and retrieved paras for stage 1 reranker
        """





if __name__ == '__main__':
    args = eval_args()
    
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{date_curr}-iterator-fp16{args.fp16}-topkparas{args.beam_size}-topks1sents{args.topk}-topks2sents{args.topk_stage2}-maxhops{args.max_hops}"

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

    logger.info(f"Loading queries from {args.predict_file} ...")
    samples = [json.loads(s) for s in tqdm(open(args.predict_file).readlines())]  # dict_keys(['question', '_id', 'answer', 'sp', 'type', 'src'])
    logger.info("Standardising query formats ...")
    for sample in tqdm(samples):
            if sample["question"].endswith("?"):
                sample["question"] = sample["question"][:-1]
            if sample.get('src') is None:
                sample['src'] = 'hotpotqa'  # superfluous but consistent. The iterator only depends on having a 'question' key and adds the other keys.
            # this hop:    
            sample['dense_retrieved'] = []   # [id2doc idx1, id2doc idx2, ...]  paras retrieved this hop q + each para = s1 query
            sample['s1'] = []   # [ {'title':.. , 'sentence':.., 'score':.., id2doc_key:.., sidx:..}, ..]  selected sentences from s1 = best sents from topk paras this hop
            sample['s2'] = []   # [ {'title':.. , 'sentence':.., 'score':.., id2doc_key:.., sidx:..}, ..]  selected sentences from s2 = best sents to date
            # this hop item appended to hist:
            sample['dense_retrieved_hist'] = []
            sample['s1_hist'] = []
            sample['s2_hist'] = []

    dense_searcher = DenseSearcher(args, logger)

    for i, sample in enumerate(samples):
        for hop in range(0, args.max_hops):
            #create mdr query 
            #dense_query = dense_searcher.encode_input(sample)
            # topkparas = dense search
            dense_searcher.search(sample)
            #TODO create stage 1 query
            #TODO topks1sents = process topkparas through stage1 model
            #TODO create stage 2 query
            #TODO topks2sents, finished = stage2 model(stage2query)
            #TODO if finished break
        if i % 500 == 0:
            logger.info(f"Processed {i} samples.")






