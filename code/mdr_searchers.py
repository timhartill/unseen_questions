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
args.model_name_stage = 'google/electra-large-discriminator'
args.init_checkpoint_stage1 = '/large_data/thar011/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
args.init_checkpoint_stage2 = '/large_data/thar011/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
args.gpu_model=True
args.gpu_faiss=True
args.hnsw = True
args.beam_size = 4      # num paras to return from retriever each hop
args.topk = 9           # max num sents to return from stage 1
args.topk_stage2 = 5    # max num sents to return from stage 2
args.max_hops = 2       # max num hops
args.fp16=True
args.max_q_len = 70
args.max_q_sp_len = 400  # retriever max input seq len
args.max_c_len = 512     # stage models max input seq length
args.sp_weight = 1.0

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

from reader.reader_model import StageModel


from utils import encode_text, load_saved, move_to_cuda, return_filtered_list, aggregate_sents, context_toks_to_ids, collate_tokens
from text_processing import get_sentence_list 

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
NON_EXTRACTIVE_OPTIONS = ' [SEP] yes no [unused0] [SEP] '


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
        """ retrieve beam_size nearest paras and put them in 'dense_retrieved' key
        """
        s2_idxs = set([s['idx'] for s in sample['s2']])  # dont retrieve id2doc keys that are already in s2 output
        dense_query = self.encode_input(sample)  # create retriever query
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
        


class Stage1Searcher():
    """ Stage 1 reranker model
    """    
    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger

        logger.info(f"Loading trained stage 1 reranker model  {args.model_name_stage}...")
        self.bert_config = AutoConfig.from_pretrained(args.model_name_stage)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_stage, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
        self.model = StageModel(self.bert_config, args)
        self.model = load_saved(self.model, args.init_checkpoint_stage1, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict
        if args.gpu_model:
            device0 = torch.device(type='cuda', index=0)
            self.model.to(device0)
        self.model.eval()
        return

    def encode_input(self, sample):
        """ Encode a question, optional title | sents and retrieved paras for stage 1 reranker
        """
        q_toks = self.tokenizer.tokenize(sample['question'])[:self.args.max_q_len]
        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.args.max_c_len - para_offset - 1
        q_sents = aggregate_sents(sample['s2'], title_sep = ' |', para_sep='[unused2]')  # aggregate sents for same title
        item_list = []  # [len(sample['dense_retrieved'])] of data used for deriving answer span
        model_inputs = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[], 'paragraph_mask':[], 'sent_offsets':[]}
        for para in sample['dense_retrieved']:
            title, sentence_spans = para["title"].strip(), para["sentence_spans"]
            sents = get_sentence_list(para["text"], sentence_spans)
            pre_sents = []
            for idx, sent in enumerate(sents):
                pre_sents.append("[unused1] " + sent.strip())
            context = title + " " + " ".join(pre_sents)
            context = q_sents + NON_EXTRACTIVE_OPTIONS + context  # ELECTRA tokenises yes, no to single tokens
            (doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, 
             all_doc_tokens, sent_starts) = context_toks_to_ids(context, self.tokenizer, sent_marker='[unused1]', special_toks=["[SEP]", "[unused0]", "[unused1]"])
            if len(all_doc_tokens) > max_toks_for_doc:
                all_doc_tokens = all_doc_tokens[:max_toks_for_doc]
            input_ids = torch.tensor([q_ids + self.tokenizer.convert_tokens_to_ids(all_doc_tokens) + [self.tokenizer.sep_token_id]],
                                     dtype=torch.int64)
            attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
            token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
            paragraph_mask = torch.zeros(input_ids.size()).view(-1)
            paragraph_mask[para_offset:-1] = 1  #set sentences part of query + para toks -> 1
            model_inputs["input_ids"].append(input_ids)
            model_inputs["token_type_ids"].append(token_type_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs['paragraph_mask'].append(paragraph_mask)
            #NOTE if query very long then 1st [SEP] will be the EOS token at pos 511 & ans_offset will be at 512 over max seq len...
            #ans_offset = torch.where(input_ids[0] == tokenizer.sep_token_id)[0][0].item()+1
            ans_offset = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0][0].item()+1  # tok after 1st [SEP] = yes = start of non extractive answer options
            if ans_offset >= 509: #non extractive ans options + eval para truncated due to very long query
                ans_offset = -1  # idx of insuff token  if insuff token != 1
            sent_offsets = []
            for idx, s in enumerate(sent_starts):
                if s >= len(all_doc_tokens): #if wp_tokens truncated, sent labels could be invalid
                    break
                sent_offsets.append(s + para_offset)
                assert input_ids.view(-1)[s+para_offset] == 2  #self.tokenizer.convert_tokens_to_ids("[unused1]")    
            model_inputs["sent_offsets"].append( torch.LongTensor(sent_offsets) )
            item_list.append( {"wp_tokens": all_doc_tokens, "doc_tokens": doc_tokens, 
                               "tok_to_orig_index": tok_to_orig_index, 'insuff_offset': ans_offset+2} )
        model_inputs["input_ids"] = collate_tokens(model_inputs["input_ids"], self.tokenizer.pad_token_id)  # [beam_size, max inp seq len]
        model_inputs["token_type_ids"] = collate_tokens(model_inputs["token_type_ids"], 0)                  # [beam_size, max inp seq len]
        model_inputs["attention_mask"] = collate_tokens(model_inputs["attention_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["paragraph_mask"] = collate_tokens(model_inputs["paragraph_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["sent_offsets"] = collate_tokens(model_inputs["sent_offsets"], 0)                      # [beam_size, max # sents]
        return model_inputs, item_list, para_offset  #TODO run collate_tokens on model_inputs to pad..




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
            if sample['answer'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                sample['answer'][0] = 'yes'
            elif sample['answer'][0] in ["REFUTES", "NOT_SUPPORTED"]:
                sample['answer'][0] = 'no'
            # this hop:    
            sample['dense_retrieved'] = []   # [id2doc idx1, id2doc idx2, ...]  paras retrieved this hop q + each para = s1 query
            sample['s1'] = []   # [ {'title':.. , 'sentence':.., 'score':.., id2doc_key:.., sidx:..}, ..]  selected sentences from s1 = best sents from topk paras this hop
            sample['s2'] = []   # [ {'title':.. , 'sentence':.., 'score':.., id2doc_key:.., sidx:..}, ..]  selected sentences from s2 = best sents to date
            # this hop item appended to hist:
            sample['dense_retrieved_hist'] = []
            sample['s1_hist'] = []
            sample['s2_hist'] = []

    dense_searcher = DenseSearcher(args, logger)
    
    stage1_searcher = Stage1Searcher(args, logger)

    for i, sample in enumerate(samples):
        for hop in range(0, args.max_hops):
            #create mdr query 
            #dense_query = dense_searcher.encode_input(sample)
            # topkparas = dense search
            dense_searcher.search(sample)  # retrieve args.beam_size nearest paras and put them in sample['dense_retrieved'] key
            #TODO create stage 1 query
            model_inputs, item_list, para_offset = stage1_searcher.encode_input(sample)
            #TODO topks1sents = process topkparas through stage1 model
            #TODO create stage 2 query
            #TODO topks2sents, finished = stage2 model(stage2query)
            #TODO if finished break
        if i % 500 == 0:
            logger.info(f"Processed {i} samples.")






