#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:52:39 2022

@author: tim hartill


iterator over dense retrieval, stage 1 and stage 2 search classes 

Progressively adds keys to the original input as follows:
'question', '_id', 'answer', 'sp', 'type', 'sp_facts', 'src', # from input
'dense_retrieved', # list of paras retrieved for current iteration
           
{'title': 'Lenovo Yoga',
 'text': 'Lenovo Yoga (stylized as Lenovo YOGA or simply as YOGΛ) is a range of laptop and tablet computer products from Lenovo, named for their ability to assume multiple form factors due to a hinged screen.',
 'sentence_spans': [[0, 198]],
 'idx': 1423926,                            # index into dense_searcher.id2doc eg dense_searcher.id2doc[ str(1423926) ]
 'score': 726.983154296875,                 # retriever score of this para in this iteration
 's1_ans_pred': '[unused0]',                # predicted answer from stage 1 using current query + this para - insuff, yes, no or span
 's1_ans_pred_score': 21.05351448059082,    # score of predicted answer from stage 1
 's1_ans_insuff_score': 21.05351448059082,  # score of insuff token from stage 1
 's1_ans_conf_delta': 0.0,                  # difference between prior 2 scores (= 0 if insuff token predicted)
 's1_para_score': 0.0006806915043853223,        # evidentiality score from stage 1 using current query + this para 
 's1_sent_score_max': 0.0007250226335600019}    # max sentence evidentiality score from stage 1 using current query + this para ie score of best sentence in para

's1', 
#list of top 9 sentences from 
prior selected sents from s2 if any sorted by (s2_para_sent_ratio*s1para_score') + (1-s2_para_sent_ratio*s2_score)
+ all paras run through stage 1 in curr iteration sorted by (s1_para_sent_ratios1*para score) + (1-s1_para_sent_ratio*s1sent score)

{'title': 'Lenovo IdeaPad Yoga 13',
 'sentence': 'The Lenovo IdeaPad Yoga 13 is a convertible laptop created by Lenovo and first announced at the International CES 2012 in January.',
 'score': 0.05224253982305527,          # this sentence evidentiality score from stage 1
 's1para_score': 0.08526720106601715,   # evidentiality score from stage 1 using current query + this para
 'idx': 1442367,                        # index into dense_searcher.id2doc eg dense_searcher.id2doc[ str(1423926) ]
 's_idx': 0,                            # index of this sentence in para ie index into 'sentence_spans' key of para
 's2_score': 0.8818813562393188,        # this sentence evidentiality score from stage 2
 's2ev_score': 0.9699244499206543}

's2', 
# list of top 5 sentences as scored after stage 2 on the top 9 sentences from stage 1
sorted by s2_para_sent_ratio*s1para_score' + (1-s2_para_sent_ratio*s2_score)

{'title': 'Lenovo IdeaPad Yoga 13',
 'sentence': 'The Lenovo IdeaPad Yoga 13 is a convertible laptop created by Lenovo and first announced at the International CES 2012 in January.',
 'score': 0.05224253982305527,          # this sentence evidentiality score from stage 1
 's1para_score': 0.08526720106601715,   # evidentiality score from stage 1 using current query + this para
 'idx': 1442367,                        # index into dense_searcher.id2doc eg dense_searcher.id2doc[ str(1423926) ]
 's_idx': 0,                            # index of this sentence in para ie index into 'sentence_spans' key of para
 's2_score': 0.8818813562393188,        # this sentence evidentiality score from stage 2
 's2ev_score': 0.9699244499206543}      # overall evidentiality score of this set of ~9 sentences input into stage 2 denormalised for convenience


's2_full',  # list of all top ~9 sentences as scored after stage 2 ie 's2' before truncation to 5

's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', # answer predictions from stage 2 as per stage 1 except for top 9 sents input, not not per-para 
's2ev_score', # overall evidentiality score of this set of ~9 sentences input into stage 2

'stop_reason', # reason for stopping iteration. For unseen eval typically 'max' ie got to max_hops. See suff_evidence() for other options typically applicable if eval set is in-domain to training

# lists of prior iterations. Note s2_hist is list of 's2_full' not 's2'
'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 

's2_hist_all', 's2_pred_hist_all', 
# like s2_hist but last iteration appended to end for convenience. Created in eval_samples(..)->get_best_hop(...)

'best_hop', 'total_hops', 's2_best', 's2_best_preds', # Created in eval_samples(..)->get_best_hop(...) 
's2_best' is the best iteration (as measured by s2ev_score) 's2_full' list of ~9 sentences ordered by s2_para_sent_ratio*s1para_score' + (1-s2_para_sent_ratio*s2_score)


# assorted eval metrics added by eval_samples:
'answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'act_hops', 'sp_r20_rdist', 'sp_r20', 'sp_r4', 'sp_ract', 'sp_covered_em', 'sp_covered_em_act', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall'])

                                               
# TEST parameters below

args.fp16=False
args.sp_weight = 1.0
args.prefix = 'TESTITER'
args.output_dir = '/large_data/thar011/out/mdr/logs'
#args.predict_file = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val_with_spfacts.jsonl'
args.predict_file = '/home/thar011/data/strategyqa/strategyqa_aristotle_qas_val_with_spfacts.jsonl'

#args.index_path = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/index.npy'
#args.corpus_dict = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/id2doc.json'
#args.index_path = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_hpqaabsgenerics_6gpubs250-09-02-2022/index.npy'
#args.corpus_dict = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_hpqaabsgenerics_6gpubs250-09-02-2022/id2doc.json'
args.index_path = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250-09-02-2022/index.npy'
args.corpus_dict = '/large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250-09-02-2022/id2doc.json'

args.model_name = 'roberta-base'
args.init_checkpoint = '/large_data/thar011/out/mdr/logs/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250_hgx2-09-02-2022-mom-seed16-bsz250-fp16True-lr1e-05-decay0.0-warm0.1-valbsz100-m0.999-k76800-t1.0-ga1-varTrue-cenone/checkpoint_q_best.pt'

args.model_name_stage = 'google/electra-large-discriminator'
args.init_checkpoint_stage1 = '/large_data/thar011/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
args.init_checkpoint_stage2 = '/large_data/thar011/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'
args.gpu_model=True
args.gpu_faiss=True
args.hnsw = False
args.save_index = True
args.beam_size = 150     # num paras to return from retriever each hop
args.topk = 9            # max num sents to return from stage 1 ie prior s2 sents + selected s1 sents <= 9
args.topk_stage2 = 5     # max num sents to return from stage 2
args.s1_use_para_score = True         # Stage 1: use s1 para_score + s1 sent score (vs s1 sent score only) to determine topk sents from stage 1
args.s1_para_sent_ratio = 0.5         # Stage 1 iter select: Ratio for formula s1_para_sent_ratio*s1 para score + (1-s1_para_sent_ratio)*s1 sent score (vs s1 sent score only) in selecting topk sentences.
args.s1_para_sent_ratio_final = -1.0  # Stage 1 final select (will be set to s1_para_sent_ratio if no gold paras): Ratio for formula s1_para_sent_ratio_final*s1 para score + (1-s1_para_sent_ratio_final)*max(s1 sent score) (vs s1 sent score only) in selecting final R@20 topk sentences.
args.s2_use_para_score = True         # Stage 2: Use s1 para score + s2 sent score (vs s2 sent score only) in selecting topk s2 sentences.
args.s2_para_sent_ratio = 0.5         # Stage 2 iter select: Ratio for formula s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting topk sentences while iterating.
args.s2_para_sent_ratio_final = -1.0  # Stage 2 final select (will be set to s1_para_sent_ratio if no gold paras): Ratio for formula s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting final topk sentences.
args.max_q_len = 70
args.max_q_sp_len = 512  # retriever max input seq len was 400
args.max_c_len = 512     # stage models max input seq length
args.max_ans_len = 35
args.predict_batch_size = 26            # batch size for stage 1 model, set small so can fit all models on 1 gpu
args.s2_sp_thresh = 0.10                # s2 sent score min for selection as part of query in next iteration
args.s2_min_take = 2                    # Min number of sentences to select from stage 2
args.max_hops = 4                       # max num hops
args.stop_ev_thresh = 1.01               # stop if s2_ev_score >= this thresh. Set > 1.0 to ignore. 0.6 - 0.91 = best per s2 train eval
args.stop_ansconfdelta_thresh = 99999.0  # stop if s2_ans_conf_delta >= this thresh. Set to large number eg 99999.0 to ignore. 5 seemed reasonable
args.stop_lowerev = False                # If True: Stop iterating if current hop s2ev_score < last hop s2ev_score. 
args.query_use_sentences = False         # If True use title: sents form for para in retriever query otherwise use full para text in query optionally prepended by title (retriever only: s1 always uses title | sents form)
args.query_add_titles = False            # If True: prepend retriever query paras with para title (only if using paras, if using sents always prepending title regardless)
args.output_dataset = '/data/thar011/data/unifiedqa/iter_test/test.tsv'

"""

import json
import logging
import os
import copy
from datetime import date
from html import unescape  # corpus titles are all unescaped when saving but some 'sp' and 'sp_facts' titles may not be

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr_config import eval_args
from mdr.retrieval.models.mhop_retriever import RobertaRetriever_var
#from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer

from reader.reader_model import StageModel, SpanAnswerer
from reader.hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp

import utils
from utils import (encode_text, load_saved, move_to_cuda, create_grouped_metrics, saveas_jsonl, flatten, unique_preserve_order,
                   aggregate_sents, encode_query_paras, concat_title_sents, context_toks_to_ids, collate_tokens)
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

        logger.info(f"Loading trained dense retrieval encoder model {args.model_name} from {args.init_checkpoint}...")
        self.bert_config = AutoConfig.from_pretrained(args.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
                logger.info(f"Creating HNSW index from {args.index_path}. WARNING this will take a long time and a lot of RAM ...")
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
                buffer_size = args.hnsw_buffersize #20000000  #1000000000  #50000
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
            if len(id2doc["0"]) == 3 or not str(id2doc["0"][3]).replace('_', '').isnumeric():
                self.id2doc = {k: {"title":v[0], "text": v[1], "sentence_spans":v[2]} for k, v in id2doc.items()}
            else:
                self.id2doc = {k: {"title":v[0], "text": v[1], "sentence_spans":v[2], "para_id": v[3]} for k, v in id2doc.items()}
                self.evidence_key = 'para_id'
        else:
            if id2doc["0"].get("para_id") is not None:
                self.evidence_key = 'para_id'
            self.id2doc = id2doc
        logger.info(f"Evidence key field: {self.evidence_key}")
        logger.info(f"Corpus size {len(self.id2doc)}")
        return
        
    def encode_input(self, sample):
        """ Encode a question (and title:sents for hops > 0) for dense encoder input optionally with a given initial context
        sample['s2'] = output from stage 2 = currently selected sents: [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:..}, ..]
        sample['s2'] = [{'title': 'Ed Wood', 'sentence': 'Edward Davis Wood Jr. (October 10, 1924\xa0– December 10, 1978) was an American filmmaker, actor, writer, producer, and director.', 'score': 1.0, 's1para_score':1.0,  'idx': 1787155, 's_idx': 0}]
        eg for sample['s2'] = [{'title':'title_a', 'sentence':'Sent 1', 'score':1.0},{'title':'title_b', 'sentence':'Sent 2', 'score':1.0},{'title':'title_a', 'sentence':' ', 'score':1.0}, {'title':'title_a', 'sentence':'Sent 3', 'score':1.0}, {'title':'title_c', 'sentence':'Sent c1', 'score':1.0}]        
        returns tokenised version of '<s>Were Scott Derrickson and Ed Wood of the same nationality</s></s>title_a:  Sent 1. Sent 3. title_b:  Sent 2. title_c:  Sent c1.</s>'
        """
        q = sample['question'].strip()
        if sample['mc_options'] != '': # treat multichoice options as part of question
            q += ' ' + sample['mc_options'].strip()
        
        if self.args.query_use_sentences:
            q_sents = aggregate_sents(sample['s2'], title_sep = ':')  # aggregate sents for each title always prepending title
            if sample['init_context'] != '':
                q_sents = sample['init_context'][:600] + ' ' + q_sents
            if len(q_sents) == 0:
                q_sents = None
        else:
            q_sents = ''
            corpus_idxs = []
            for sent in sample['s2']:
                if sent['idx'] not in corpus_idxs:
                    corpus_idxs.append(sent['idx'])
                    para = self.id2doc[ str(sent['idx']) ]
                    q_sents += ' ' + encode_query_paras(para['text'], para['title'], 
                                                        use_sentences=False, prepend_title=self.args.query_add_titles, 
                                                        title_sep=':')
            if sample['init_context'] != '': # treat initial given para like it was the 1st retrieved para
                q_sents = sample['init_context'][:600] + ' ' + q_sents
        
        m_input = encode_text(self.tokenizer, q, text_pair=q_sents, max_input_length=self.args.max_q_sp_len, 
                              truncation=True, padding=False, return_tensors="pt") 
        return m_input
    
    def search(self, sample):
        """ retrieve beam_size nearest paras and put them in 'dense_retrieved' key
        """
        with torch.inference_mode():
            s2_idxs = set([s['idx'] for s in sample['s2']])  # skip id2doc keys that are already in s2 output
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
                if idx not in s2_idxs: # skip paras already selected by s2 model
                    sample['dense_retrieved'].append( copy.deepcopy(self.id2doc[str(idx)]) )
                    sample['dense_retrieved'][-1]['idx'] = int(idx)
                    sample['dense_retrieved'][-1]['score'] = float(D[0, i])
        return
        

class Stage1Searcher():
    """ Stage 1 reranker model
    """    
    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger

        logger.info(f"Loading trained stage 1 reranker model {args.model_name_stage} from {args.init_checkpoint_stage1}...")
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
        """ Encode a question, possible retrieved title | sents/para for stage 1 reranker including optional given initial context
        Return:
            model_inputs: dict with keys(['input_ids', 'token_type_ids', 'attention_mask', 'paragraph_mask', 'sent_offsets'])
            batch_extras: dict with keys(['wp_tokens', 'doc_tokens', 'tok_to_orig_index', 'insuff_offset'])
        """
        q = sample['question'].strip()
        if sample['mc_options'] != '': # treat multichoice options as part of question
            q += ' ' + sample['mc_options'].strip()
        
        q_toks = self.tokenizer.tokenize(q)[:self.args.max_q_len]
        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.args.max_c_len - para_offset - 1
        q_sents = aggregate_sents(sample['s2'], title_sep = ' |', para_sep='[unused2]')  # aggregate sents for each title: '' if sample['s2']=[]
        if sample['init_context'] != '': # treat given initial context like it was 1st retrieved para
            q_sents = '[unused2] ' + sample['init_context'][:600].replace(':', ' |', 1) + ' ' + q_sents
        batch_extras = {'wp_tokens':[], 'doc_tokens':[], 'tok_to_orig_index':[], 'insuff_offset':[]}  # each key [len(sample['dense_retrieved'])] of data used for deriving answer span
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
            model_inputs["input_ids"].append(input_ids)
            model_inputs["token_type_ids"].append(token_type_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs['paragraph_mask'].append(paragraph_mask)
            model_inputs["sent_offsets"].append( torch.LongTensor(sent_offsets) )
            batch_extras["wp_tokens"].append(all_doc_tokens)
            batch_extras["doc_tokens"].append(doc_tokens)
            batch_extras["tok_to_orig_index"].append(tok_to_orig_index)
            batch_extras["insuff_offset"].append(ans_offset+2)
        model_inputs["input_ids"] = collate_tokens(model_inputs["input_ids"], self.tokenizer.pad_token_id)  # [beam_size, max inp seq len]
        model_inputs["token_type_ids"] = collate_tokens(model_inputs["token_type_ids"], 0)                  # [beam_size, max inp seq len]
        model_inputs["attention_mask"] = collate_tokens(model_inputs["attention_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["paragraph_mask"] = collate_tokens(model_inputs["paragraph_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["sent_offsets"] = collate_tokens(model_inputs["sent_offsets"], 0)                      # [beam_size, max # sents]
        return model_inputs, batch_extras, para_offset  


    def search(self, sample):
        """ select topk sentences + each para evidentiality from the beam_size 'dense_retrieved' paras and put them in the 's1' key.
        """
        with torch.inference_mode():
            model_inputs, batch_extras, para_offset = self.encode_input(sample)        
            scores = []  # para rank score
            sp_scores = [] # sent scores
            start_logits_all = []
            end_logits_all = []
            for h_start in range(0, model_inputs['input_ids'].shape[0], self.args.predict_batch_size): 
                batch_encodes = {'input_ids': model_inputs['input_ids'][h_start:h_start+self.args.predict_batch_size],
                                 'token_type_ids': model_inputs['token_type_ids'][h_start:h_start+self.args.predict_batch_size],
                                 'attention_mask': model_inputs['attention_mask'][h_start:h_start+self.args.predict_batch_size],
                                 'paragraph_mask': model_inputs['paragraph_mask'][h_start:h_start+self.args.predict_batch_size],
                                 'sent_offsets': model_inputs['sent_offsets'][h_start:h_start+self.args.predict_batch_size] }
                if self.args.gpu_model:
                    batch_encodes = move_to_cuda(dict(batch_encodes))
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(batch_encodes)  # dict_keys(['start_logits', 'end_logits', 'rank_score', 'sp_score'])
                    scores_b = outputs["rank_score"]
                    scores_b = scores_b.sigmoid().view(-1).tolist()  
                    sp_scores_b = outputs["sp_score"]    # [bs, max#sentsinbatch]
                    sp_scores_b = sp_scores_b.float().masked_fill(batch_encodes["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores_b)  #mask scores past end of # sents in sample
                    sp_scores_b = sp_scores_b.sigmoid().tolist()  # [bs, max#sentsinbatch]  [0.678, 0.5531, 0.0, 0.0, ...]
                    start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]  # [ [bs, maxseqleninbatch], [bs, maxseqleninbatch] 
                scores.extend(scores_b)
                sp_scores.extend(sp_scores_b)
                start_logits_all.append(start_logits)
                end_logits_all.append(end_logits)
            
            outs = [torch.cat(start_logits_all, dim=0), torch.cat(end_logits_all, dim=0)]
            span_answerer = SpanAnswerer(batch_extras, outs, para_offset, batch_extras["insuff_offset"], self.args.max_ans_len) # answer prediction
            
            out_list = []  # [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:.., 's1para_score':..}, ..]
            for idx in range(len(scores)):
                rank_score = scores[idx]
                para = sample['dense_retrieved'][idx]
                para['s1_ans_pred'] = span_answerer.pred_strs[idx]
                para['s1_ans_pred_score'] = span_answerer.span_scores[idx]
                para['s1_ans_insuff_score'] = span_answerer.insuff_scores[idx]
                para['s1_ans_conf_delta'] = span_answerer.ans_delta[idx]
                para['s1_para_score'] = rank_score
                best_s1sent_score = -1.0
                # get the sp sentences [ [title1, 0], [title1, 2], ..]
                for s_idx, sp_score in enumerate(sp_scores[idx]):
                    if int(model_inputs['sent_offsets'][idx, s_idx]) == 0:  # 0 = padding = past # sents in this sample
                        break
                    s, e = para['sentence_spans'][s_idx]
                    sent = para['text'][s:e].strip()
                    if sp_score > best_s1sent_score:
                        best_s1sent_score = sp_score
                    if para.get('para_id') is None:  #hpqa abstracts
                        out_list.append( {'title': para['title'], 'sentence': sent, 'score': sp_score, 's1para_score': rank_score,
                                          'idx': para['idx'], 's_idx': s_idx}) 
                    else: # full wiki
                        out_list.append( {'title': para['title'], 'sentence': sent, 'score': sp_score, 's1para_score': rank_score,
                                          'idx': para['idx'], 'para_id': para['para_id'], 's_idx': s_idx}) 
                para['s1_sent_score_max'] = best_s1sent_score

            out_list.sort(key=lambda k: self.args.s1_para_sent_ratio*k['s1para_score'] + 
                                        (1-self.args.s1_para_sent_ratio)*k['score'] if self.args.s1_use_para_score else k['score'], 
                          reverse=True)
            if sample['s1'] != []:
                sample['s1_hist'].append( copy.deepcopy(sample['s1']) )
            sample['s1'] = (sample['s2'] + out_list)[:self.args.topk]
            return 


class Stage2Searcher():
    """ Stage 2 reranker model
    """
    def __init__(self, args, logger): 
        self.args = args
        self.logger = logger

        logger.info(f"Loading trained stage 2 reranker model {args.model_name_stage} from {args.init_checkpoint_stage2}...")
        self.bert_config = AutoConfig.from_pretrained(args.model_name_stage)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_stage, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
        self.model = StageModel(self.bert_config, args)
        self.model = load_saved(self.model, args.init_checkpoint_stage2, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict
        if args.gpu_model:
            device0 = torch.device(type='cuda', index=0)
            self.model.to(device0)
        self.model.eval()
        return

    def encode_input(self, sample):
        """ Encode a question, possible title | sents and retrieved paras for stage 2 reranker
        Return:
            model_inputs: dict with keys(['input_ids', 'token_type_ids', 'attention_mask', 'paragraph_mask', 'sent_offsets'])
            batch_extras: dict with keys(['wp_tokens', 'doc_tokens', 'tok_to_orig_index', 'insuff_offset'])
        """
        q = sample['question'].strip()
        if sample['mc_options'] != '': # treat multichoice options as part of question
            q += ' ' + sample['mc_options'].strip()

        q_toks = self.tokenizer.tokenize(q)[:self.args.max_q_len]
        if sample['init_context'] != '':
            init_toks = self.tokenizer.tokenize(' ' + sample['init_context'][:600])
            q_toks += init_toks
        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.args.max_c_len - para_offset - 1
        #q_sents = aggregate_sents(sample['s2'], title_sep = ' |', para_sep='[unused2]')  # aggregate sents for same title
        batch_extras = {'wp_tokens':[], 'doc_tokens':[], 'tok_to_orig_index':[], 'insuff_offset':[]}  # each key [bs=1] of data used for deriving answer span
        model_inputs = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[], 'paragraph_mask':[], 'sent_offsets':[]}  # each key [bs=1]
        context = NON_EXTRACTIVE_OPTIONS + concat_title_sents(sample['s1'])
        (doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, 
         all_doc_tokens, sent_starts) = context_toks_to_ids(context, self.tokenizer, sent_marker='[unused1]', special_toks=["[SEP]", "[unused0]", "[unused1]"])
        if len(all_doc_tokens) > max_toks_for_doc:
            all_doc_tokens = all_doc_tokens[:max_toks_for_doc]
        input_ids = torch.tensor([q_ids + self.tokenizer.convert_tokens_to_ids(all_doc_tokens) + [self.tokenizer.sep_token_id]],
                                 dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
        paragraph_mask = torch.zeros(input_ids.size()).view(-1)
        paragraph_mask[para_offset:-1] = 1  #set para toks -> 1
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
        model_inputs["input_ids"].append(input_ids)
        model_inputs["token_type_ids"].append(token_type_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs['paragraph_mask'].append(paragraph_mask)
        model_inputs["sent_offsets"].append( torch.LongTensor(sent_offsets) )
        batch_extras["wp_tokens"].append(all_doc_tokens)
        batch_extras["doc_tokens"].append(doc_tokens)
        batch_extras["tok_to_orig_index"].append(tok_to_orig_index)
        batch_extras["insuff_offset"].append(ans_offset+2)
        model_inputs["input_ids"] = collate_tokens(model_inputs["input_ids"], self.tokenizer.pad_token_id)  # [beam_size, max inp seq len]
        model_inputs["token_type_ids"] = collate_tokens(model_inputs["token_type_ids"], 0)                  # [beam_size, max inp seq len]
        model_inputs["attention_mask"] = collate_tokens(model_inputs["attention_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["paragraph_mask"] = collate_tokens(model_inputs["paragraph_mask"], 0)                  # [beam_size, max inp seq len]
        model_inputs["sent_offsets"] = collate_tokens(model_inputs["sent_offsets"], 0)                      # [beam_size, max # sents]
        return model_inputs, batch_extras, para_offset


    def search(self, sample):
        """ select sentences + sents evidentiality from the topk s1 sents and put them in the 's2' key.
        """
        with torch.inference_mode():
            model_inputs, batch_extras, para_offset = self.encode_input(sample)        
            scores = []  # para rank score
            sp_scores = [] # sent scores
            start_logits_all = []
            end_logits_all = []
            for h_start in range(0, model_inputs['input_ids'].shape[0], self.args.predict_batch_size): #unnecessary since bs=1 here but keep for consistency with s1 model
                batch_encodes = {'input_ids': model_inputs['input_ids'][h_start:h_start+self.args.predict_batch_size],
                                 'token_type_ids': model_inputs['token_type_ids'][h_start:h_start+self.args.predict_batch_size],
                                 'attention_mask': model_inputs['attention_mask'][h_start:h_start+self.args.predict_batch_size],
                                 'paragraph_mask': model_inputs['paragraph_mask'][h_start:h_start+self.args.predict_batch_size],
                                 'sent_offsets': model_inputs['sent_offsets'][h_start:h_start+self.args.predict_batch_size] }
                if self.args.gpu_model:
                    batch_encodes = move_to_cuda(dict(batch_encodes))
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(batch_encodes)  # dict_keys(['start_logits', 'end_logits', 'rank_score', 'sp_score'])
                    scores_b = outputs["rank_score"]
                    scores_b = scores_b.sigmoid().view(-1).tolist()  # added .sigmoid()  list [bs] = 0.46923
                    sp_scores_b = outputs["sp_score"]    # [bs, max#sentsinbatch]
                    sp_scores_b = sp_scores_b.float().masked_fill(batch_encodes["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores_b)  #mask scores past end of # sents in sample
                    sp_scores_b = sp_scores_b.sigmoid().tolist()  # [bs, max#sentsinbatch]  [0.678, 0.5531, 0.0, 0.0, ...]
                    start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]  # [ [bs, maxseqleninbatch], [bs, maxseqleninbatch] 
                scores.extend(scores_b)
                sp_scores.extend(sp_scores_b)
                start_logits_all.append(start_logits)
                end_logits_all.append(end_logits)
            
            outs = [torch.cat(start_logits_all, dim=0), torch.cat(end_logits_all, dim=0)]
            span_answerer = SpanAnswerer(batch_extras, outs, para_offset, batch_extras["insuff_offset"], self.args.max_ans_len) # answer prediction
            
            if sample['s2ev_score'] != -1.0:
                sample['s2_pred_hist'].append( [sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2ev_score']] )
            idx=0
            rank_score = scores[idx]
            sample['s2_ans_pred'] = span_answerer.pred_strs[idx]
            sample['s2_ans_pred_score'] = span_answerer.span_scores[idx]
            sample['s2_ans_insuff_score'] = span_answerer.insuff_scores[idx]
            sample['s2_ans_conf_delta'] = span_answerer.ans_delta[idx]
            sample['s2ev_score'] = rank_score

            out_list = []  # [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:.., 's1para_score':..}, ..]
            # get the sp sentences [ [title1, 0], [title1, 2], ..]
            for s_idx, sp_score in enumerate(sp_scores[idx]):
                if int(model_inputs['sent_offsets'][idx, s_idx]) == 0:  # 0 = padding = past # sents in this sample
                    break
                out = sample['s1'][s_idx]  # add/update s2 keys in s1 list as well as s2
                out['s2_score'] = sp_score
                out['s2ev_score'] = rank_score
                out_list.append( out )

            out_list.sort(key=lambda k: self.args.s2_para_sent_ratio*k['s1para_score'] + 
                                        (1-self.args.s2_para_sent_ratio)*k['s2_score'] if self.args.s2_use_para_score else k['s2_score'], 
                          reverse=True)

            if sample['s2_full'] != []:
                sample['s2_hist'].append( copy.deepcopy(sample['s2_full']) )  # in eval, filter out sents < minscore if want to match 's2' history.. 
            sample['s2_full'] = copy.deepcopy(out_list)
            
            if len(out_list) > self.args.s2_min_take:  # take at least min_take, filter out any below args.s2_sp_thresh later if desired
                minscore = min(self.args.s2_sp_thresh, out_list[self.args.s2_min_take-1]['s2_score'] - 1e-10)
                out_list = [o for o in out_list if o['s2_score'] > minscore]
            sample['s2'] = out_list[:self.args.topk_stage2]
            return


def suff_evidence(args, hop, sample):
    """ Determine whether to stop or continue iteration.
    """
    if sample['s2ev_score'] >= args.stop_ev_thresh:
        sample['stop_reason'].append('evsuff')
    if sample['s2_ans_conf_delta'] >= args.stop_ansconfdelta_thresh:
        sample['stop_reason'].append('aconf')
    if hop+1 >= args.max_hops:
        sample['stop_reason'].append('max')
    if args.stop_lowerev and len(sample['s2_pred_hist']) > 0 and sample['s2_pred_hist'][-1][4] >= sample['s2ev_score']:
        sample['stop_reason'].append('evlower')
    return True if len(sample['stop_reason']) > 0 else False
    

def get_best_hop(sample):
    """ for s2 append the final hop to the historical hops and choose the best one based on highest s2ev_score
    If run to max_hops (eg ev score thresh higher than max ev score encountered) it's possible for an intermediate hop
    to actually be the best one.
    sample['s2_pred_hist']: list of [sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2_ev_score']]
    """
    sample['s2_hist_all'] = copy.deepcopy( sample['s2_hist'] )
    if sample.get('s2_full') is not None:
        sample['s2_hist_all'].append( sample['s2_full'] )  # current s2 is final hop so put it on end of hist list
    else:  # backwards compatability
        sample['s2_hist_all'].append( sample['s2'] )
    sample['s2_pred_hist_all'] = copy.deepcopy( sample['s2_pred_hist'] )
    sample['s2_pred_hist_all'].append( [ sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2ev_score'] ] )
    best_score = -1.0
    best_hop = -1
    for i, pred_hist in enumerate(sample['s2_pred_hist_all']):
        if pred_hist[4] > best_score:
            best_score = pred_hist[4]
            best_hop = i
            sample['best_hop'] = best_hop + 1
            sample['total_hops'] = len(sample['s2_hist_all'])
            sample['s2_best'] = sample['s2_hist_all'][best_hop]
            for s in sample['s2_best']:
                s['s2ev_score'] = best_score  # early bug had s2_hist ev scores all set to the latest one so copy the true s2_evscore back from pred_hist
            sample['s2_best_preds'] = {'s2_ans_pred': pred_hist[0], 's2_ans_pred_score':pred_hist[1], 
                                       's2_ans_insuff_score': pred_hist[2], 's2_ans_conf_delta': pred_hist[3],
                                       's2ev_score': pred_hist[4], 's2_best_hop': best_hop}
    return
    

def eval_samples(args, logger, samples):
    """ Eval on samples
        # para/psg/sp EM, para F1/prec/recall : topk final para evidence field = sp - can use for bqa/full wiki
        # para R@k : all sp titles in top k retrieved paras (irrespective of how many paras actually retrieved...look through dense hist and/or look through s1 hist)
        # sentences em, sentences f1/prec/recall on [ [evidencefieldA, possentidx1], [evidencefieldA, possentidx2], ... ] - hpqa corpus only
        # ans em, ans f1 - on s2 answer
        # joint em, f1
        # above per stop_reason  ['evsuff', 'aconf', 'max'] also by type and act_hops = len(sp)
    """
    if samples[0]['dense_retrieved'][0].get('para_id'):
        evidence_key = 'para_id'    # full wiki
    else:
        evidence_key = 'title'      # hpqa abstracts

    logger.info("Calculating best hop for each sample..")
    for sample in samples:
        get_best_hop(sample)  # obtain best data from best hop by s2_ev_score

    # calc best args.s1_para_sent_ratio_final and args.s2_para_sent_ratio_final
    if (args.s1_use_para_score or args.s2_use_para_score) and (args.s1_para_sent_ratio_final == -1.0 or args.s2_para_sent_ratio_final == -1.0):
        para_to_sent_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        best_s2_ratio_sp_facts_em = -1.0
        best_s2_sp_facts_em = -1.0
        best_s2_ratio_sp_em = -1.0
        best_s2_sp_em = -1.0
        best_ratio_r20 = -1.0
        best_r20 = -1.0
        best_ratio_ract = -1.0
        best_ract = -1.0
        if samples[0]['dense_retrieved'][-1].get('s1_sent_score_max') is None:  #older version of iterator didn't add 's1_sent_score_max' key in dense_retrieved[_hist] so add it from s1 + s1_hist
            logger.info("Detected older version of samples, adding 's1_sent_score_max' key where possible...")
            for sample in samples:
                s1_dict = {}
                for s1_sent in sample['s1']:
                    if s1_dict.get( s1_sent['idx'] ) is None:
                        s1_dict[s1_sent['idx']] = -1.0
                    if s1_sent['score'] > s1_dict[ s1_sent['idx'] ]:
                        s1_dict[s1_sent['idx']] = s1_sent['score']
                for dr in sample['dense_retrieved']:
                    dr['s1_sent_score_max'] = s1_dict.get( dr['idx'], 0.0 )
                for h, s1 in enumerate(sample['s1_hist']):
                    s1_dict = {}
                    for s1_sent in s1:
                        if s1_dict.get( s1_sent['idx'] ) is None:
                            s1_dict[s1_sent['idx']] = -1.0
                        if s1_sent['score'] > s1_dict[ s1_sent['idx'] ]:
                            s1_dict[s1_sent['idx']] = s1_sent['score']
                    for dr in sample['dense_retrieved_hist'][h]:
                        dr['s1_sent_score_max'] = s1_dict.get( dr['idx'], 0.0 )
        
        logger.info("Calculating best s1 and s2 para score to sentence score ratios (if gold paras present)...")                        
        for ps_ratio in para_to_sent_ratios:
            sp_facts_ems = []
            sp_ems = []
            r20_ems = []
            ract_ems = []
            for sample in samples:
                s2bestsorted = sorted(sample['s2_best'], 
                                      key=lambda k: ps_ratio*k['s1para_score'] +
                                                   (1-ps_ratio)*k['s2_score'] if args.s2_use_para_score else k['s2_score'], 
                                      reverse=True)
                sp_sorted_unique = unique_preserve_order([s[evidence_key] for s in s2bestsorted])
                if sample.get('sp_facts') is not None and sample['sp_facts'] != []:
                    act_sents = len(sample['sp_facts'])
                    sp_facts_pred = [ [s[evidence_key], s['s_idx']] for s in s2bestsorted ][:act_sents]
                    sp_facts_covered = [sp_fact in sp_facts_pred for sp_fact in sample['sp_facts']]
                    sp_facts_ems.append( int(np.sum(sp_facts_covered) == len(sp_facts_covered)) ) #works for variable # of sp facts
                if sample.get('sp') is not None and sample['sp'] != []:
                    act_hops = len(sample['sp'])
                    sp_pred = sp_sorted_unique[:act_hops]
                    sp_covered = [sp_title in sp_pred for sp_title in sample['sp']]
                    sp_ems.append( int(np.sum(sp_covered) == len(sp_covered)) )
                    all_retrieved = sample['dense_retrieved'] + flatten(sample['dense_retrieved_hist'])
                    all_retrieved.sort(key=lambda k: ps_ratio*k['s1_para_score'] + 
                                                     (1-ps_ratio)*k.get('s1_sent_score_max', 0.0), 
                                       reverse=True)
                    all_retrieved = all_retrieved[:100]
                    sp_r20_ev = unique_preserve_order([r[evidence_key] for r in all_retrieved])
                    sp_covered = [sp_title in sp_r20_ev[:20] for sp_title in sample['sp']]
                    r20_ems.append( int( np.sum(sp_covered) == len(sp_covered) ) ) 
                    sp_covered = [sp_title in sp_r20_ev[:act_hops] for sp_title in sample['sp']]
                    ract_ems.append( int( np.sum(sp_covered) == len(sp_covered) ) )
                    
            if len(sp_facts_ems) > 0:
                mean_sp_facts = np.mean(sp_facts_ems)
                if mean_sp_facts > best_s2_sp_facts_em:
                    best_s2_ratio_sp_facts_em = ps_ratio
                    best_s2_sp_facts_em = mean_sp_facts
            if len(sp_ems) > 0:
                mean_sp = np.mean(sp_ems)
                if mean_sp > best_s2_sp_em:
                    best_s2_ratio_sp_em = ps_ratio
                    best_s2_sp_em = mean_sp
            if len(r20_ems) > 0:
                mean_r20 = np.mean(r20_ems)
                if mean_r20 > best_r20:
                    best_ratio_r20 = ps_ratio
                    best_r20 = mean_r20
            if len(ract_ems) > 0:
                mean_ract = np.mean(ract_ems)
                if mean_ract > best_ract:
                    best_ratio_ract = ps_ratio
                    best_ract = mean_ract
                    
        logger.info(f"Best s2_para_sent_ratio_final by sp_facts_em: {best_s2_ratio_sp_facts_em}  sp_facts_em:{best_s2_sp_facts_em} [info only]")
        logger.info(f"Best s2_para_sent_ratio_final by sp_em: {best_s2_ratio_sp_em}  sp_em:{best_s2_sp_em}")
        logger.info(f"Best s1_para_sent_ratio_final by r@20: {best_ratio_r20}  r@20:{best_r20}")
        logger.info(f"Best s1_para_sent_ratio_final by r@acthops: {best_ratio_ract}  r@acthops:{best_ract} [info only]")
        if args.s2_use_para_score and args.s2_para_sent_ratio_final == -1.0:
            if best_s2_ratio_sp_em != -1.0:
                logger.info(f"Setting s2_para_sent_ratio_final to {best_s2_ratio_sp_em}")
                args.s2_para_sent_ratio_final = best_s2_ratio_sp_em
            else:
                logger.info(f"New s2_para_sent_ratio_final not found. Setting to {args.s2_para_sent_ratio}")
                args.s2_para_sent_ratio_final = args.s2_para_sent_ratio
        if args.s1_use_para_score and args.s1_para_sent_ratio_final == -1.0:
            if best_ratio_r20 != -1.0:
                logger.info(f"Setting s1_para_sent_ratio_final to {best_ratio_r20}")
                args.s1_para_sent_ratio_final = best_ratio_r20
            else:
                logger.info(f"New s1_para_sent_ratio_final not found. Setting to {args.s1_para_sent_ratio}")
                args.s1_para_sent_ratio_final = args.s1_para_sent_ratio

    logger.info(f"Eval using s1_para_sent_ratio_final: {args.s1_para_sent_ratio_final} and s2_para_sent_ratio_final: {args.s2_para_sent_ratio_final}")
    for sample in samples: # calc final numbers on best hop using best s1_para_sent_ratio_final and s2_para_sent_ratio_final
        #get_best_hop(sample)
        s2bestsorted = sorted(sample['s2_best'], 
                              key=lambda k: args.s2_para_sent_ratio_final*k['s1para_score'] +
                                           (1-args.s2_para_sent_ratio_final)*k['s2_score'] if args.s2_use_para_score else k['s2_score'], 
                              reverse=True)
        sp_sorted_unique = unique_preserve_order([s[evidence_key] for s in s2bestsorted])

        sample['answer_em'] = exact_match_score(sample['s2_best_preds']['s2_ans_pred'], sample['answer'][0]) # not using multi-answer versions of exact match, f1
        f1, prec, recall = f1_score(sample['s2_best_preds']['s2_ans_pred'], sample['answer'][0])
        sample['answer_f1'] = f1

        joint_em, joint_f1, sf_em = -1.0, -1.0, -1.0
        if sample.get('sp_facts') is not None and sample['sp_facts'] != []:
            act_sents = len(sample['sp_facts'])
            metrics = {'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0}
            sample['sp_facts_pred'] = [ [s[evidence_key], s['s_idx']] for s in s2bestsorted ][:act_sents] #if s['s2_score'] > args.s2_sp_thresh]
            update_sp(metrics, sample['sp_facts_pred'], sample['sp_facts'])
            joint_prec = prec * metrics['sp_prec']
            joint_recall = recall * metrics['sp_recall']
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = sample['answer_em'] * metrics['sp_em']
            
            sp_facts_all = [ [s[evidence_key], s['s_idx']] for s in s2bestsorted ]
            sp_facts_covered = [sp_fact in sp_facts_all for sp_fact in sample['sp_facts']]
            sf_em = int(np.sum(sp_facts_covered) == len(sp_facts_covered))  #works for variable # of sp facts
        else:
            metrics = {'sp_em': -1.0, 'sp_f1': -1.0, 'sp_prec': -1.0, 'sp_recall': -1.0}
        sample['sp_facts_covered_em'] = sf_em   # all gold facts in s2_full
        sample['sp_facts_em'] = metrics['sp_em']  #NB in s1/s2 train eval the sp_ keys ie sentence metrics are called sp_facts here since sp_ here is used for para level metrics
        sample['sp_facts_f1'] = metrics['sp_f1']
        sample['sp_facts_prec'] = metrics['sp_prec']
        sample['sp_facts_recall'] = metrics['sp_recall']
        sample['joint_em'] = joint_em
        sample['joint_f1'] = joint_f1

        act_hops, p_em, p_r20_rdist, p_r20, p_r4, p_ract, p_em_act = -1, -1, -1, -1, -1, -1, -1
        if sample.get('sp') is not None and sample['sp'] != []:
            act_hops = len(sample['sp'])
            
            # R@20_rdist = all gold_paras in top 10 retrieved paras from each hop ordered by retriever distance
            ret_by_hop = sample['dense_retrieved_hist'] + [sample['dense_retrieved']]
            #ret_by_hop = ret_by_hop[:sample['best_hop']]
            take_per_hop = 10 #// sample['best_hop']
            #if 20 % sample['best_hop'] > 0:  # best_hop = 3
            #    take_per_hop += 1
            top_retrieved = []
            for ret in ret_by_hop:
                ret.sort(key=lambda k: args.s1_para_sent_ratio_final*k['s1_para_score'] + 
                                             (1-args.s1_para_sent_ratio_final)*k.get('s1_sent_score_max', 0.0), 
                         reverse=True)
                top_retrieved.extend(ret[:take_per_hop])
            top_retrieved = unique_preserve_order([r[evidence_key] for r in top_retrieved])
            sp_covered = [sp_title in top_retrieved[:20] for sp_title in sample['sp']]
            p_r20_rdist = int( np.sum(sp_covered) == len(sp_covered) )

            # R@20: = all gold paras in top 20 retrieved paras from all retrieved paras
            all_retrieved = sample['dense_retrieved'] + flatten(sample['dense_retrieved_hist'])
            all_retrieved.sort(key=lambda k: args.s1_para_sent_ratio_final*k['s1_para_score'] + 
                                             (1-args.s1_para_sent_ratio_final)*k.get('s1_sent_score_max', 0.0), 
                               reverse=True)
            all_retrieved = all_retrieved[:100]
            sample['sp_r20_ev'] = unique_preserve_order([r[evidence_key] for r in all_retrieved])  # occasionally there are duplicates in top 20..
            sp_covered = [sp_title in sample['sp_r20_ev'][:20] for sp_title in sample['sp']]
            p_r20 = int( np.sum(sp_covered) == len(sp_covered) )  #works for variable # of sp paras
            sp_covered = [sp_title in sample['sp_r20_ev'][:4] for sp_title in sample['sp']]
            p_r4 = int( np.sum(sp_covered) == len(sp_covered) )  #works for variable # of sp paras
            sp_covered = [sp_title in sample['sp_r20_ev'][:act_hops] for sp_title in sample['sp']]
            p_ract = int( np.sum(sp_covered) == len(sp_covered) )  #works for variable # of sp paras
            
            sample['sp_pred'] = sp_sorted_unique[:act_hops]
            sp_covered = [sp_title in sample['sp_pred'] for sp_title in sample['sp']]
            p_em_act = int( np.sum(sp_covered) == len(sp_covered) )  #redundant with sp_em, kept for prior compatibility
            metrics = {'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0}
            update_sp(metrics, sample['sp_pred'], sample['sp'])
            
            sp_all = list(set( [s[evidence_key] for s in sample['s2_best'] ] ))
            sp_covered = [sp_title in sp_all for sp_title in sample['sp']]
            p_em = int( np.sum(sp_covered) == len(sp_covered) )  #works for variable # of sp paras
        else:
            metrics = {'sp_em': -1.0, 'sp_f1': -1.0, 'sp_prec': -1.0, 'sp_recall': -1.0}
        sample['act_hops'] = act_hops
        sample['sp_r20_rdist'] = p_r20_rdist
        sample['sp_r20'] = p_r20
        sample['sp_r4'] = p_r4
        sample['sp_ract'] = p_ract
        sample['sp_covered_em'] = p_em  # all gold paras in s2_full preds  sp_em for MDR / psg-EM for Baleen
        sample['sp_covered_em_act'] = p_em_act # all gold paras in top act_hops preds
        sample.update(metrics)
    return 


def build_context(args, logger, samples, tokenizer, max_toks=507, output_tsv=True):
    """ Build context from samples using info in 's2_best' key which is the best hop in s2_full as measured by s2 ev score.

    's2_best' is list of:    
    {'title': 'Lenovo IdeaPad Yoga 13',
     'sentence': 'The Lenovo IdeaPad Yoga 13 is a convertible laptop created by Lenovo and first announced at the International CES 2012 in January.',
     'score': 0.05224253982305527,          # this sentence evidentiality score from stage 1
     's1para_score': 0.08526720106601715,   # evidentiality score from stage 1 using current query + this para
     'idx': 1442367,                        # index into dense_searcher.id2doc eg dense_searcher.id2doc[ str(1423926) ]
     's_idx': 0,                            # index of this sentence in para ie index into 'sentence_spans' key of para
     's2_score': 0.8818813562393188,        # this sentence evidentiality score from stage 2
     's2ev_score': 0.9699244499206543}      # overall evidentiality score of this set of ~9 sentences input into stage 2 denormalised for convenience

    Sometimes highest scoring sentences contain only partial info. Therefore we optionally take a sentence 
    before and after each high scoring sentence in a paragraph where possible. 
 
    eg for "Did Aristotle use a laptop" we get two sentences from the paragraph:
    
    {'title': 'Epson HX-20',
     'text': 'The Epson HX-20 (also known as the HC-20) was the first laptop computer. It was invented in July 1980 by Yukio Yokozawa, who worked for Suwa Seikosha, a branch of Japanese company Seiko (now Seiko Epson), receiving a patent for the invention. It was announced in 1981 as the HC-20 in Japan, and was introduced by Epson in North America as the HX-20 in 1981, at the COMDEX computer show in Las Vegas, where it drew significant attention for its portability. It had a mass-market release in July 1982, as the HC-20 in Japan and as the Epson HX-20 in North America. The size of an A4 notebook and weighing 1.6\xa0kg, it was both the first notebook and handheld computer, for which it was hailed by "BusinessWeek" magazine as the “fourth revolution in personal computing”.',
     'sentence_spans': [[0, 72], [72, 242], [242, 456], [456, 562], [562, 765]]}
    
    The second sentence "It was invented in July 1980..." has s2_score 0.81 however without the first sentence we 
    lack the context that the HX-20 is a laptop.
    In this case the first sentence is also in 's2_best' with a lower score, but that may not necessarily be the case.

    Then we output the resulting condensed paras in s2_para_sent_ratio*s1para_score' + (1-s2_para_sent_ratio*s2_score), 
    stopping when more than max_toks are in the context
    
    tokenizer = dense_searcher.tokenizer = roberta = bart but without ind digit tokenisation so only approximate.
    Also question may have been expanded with implicit relations.
        
    Note: Must call eval_samples(...) first as it builds prerequisite keys..
    """
    ps_ratio = 0.5
    if logger is not None:
        logger.info("Building context for each sample from best hop...")
    outlist = []
    for sample in tqdm(samples):
        all_retrieved = sample['dense_retrieved'] + flatten(sample['dense_retrieved_hist'])
        all_retrieved.sort(key=lambda k: ps_ratio*k['s1_para_score'] + 
                                         (1-ps_ratio)*k.get('s1_sent_score_max', 0.0), 
                           reverse=True) # sorting unnecessary but is a rough heuristic to minimise looping time below.. 
        context_dict = {}
        for sent in sample['s2_best']:  # context will be sorted by s2_para_sent_ratio*s1para_score' + (1-s2_para_sent_ratio)*s2_score
            s_idx = sent['s_idx']
            p_idx = sent['idx']
            if context_dict.get(p_idx) is None:
                foundpara = None
                for para in all_retrieved:
                    if para['idx'] == p_idx:
                        foundpara = para
                        break
                assert foundpara is not None
                context_dict[p_idx] = {'para': foundpara, 's_idxs':[]}
            if s_idx > 0 and s_idx-1 not in context_dict[p_idx]['s_idxs'] and not args.ctx_gold_sents_only:   # Add sentence before
                context_dict[p_idx]['s_idxs'].append(s_idx-1)
            if s_idx not in context_dict[sent['idx']]['s_idxs']:
                context_dict[p_idx]['s_idxs'].append(s_idx)
            if s_idx+1 < len(context_dict[p_idx]['para']['sentence_spans'])-1 and s_idx+1 not in context_dict[sent['idx']]['s_idxs'] and not args.ctx_gold_sents_only:  # Add sentence after
                context_dict[p_idx]['s_idxs'].append(s_idx+1)
        context_list = []
        for p_idx in context_dict:
            context_dict[p_idx]['s_idxs'].sort()      # Put sentences in order within a para
            context_str = unescape(context_dict[p_idx]['para']['title']).strip() + ': '
            for s_idx in context_dict[p_idx]['s_idxs']:
                s, e = context_dict[p_idx]['para']['sentence_spans'][s_idx]
                context_str += context_dict[p_idx]['para']['text'][s:e].strip() + ' '
            context_dict[p_idx]['para_summary'] = context_str.strip()
            context_list.append(context_dict[p_idx])
        #context_list.sort(key=lambda k: ps_ratio*k['para']['s1_para_score']+(1-ps_ratio)*k.get('s1_sent_score_max', 0.0), reverse=True)
        sample['final_context_all'] = ' '.join([c['para_summary'] for c in context_list])
        if sample['init_context'] != '':
            sample['final_context_all'] = sample['init_context'] + ' ' + sample['final_context_all']
        tok_count = len(tokenizer.tokenize(sample['question'])) # only approximate - not using ind digit tokenisation
        context_str_fitted = ''
        pcount = 0
        for context in context_list:
            if args.ctx_topk_paras != -1 and pcount >= args.ctx_topk_paras:
                break
            para_summary = context['para_summary']
            psum_len = len(tokenizer.tokenize(para_summary))+2
            if tok_count + psum_len < max_toks:
                context_str_fitted += ' ' + para_summary
                pcount += 1
                tok_count += psum_len
            else:
                break
        sample['final_context_fitted'] = context_str_fitted.strip()
        if sample['init_context'] != '':
            sample['final_context_fitted'] = sample['init_context'] + ' ' + sample['final_context_fitted']
        sample['final_context_fitted_count'] = pcount
        q = sample['question'].strip()
        a = sample['answer'][0] if len(sample['answer']) <= 1 else sample['answer']
        
        outlist.append( utils.create_uqa_example(q, 
                                                 utils.create_uqa_context(sample['mc_options'], sample['final_context_fitted']),
                                                 a, append_q_char='?') 
                      )
    if output_tsv:
        out_dir, out_file = os.path.split(args.output_dataset)
        utils.save_uqa(outlist, out_dir, out_file)
        if logger is not None:
            logger.info(f"Saved into {args.output_dataset}")
    return        


if __name__ == '__main__':
    args = eval_args()
    
    samples = None
    num_already_processed = 0
    if args.resume_dir is not None and args.resume_dir.strip() != '':
        args.output_dir = args.resume_dir
        resume_file = os.path.join(args.output_dir, 'samples_with_context.jsonl')
        if os.path.exists(resume_file):
            samples = utils.load_jsonl(resume_file)
            num_already_processed = -1
            for i, sample in enumerate(samples):
                if sample['dense_retrieved'] == []:  # no retrieved paras in dense_retrieved key = unprocessed sample
                    num_already_processed = i
                    break
            if num_already_processed == -1:
                num_already_processed = len(samples)  # all samples already processed, loading so can output using alternative context-building params
        else:
            print(f"Processed samples file not found: {resume_file}. Did you intend to set --resume_dir?")
            assert os.path.exists(resume_file)
            
    else:
        date_curr = date.today().strftime("%m-%d-%Y")
        model_name = f"{args.prefix}-{date_curr}-ITER-16{args.fp16}-tkparas{args.beam_size}-s1tksents{args.topk}-s1useparascr{args.s1_use_para_score}-s2tksents{args.topk_stage2}-s2minsentscr{args.s2_sp_thresh}-stmaxhops{args.max_hops}-stevthresh{args.stop_ev_thresh}-stansconf{args.stop_ansconfdelta_thresh}-rusesents{args.query_use_sentences}-rtitles{args.query_add_titles}"
        args.output_dir = os.path.join(args.output_dir, model_name)
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)    
    logger.info(f"Output log eval_log.txt will be written to: {args.output_dir}")
    n_gpu = torch.cuda.device_count()
    if samples is None:
        logger.info(f"eval_log.txt and samples_with_context.jsonl will be output to: {args.output_dir}")
    elif num_already_processed < len(samples):
        logger.info("Existing partially processed samples_with_context.jsonl found and loaded. Processing will continue and updates written to this file before outputting tsv-formatted datasets.")
    else:
        logger.info("Existing fully processed samples_with_context.jsonl found and loaded. Outputting tsv formatted datasets will continue.")
        
    logger.info(f"Visible gpus: {n_gpu}")
    
    if args.output_dataset is None or args.output_dataset.strip() == '':
        logger.info("WARNING: --output_dataset not specified. tsv-formatted file output will be skipped. If this is in error, stop this job and rerun with --output_dataset /dir/to/output/dataset_name/train|dev|test.tsv")
    else: 
        logger.info(f"tsv-formatted file will be output to: {args.output_dataset}")
        out_tuple = os.path.split(args.output_dataset)
        if out_tuple[1].endswith('.tsv'):
            os.makedirs(out_tuple[0], exist_ok=True)
        else:
            assert out_tuple[1].endswith('.tsv') 
           
    if args.gpu_faiss and n_gpu > 1:  #Note: FAISS freezes at index_cpu_to_gpu_multiple if gpu_resources is not a list of res's with global scope, hence defining here..
        tempmem = 0
        logger.info(f"Preparing FAISS resources for {n_gpu} GPUs")
        gpu_resources = []
        for i in range(n_gpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

    if samples is None:
        logger.info(f"Loading queries from {args.predict_file} ...")
        if args.predict_file.endswith('.tsv'):
            logger.info('Loading as tsv-formatted...')
            samples_tsv = utils.load_uqa_supervised(args.predict_file, ans_lower=False, verbose=True, return_parsed=True)
            # [{'question': 'full q input txt', 'answer': 'ans txt', 'q_only', 'q only', 'mc_options': 'mc options', 'context': 'context'}]
            samples = []
            for i, sample in enumerate(samples_tsv):
                if sample.get('mc_options') is None or sample['mc_options'].strip() == '':
                    mc_options = ''
                    if i == 0:
                        logger.info("No multichoice options. Running question only or question + context.")
                else:
                    mc_options = sample['mc_options'].strip()
                    if i == 0:
                        logger.info('Multichoice options found. Will be added to question in retrieval and will be added to final output..')
                if sample.get('context') is None or sample['context'].strip() == '':
                    init_context = ''
                    if i == 0:
                        logger.info("No initial context. Starting from question only.")
                else:
                    init_context = sample['context'].strip()
                    if i == 0:
                        logger.info("Utilising question+initial context in retriever, s1 and s2...initial context will be output as first portion of new context.")
                    
                samples.append( {'question': sample['q_only'], 
                                 'answer': sample['answer'] if type(sample['answer'])==list else [sample['answer']],
                                 'mc_options': mc_options,
                                 'init_context': init_context,
                                 'src': 'tsv', 'type': 'tsv', '_id': str(i)} )
        else:
            logger.info(f"Loading as jsonl 'qas_val' formatted from {args.predict_file}...")
            samples = [json.loads(s) for s in tqdm(open(args.predict_file).readlines())]
            # dict_keys(['question', '_id', 'answer', 'sp', 'type', 'src'])
    
        logger.info("Standardising query formats ...")
        i = 0
        for sample in tqdm(samples):
            if sample["question"].endswith("?"):
                sample["question"] = sample["question"][:-1]
            if sample.get('src') is None:
                sample['src'] = 'hotpotqa'  # superfluous but consistent. The iterator only depends on having a 'question' key and adds the other keys.
            if sample['answer'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                sample['answer'][0] = 'yes'
            elif sample['answer'][0] in ["REFUTES", "NOT_SUPPORTED"]:
                sample['answer'][0] = 'no'
            if sample.get('sp') is not None:        # a few hpqa pos_para titles are escaped
                sample['sp'] = [unescape(t) for t in sample['sp']]
            if sample.get('sp_facts') is not None:  # a few hpqa pos_para titles are escaped
                for s in sample['sp_facts']:
                    s[0] = unescape(s[0])
            if sample.get('mc_options') is None:  # just to make downstream tsv outputting logic easier..
                sample['mc_options'] = ''
            if sample.get('init_context') is None:  # just to make downstream tsv outputting logic easier..
                sample['init_context'] = ''
            
            # this hop:    
            sample['dense_retrieved'] = []   # [id2doc idx1 para, id2doc idx2 para, ...]  paras retrieved this hop q + each para = s1 query
            sample['s1'] = []   # [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:.., 's1para_score':..}, ..]  selected sentences from s1 = best sents from topk paras this hop
            sample['s2'] = []   # [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:.., 's1para_score':..}, ..]  selected sentences from s2 = best sents to date filtered to s2topk and min score and min take
            sample['s2_full'] = []   # [ {'title':.. , 'sentence':.., 'score':.., idx:.., sidx:.., 's1para_score':..}, ..]  selected sentences from s2 = best sents to date - full unfiltered s2
            sample['s2_ans_pred'] = ''
            sample['s2_ans_pred_score'] = -1.0
            sample['s2_ans_insuff_score'] = -1.0
            sample['s2_ans_conf_delta'] = -1.0
            sample['s2ev_score'] = -1.0
            sample['stop_reason'] = [] # list of stop reasons (can be more than one of ['evsuff', 'aconf', 'max'])
            # this hop item appended to hist:
            sample['dense_retrieved_hist'] = [] # retriever paras history
            sample['s1_hist'] = []  # stage 1 sents history
            sample['s2_hist'] = []  # stage 2 sents history
            sample['s2_pred_hist'] = [] # stage 2 history of pred answers, rank scores etc: list of [sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2_ev_score']]
    else: # samples is not None - resume
        logger.info(f"Resuming job. Number already processed: {num_already_processed} of {len(samples)}. Input data loaded from: {resume_file}")


    if len(samples) > num_already_processed:
        dense_searcher = DenseSearcher(args, logger)
        stage1_searcher = Stage1Searcher(args, logger)
        stage2_searcher = Stage2Searcher(args, logger)
        
        for i, sample in enumerate(samples):
            if i < num_already_processed-1:
                continue
            for hop in range(0, args.max_hops):
                dense_searcher.search(sample)  # retrieve args.beam_size nearest paras and put them in sample['dense_retrieved'] key
                stage1_searcher.search(sample)
                stage2_searcher.search(sample)
                if suff_evidence(args, hop, sample):
                    break
            if i % 500 == 0:
                logger.info(f"Processed {i+num_already_processed} of {len(samples)} samples. Saving partially completed file..")
                saveas_jsonl(samples, os.path.join(args.output_dir, 'samples_with_context.jsonl'))
                logger.info(f"Saved partially completed samples file to {os.path.join(args.output_dir, 'samples_with_context.jsonl')}")

        logger.info("Finished processing all samples.")
        logger.info(f"Saving full updated samples file to {os.path.join(args.output_dir, 'samples_with_context.jsonl')}")
        saveas_jsonl(samples, os.path.join(args.output_dir, 'samples_with_context.jsonl'))
        tokenizer = dense_searcher.tokenizer
    else:  #all samples processed
        logger.info(f"All samples already processed ({num_already_processed}). Skipping inference and moving directly to eval and output stages...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name) 

    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/TESTITER-06-24-2022-iterator-fp16False-topkparas4-topks1sents9-topks2sents5-maxhops2-s1_use_para_scoreTrue/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/TESTITER-06-28-2022-iterator-fp16False-topkparas25-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops2-stopevthresh0.91-stopansconf18.0/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test4_beam100_maxh4-07-01-2022-iterator-fp16False-topkparas100-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops4-stopevthresh0.91-stopansconf18.0-retusesentsTrue-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test3_beam150_maxh3-07-01-2022-iterator-fp16False-topkparas150-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops3-stopevthresh0.91-stopansconf18.0/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test5_beam100_maxh2_paras-07-02-2022-iterator-fp16False-topkparas100-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops2-stopevthresh0.91-stopansconf18.0-retusesentsFalse-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test6_beam100_maxh2_paras_momentum-07-02-2022-iterator-fp16False-topkparas100-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops2-stopevthresh0.91-stopansconf18.0-retusesentsFalse-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test7_beam100_maxh4_paras_momentum-07-02-2022-iterator-fp16False-topkparas100-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops4-stopevthresh0.91-stopansconf18.0-retusesentsFalse-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test8_beam100_maxh2_paras_mdr_orig_bs150-07-03-2022-iterator-fp16False-topkparas100-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops4-stopevthresh0.91-stopansconf18.0-retusesentsFalse-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test9_beam150_maxh4_paras_mdr_orig_bs150-07-03-2022-iterator-fp16False-topkparas150-s1topksents9-s1useparascoreTrue-s2topksents5-s2minsentscore0.1-stopmaxhops4-stopevthresh0.91-stopansconf18.0-retusesentsFalse-rettitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test12_beam150_maxh4_gpufaiss_paras_mdr_orig_bs150-07-04-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh0.91-stansconf18.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test11_beam150_maxh4_allmax_paras_mdr_orig_bs150-07-04-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test13_beam150_maxh4_gpufaiss_m1_paras_mdr_orig_bs150-07-05-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh0.91-stansconf18.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test14_beam150_maxh4_gpufaiss_s2pscr_paras_mdr_orig_bs150-07-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh0.91-stansconf18.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test15_beam150_maxh4_gpufaiss_lwrev_paras_mdr_orig_bs150-07-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test16_beam150_maxh4_gpufaiss_s1psr_paras_mdr_orig_bs150-07-07-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh0.91-stansconf18.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test17_beam150_maxh4_gpufaiss_s2.4_paras_mdr_orig_bs150-07-07-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh0.91-stansconf18.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test18_beam200_maxh4_gpufaiss_paras_mdr_orig_bs150-07-08-2022-ITER-16False-tkparas200-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hpqaeval_test19_beam400_maxh2_gpufaiss_paras_mdr_orig_bs150-07-08-2022-ITER-16False-tkparas400-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops2-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')

    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hovereval_test20_beam25_maxh4_gpufaiss_paras_mdr_orig_bs150-07-10-2022-ITER-16False-tkparas25-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_hovereval_test21_beam25_maxh4_hovertrained-07-10-2022-ITER-16False-tkparas25-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsTrue-rtitlesFalse/samples_with_context.jsonl')

    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_hpqaabst_aristoeval_test24_beam200_maxh4_orig8gpu_bs150-07-12-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_wiki_aristoeval_test25_beam150_maxh4_bqanosquadnqtqabs24-07-12-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')
    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_wiki_aristoeval_test26_beam400_maxh4_bqanosquadnqtqabs24-07-12-2022-ITER-16False-tkparas400-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')


    #samples = utils.load_jsonl('/large_data/thar011/out/mdr/logs/ITER_fullwiki_aristoeval_test40_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesFalse/samples_with_context.jsonl')

    eval_samples(args, logger, samples)  # adds 's2_best' and other keys as well as calculating metrics
    # samples after eval_samples: dict_keys(['question', '_id', 'answer', 'sp', 'type', 'sp_facts', 'src', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'act_hops', 'sp_r20_rdist', 'sp_r20', 'sp_r4', 'sp_ract', 'sp_covered_em', 'sp_covered_em_act', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall'])

    if args.output_dataset is not None and args.output_dataset.strip() != '':
        logger.info("Building contexts and outputting tsv-formatted datasets...")
        build_context(args, logger, samples, tokenizer=tokenizer)
        create_grouped_metrics(logger, samples, group_key='src', metric_keys = ['answer_em', 'answer_f1'])
    else:
        #create_grouped_metrics(logger, samples, group_key='ALL', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist'])
        create_grouped_metrics(logger, samples, group_key='src', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em','sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist'])
        create_grouped_metrics(logger, samples, group_key='stop_reason', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist'])
        create_grouped_metrics(logger, samples, group_key='type', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist'])
        create_grouped_metrics(logger, samples, group_key='total_hops', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist', 'best_hop'])
        create_grouped_metrics(logger, samples, group_key='best_hop', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist', 'total_hops'])
        create_grouped_metrics(logger, samples, group_key='act_hops', metric_keys = ['answer_em', 'answer_f1', 'sp_facts_covered_em', 'sp_facts_em', 'sp_facts_f1', 'sp_facts_prec', 'sp_facts_recall', 'joint_em', 'joint_f1', 'sp_covered_em', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'sp_covered_em_act', 'sp_ract', 'sp_r4', 'sp_r20', 'sp_r20_rdist'])
    
    logger.info('Finished!')
    

    

