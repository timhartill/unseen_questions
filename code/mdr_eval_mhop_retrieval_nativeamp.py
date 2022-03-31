# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Evaluating trained retrieval model.

Usage:
python eval_mhop_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
     --batch-size 50 \
     --beam-size-1 20 \
     --beam-size-2 5 \
     --topk 20 \
     --shared-encoder \
     --gpu \
     --save-path ${PATH_TO_SAVE_RETRIEVAL}
TJH:     
python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path timtests/hpqa_val_test.json


args.model_name='roberta-base'
args.model_path = 'models/q_encoder.pt'
"""
import argparse
import collections
import json
import logging
import os
from os import path
import time

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from mdr.retrieval.models.mhop_retriever import RobertaRetriever, RobertaRetriever_var
from mdr.retrieval.utils.basic_tokenizer import SimpleTokenizer
from mdr.retrieval.utils.utils import (load_saved, move_to_cuda, para_has_answer)

from utils import encode_text


def get_gpu_resources_faiss(n_gpu, gpu_start=0, gpu_end=-1, tempmem=0):
    """ return vectors of device ids and resources useful for faiss gpu_multiple
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
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', type=str, default=None, help="File containing the evaluation samples.")
    parser.add_argument('--index_path', type=str, default=None, help="index.npy file containing para embeddings [num_paras, emb_dim]")
    parser.add_argument('--corpus_dict', type=str, default=None, help="id2doc.json file containing dict with key id -> title+txt")
    parser.add_argument('--model_path', type=str, default=None, help="Model checkpoint file to load.")
    parser.add_argument('--topk', type=int, default=2, help="topk paths/para sequences to return. Must be <= beam-size^num_steps")
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--max_q_len', type=int, default=70)
    parser.add_argument('--max_c_len', type=int, default=300)
    parser.add_argument('--max_q_sp_len', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=5, help="Number of beams each step (number of nearest neighbours to append each step.).")
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--gpu_faiss', action="store_true", help="Put Faiss index on visible gpu(s).")
    parser.add_argument('--gpu_model', action="store_true", help="Put q encoder on gpu 0 of the visible gpu(s).")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--save_index', action="store_true")
    parser.add_argument('--only_eval_ans', action="store_true")
#    parser.add_argument('--shared_encoder', action="store_true")
    parser.add_argument("--output_dir", type=str, default="", help="Dir to save retrieved para augmented eval samples and eval_log to.")
#    parser.add_argument("--stop_drop", default=0, type=float)
    parser.add_argument('--hnsw', action="store_true", help="Non-exhaustive but fast and relatively accurate. Suitable for FAISS use on cpu.")
    parser.add_argument('--strict', action="store_true")  #TJH Added - load ckpt in 'strict' mode
    parser.add_argument('--exact', action="store_true")  #TJH Added - filter ckpt in 'exact' mode
    parser.add_argument("--use_var_versions", action="store_true", help="Use the generic variable step '..._var' versions.")
    args = parser.parse_args()

#args.gpu_model=True
#args.use_var_versions=True
#args.eval_data='/home/thar011/data/mdr/hotpot/hotpot_qas_val.json'
#args.batch_size=10
#args.fp16=True
#args.gpu_faiss=True

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    print(args)
    
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.eval_data).readlines()]
#{"question": "Were Scott Derrickson and Ed Wood of the same nationality?", "_id": "5a8b57f25542995d1e6f1371", "answer": ["yes"], "sp": ["Scott Derrickson", "Ed Wood"], "type": "comparison"}
#{"question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", "_id": "5a8c7595554299585d9e36b6", "answer": ["Chief of Protocol"], "sp": ["Kiss and Tell (1945 film)", "Shirley Temple"], "type": "bridge"}

    #print(f"ds_items length: {len(ds_items)}")

    # filter
    if args.only_eval_ans:
        ds_items = [_ for _ in ds_items if _["answer"][0] not in ["yes", "no"]]

    logger.info("Loading trained model...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.use_var_versions:
        model = RobertaRetriever_var(bert_config, args)
    else:    
        model = RobertaRetriever(bert_config, args)
    model = load_saved(model, args.model_path, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict
    simple_tokenizer = SimpleTokenizer()

    n_gpu = torch.cuda.device_count()
    logger.info(f"Visible gpus: {n_gpu}")
    if args.gpu_model:
        device0 = torch.device(type='cuda', index=0)
        #cuda = torch.device('cuda')
        model.to(device0)
    if args.gpu_faiss and n_gpu > 1:  #Note: FAISS freezes at index_cpu_to_gpu_multiple if gpu_resources is not a list of res's with global scope, hence defining here..
        tempmem = 0
        print(f"Preparing resources for {n_gpu} GPUs")   
        gpu_resources = []    
        for i in range(n_gpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)
        
    #from apex import amp
    #model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info("Building index...")
    d = 768
    xb = np.load(args.index_path).astype('float32')
    
#    d = 64                           # dimension
#    nb = 1000                      # database size
#    nq = 10                       # nb of queries
#    np.random.seed(1234)             # make reproducible
#    xb = np.random.random((nb, d)).astype('float32')
#    xb[:, 0] += np.arange(nb) / 1000.
#    xq = np.random.random((nq, d)).astype('float32')
#    xq[:, 0] += np.arange(nq) / 1000.

    if args.hnsw:
        if os.path.exists(os.path.join(args.output_dir, "wiki_index_hnsw.index")):
            index = faiss.read_index(os.path.join(args.output_dir, "wiki_index_hnsw.index"))
        else:
            index = faiss.IndexHNSWFlat(d + 1, 512)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 200
            phi = 0
            for i, vector in enumerate(xb):
                norms = (vector ** 2).sum()
                phi = max(phi, norms)
            logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))

            data = xb
            buffer_size = 50000
            n = len(data)
            print(n)
            for i in tqdm(range(0, n, buffer_size)):
                vectors = [np.reshape(t, (1, -1)) for t in data[i:i + buffer_size]]
                norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
                aux_dims = [np.sqrt(phi - norm) for norm in norms]
                hnsw_vectors = [np.hstack((doc_vector, aux_dims[idx].reshape(-1, 1))) for idx, doc_vector in enumerate(vectors)]
                hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
                index.add(hnsw_vectors)
        if args.save_index:
            faiss.write_index(index, os.path.join(args.output_dir, "wiki_index_hnsw"))
    else:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        if args.gpu_faiss:
            if n_gpu == 1:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index) #TJH was 6 which would take 7 gpus but fails if < 7 available.
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                vres, vdev = get_gpu_resources_faiss(n_gpu, gpu_start=0, gpu_end=-1)
                index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
            #tjh https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
            #tjh https://github.com/belvo/faiss/blob/master/benchs/bench_gpu_1bn.py contains example of multigpu sharded index

    
    logger.info(f"Loading corpus...")
    id2doc = json.load(open(args.corpus_dict))
    if isinstance(id2doc["0"], list):
        id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    

    logger.info("Encoding questions and searching")
    questions = [_["question"][:-1] if _["question"].endswith("?") else _["question"] for _ in ds_items]
    metrics = []
    retrieval_outputs = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            # TJH test b_start=0
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ann = ds_items[b_start:b_start + args.batch_size]
            bsize = len(batch_q)
            #TJH for ['a','b','c'] get: {'input_ids': [[0, 102, 2, 1, 1, 1, 1, 1], [0, 428, 2, 1, 1, 1, 1, 1], [0, 438, 2, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]}
            batch_q_encodes = encode_text(tokenizer, batch_q, text_pair=None, max_input_length=args.max_q_len, truncation=True, padding='max_length', return_tensors="pt")
            #batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, padding='max_length', truncation=True, return_tensors="pt")
            if args.gpu_model:
                batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            with torch.cuda.amp.autocast(enabled=args.fp16):
                q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))

            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            if args.hnsw:
                q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
            D, I = index.search(q_embeds_numpy, args.beam_size)  #TJH Return beam_size neighbours for each question

            # 2hop search
            query_pairs = []
            for b_idx in range(bsize):
                for _, doc_id in enumerate(I[b_idx]):  # TJH For each neighbour returned for current question
                    doc = id2doc[str(doc_id)]["text"]
                    if "roberta" in  args.model_name and doc.strip() == "":
                        # doc = "fadeaxsaa" * 100
                        doc = id2doc[str(doc_id)]["title"]
                        D[b_idx][_] = float("-inf")
                    query_pairs.append((batch_q[b_idx], doc))  #TJH question + retrieved doc text for each neighbour
            #TJH given query_pairs = [('a','a'), ('a','b'), ('a','c'), ('b','a'),('b','b'),('b','c')] where a = 102, b = 428, c = 438
            #    the following encodes to {'input_ids': [[0, 102, 2, 2, 102, 2, 1, 1, 1, 1], [0, 102, 2, 2, 428, 2, 1, 1, 1, 1], [0, 102, 2, 2, 438, 2, 1, 1, 1, 1], [0, 428, 2, 2, 102, 2, 1, 1, 1, 1], [0, 428, 2, 2, 428, 2, 1, 1, 1, 1], [0, 428, 2, 2, 438, 2, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]}
            batch_q_sp_encodes = encode_text(tokenizer, query_pairs, text_pair=None, max_input_length=args.max_q_sp_len, truncation=True, padding='max_length', return_tensors="pt")
            #batch_q_sp_encodes = tokenizer.batch_encode_plus(query_pairs, max_length=args.max_q_sp_len, padding='max_length', truncation=True, return_tensors="pt")
            if args.gpu_model:
                batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
            s1 = time.time()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], batch_q_sp_encodes.get("token_type_ids", None))
            # print("Encoding time:", time.time() - s1)

            
            q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()
            s2 = time.time()
            if args.hnsw:
                q_sp_embeds = convert_hnsw_query(q_sp_embeds)
            D_, I_ = index.search(q_sp_embeds, args.beam_size)

            D_ = D_.reshape(bsize, args.beam_size, args.beam_size)
            I_ = I_.reshape(bsize, args.beam_size, args.beam_size)

            # aggregate path scores
            path_scores = np.expand_dims(D, axis=2) + D_  #TJH Add neighbour distances to path so for each orig question now have [beamsize, beamsize] containing (d, d_)

            if args.hnsw:
                path_scores = - path_scores

            for idx in range(bsize): #TJH Score, rank paths, return top k paths and eval vs gt
                search_scores = path_scores[idx]
                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
                                           (args.beam_size, args.beam_size))).transpose()
                retrieved_titles = []
                hop1_titles = []
                paths, path_titles = [], []
                for _ in range(args.topk):
                    path_ids = ranked_pairs[_]
                    hop_1_id = I[idx, path_ids[0]]
                    hop_2_id = I_[idx, path_ids[0], path_ids[1]]
                    retrieved_titles.append(id2doc[str(hop_1_id)]["title"])
                    retrieved_titles.append(id2doc[str(hop_2_id)]["title"])

                    paths.append([str(hop_1_id), str(hop_2_id)])
                    path_titles.append([id2doc[str(hop_1_id)]["title"], id2doc[str(hop_2_id)]["title"]])
                    hop1_titles.append(id2doc[str(hop_1_id)]["title"])
                
                if args.only_eval_ans:
                    gold_answers = batch_ann[idx]["answer"]
                    concat_p = "yes no "
                    for p in paths:
                        concat_p += " ".join([id2doc[doc_id]["title"] + " " + id2doc[doc_id]["text"] for doc_id in p])
                    metrics.append({
                        "question": batch_ann[idx]["question"],
                        "ans_recall": int(para_has_answer(gold_answers, concat_p, simple_tokenizer)),
                        "type": batch_ann[idx].get("type", "single")
                    })
                    
                else:
                    sp = batch_ann[idx]["sp"]
                    assert len(set(sp)) == 2
                    type_ = batch_ann[idx]["type"]
                    question = batch_ann[idx]["question"]
                    p_recall, p_em = 0, 0
                    sp_covered = [sp_title in retrieved_titles for sp_title in sp]
                    if np.sum(sp_covered) > 0:
                        p_recall = 1  #TJH either retrieved para in gold paras
                    if np.sum(sp_covered) == len(sp_covered):
                        p_em = 1      #TJH both retrieved para in gold paras
                    path_covered = [int(set(p) == set(sp)) for p in path_titles]  #TJH alternative way of calculating p_em..
                    path_covered = np.sum(path_covered) > 0
                    recall_1 = 0
                    covered_1 = [sp_title in hop1_titles for sp_title in sp] # 1st retrieved para in gold paras 
                    if np.sum(covered_1) > 0: recall_1 = 1
                    metrics.append({
                    "question": question,
                    "p_recall": p_recall,
                    "p_em": p_em,
                    "type": type_,
                    'recall_1': recall_1,
                    'path_covered': int(path_covered)
                    })


                    # saving when there's no annotations
                    candidate_chains = []
                    for cpath in paths:
                        candidate_chains.append([id2doc[cpath[0]], id2doc[cpath[1]]])
                    
                    retrieval_outputs.append({
                        "_id": batch_ann[idx]["_id"],
                        "question": batch_ann[idx]["question"],
                        "candidate_chains": candidate_chains,
                        # "sp": sp_chain,
                        # "answer": gold_answers,
                        # "type": type_,
                        # "coverd_k": covered_k
                    })

    with open(os.path.join(args.output_dir, 'hpqa_val_test.jsonl'), "w") as out:
        for l in retrieval_outputs:
            out.write(json.dumps(l) + "\n")

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)
    if args.only_eval_ans:
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    else:
        logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in metrics])}')
        logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in metrics])}')
        logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in metrics])}')
        logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'\tAvg PR: {np.mean([m["p_recall"] for m in type2items[t]])}')
            logger.info(f'\tAvg P-EM: {np.mean([m["p_em"] for m in type2items[t]])}')
            logger.info(f'\tAvg 1-Recall: {np.mean([m["recall_1"] for m in type2items[t]])}')
            logger.info(f'\tPath Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')
