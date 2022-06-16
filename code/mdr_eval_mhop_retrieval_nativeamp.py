# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Evaluating trained retrieval model.

Usage:
python eval_mhop_retrieval.py ${predict_file} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
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
args.init_checkpoint = 'models/q_encoder.pt'
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

from mdr.retrieval.models.mhop_retriever import RobertaRetriever, RobertaRetriever_var
from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer

from utils import encode_text, load_saved, move_to_cuda, return_filtered_list


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
    """ Convert query vectors from DotProduct space to L2 space as H
    """
    aux_dim = np.zeros(len(query_vectors), dtype='float32')
    query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
    return query_nhsw_vectors


if __name__ == '__main__':

    args = eval_args()

#args.gpu_model=True
#args.use_var_versions=True
#args.predict_file='/home/thar011/data/mdr/hotpot/hotpot_qas_val.json'
#args.predict_batch_size=10
#args.fp16=True
#args.gpu_faiss=True
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "eval_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    print(args)
    
    logger.info("Loading data...")
    ds_items = [json.loads(_) for _ in open(args.predict_file).readlines()]
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
    model = load_saved(model, args.init_checkpoint, exact=args.exact, strict=args.strict) #TJH added  strict=args.strict
    simple_tokenizer = SimpleTokenizer()

    n_gpu = torch.cuda.device_count()
    logger.info(f"Visible gpus: {n_gpu}")
    if args.gpu_model:
        device0 = torch.device(type='cuda', index=0)
        #cuda = torch.device('cuda')
        model.to(device0)
    if args.gpu_faiss and n_gpu > 1:  #Note: FAISS freezes at index_cpu_to_gpu_multiple if gpu_resources is not a list of res's with global scope, hence defining here..
        tempmem = 0
        logger.info(f"Preparing FAISS resources for {n_gpu} GPUs")
        gpu_resources = []
        for i in range(n_gpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)
        
    model.eval()

    logger.info("Loading index...")
    d = 768
    
#    xb = np.load(args.index_path).astype('float32')
    
#    d = 64                           # dimension
#    nb = 10000                      # database size
#    nq = 10                       # nb of queries
#    np.random.seed(1234)             # make reproducible
#    xb = np.random.random((nb, d)).astype('float32')
#    xb[:, 0] += np.arange(nb) / 1000.
#    xq = np.random.random((nq, d)).astype('float32')
#    xq[:, 0] += np.arange(nq) / 1000.

    if args.hnsw:
        index_path = os.path.join(os.path.split(args.index_path)[0], "index_hnsw.index")
        if os.path.exists(index_path):
            logger.info(f"Reading HNSW index from {index_path} ...")
            index = faiss.read_index(index_path)
        else:
            xb = np.load(args.index_path).astype('float32')  # , mmap_mode='r' useless - loads whole file into ram
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

            #data = xb
            buffer_size = 20000000  #1000000000  #50000
            n = len(xb)
            logger.info(f"Indexing {n} vectors with buffer size {buffer_size}...")
            index.verbose = True
            for i in range(0, n, buffer_size):
                #i=0
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
            #del xb
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
            #tjh https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
            #tjh https://github.com/belvo/faiss/blob/master/benchs/bench_gpu_1bn.py contains example of multigpu sharded index
        del xb


    
    logger.info(f"Loading corpus mapped to dense index from {args.corpus_dict}...")
    id2doc = json.load(open(args.corpus_dict))
    evidence_key = 'title'
    if isinstance(id2doc["0"], list):
        if len(id2doc["0"]) == 2 or not str(id2doc["0"][2]).replace('_', '').isnumeric():
            id2doc = {k: {"title":v[0], "text": v[1]} for k, v in id2doc.items()}
        else:
            id2doc = {k: {"title":v[0], "text": v[1], "para_id": v[2]} for k, v in id2doc.items()}
            evidence_key = 'para_id'
    logger.info(f"Evidence key field: {evidence_key}")        
    # title2text = {v[0]:v[1] for v in id2doc.values()}
    logger.info(f"Corpus size {len(id2doc)}")
    

    logger.info("Encoding questions and searching")
    questions = [q["question"][:-1] if q["question"].endswith("?") else q["question"] for q in ds_items]
    metrics = []
    retrieval_outputs = []
    firsterr = True
    
    for b_start in tqdm(range(0, len(questions), args.predict_batch_size)):
        with torch.no_grad():
            # TJH test b_start=0
            batch_q = questions[b_start:b_start + args.predict_batch_size]
            batch_ann = ds_items[b_start:b_start + args.predict_batch_size]
            bsize = len(batch_q)
            #TJH for ['a','b','c'] get: {'input_ids': [[0, 102, 2, 1, 1, 1, 1, 1], [0, 428, 2, 1, 1, 1, 1, 1], [0, 438, 2, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]}
            batch_q_encodes = encode_text(tokenizer, batch_q, text_pair=None, max_input_length=args.max_q_len, truncation=True, padding='max_length', return_tensors="pt")
            #batch_q_encodes = tokenizer.batch_encode_plus(batch_q, max_length=args.max_q_len, padding='max_length', truncation=True, return_tensors="pt")
            if args.gpu_model:
                batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            with torch.cuda.amp.autocast(enabled=args.fp16):
                q_embeds = model.encode_q(batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], 
                                          batch_q_encodes.get("token_type_ids", None), include_stop=False)

            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            if args.hnsw:
                q_embeds_numpy = convert_hnsw_query(q_embeds_numpy)
            D, I = index.search(q_embeds_numpy, args.beam_size)  #D,I = [bs, #beams]
            # xq_bs2 = xq[:2]
            # D, I = index.search(xq_bs2, args.beam_size)  #D,I = [bs=2, #beams=2]
            # D = array([[19.885868, 19.528814], [19.600351, 19.417498]], dtype=float32)
            # I = array([[250, 621], [860, 309]])

            # n-hop search
            #stop_on_hop = np.full((bsize, args.max_hops-1), -100, dtype=np.int64)  # [bs, max_hops-1] since we don't calc for q_only hence start with q+sp1 in 0th position
            path_scores = D.copy()
            curr_D = D
            curr_I = I
            curr_doc = ['' for i in range(curr_D.shape[0])]
            curr_q = batch_q.copy()
            I_list = [I.copy()]  # list of max_hops np arrays [ [bs, #beams], [bs, #beams, #beams], .. ] idx 0 = q_only
            stop_on_hop_list = []     # list of max_hops-1 np arrays [ [bs, #beams], [bs, #beams, #beams], .. ] idx 0 = q+sp1

            for curr_hop in range(args.max_hops-1):
                #curr_hop = 0
                curr_size = curr_D.shape[0]
                query_pairs = []
                for b_idx in range(curr_size):
                    for n, doc_id in enumerate(curr_I[b_idx]):  # For each neighbour returned for current question
                        doc = id2doc[str(doc_id)]["text"].strip()  # TJH update for extra sentence prediction model...
                        #doc = str(n)+'_'+str(doc_id)  
                        if "roberta" in  args.model_name and doc.strip() == "":
                            doc = id2doc[str(doc_id)]["title"]
                            curr_D[b_idx][n] = float("-inf")
                        if doc[-1] not in ['.', '?', '!']:  # Force full stop at end
                            doc += '.'
                        doc_path = (curr_doc[b_idx] + ' ' + doc).strip() #concat sps together separated by <space>
                        query_pairs.append((curr_q[b_idx], doc_path))  # q + retrieved doc(s) text for each neighbour q+sp1

                # save questions, doc_paths for next hop:
                curr_q = [qp[0] for qp in query_pairs]  # [bsize * #beams * (curr_hop+1)]
                curr_doc = [qp[1] for qp in query_pairs]
                
                q_sp_embeds_list = []
                stop_01_list = []
                for h_start in range(0, len(query_pairs), args.predict_batch_size): 
                    curr_query_pairs = query_pairs[h_start:h_start + args.predict_batch_size]
                    #TJH given query_pairs = [('a','a'), ('a','b'), ('a','c'), ('b','a'),('b','b'),('b','c')] where a = 102, b = 428, c = 438
                    #    the following encodes to {'input_ids': [[0, 102, 2, 2, 102, 2, 1, 1, 1, 1], [0, 102, 2, 2, 428, 2, 1, 1, 1, 1], [0, 102, 2, 2, 438, 2, 1, 1, 1, 1], [0, 428, 2, 2, 102, 2, 1, 1, 1, 1], [0, 428, 2, 2, 428, 2, 1, 1, 1, 1], [0, 428, 2, 2, 438, 2, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]}
                    batch_q_sp_encodes = encode_text(tokenizer, curr_query_pairs, text_pair=None, max_input_length=args.max_q_sp_len, truncation=True, padding='max_length', return_tensors="pt")
                    if args.gpu_model:
                        batch_q_sp_encodes = move_to_cuda(dict(batch_q_sp_encodes))
    
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        if args.eval_stop:
                            q_sp_embeds, stop_01 = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], 
                                                                  batch_q_sp_encodes.get("token_type_ids", None), include_stop=True) 
                            stop_01_numpy = stop_01.cpu().contiguous().numpy()  # [bs * #beams * (curr_hop+1), 1]
                            # stop_01_numpy = np.array([[0,1,2,3]], dtype=np.int64)
                            stop_01_list.append(stop_01_numpy)
                        else:
                            q_sp_embeds = model.encode_q(batch_q_sp_encodes["input_ids"], batch_q_sp_encodes["attention_mask"], 
                                                         batch_q_sp_encodes.get("token_type_ids", None), include_stop=False)
                        
                    q_sp_embeds = q_sp_embeds.contiguous().cpu().numpy()  # [bs*#beams*(curr_hop+1), hs]
                    q_sp_embeds_list.append( q_sp_embeds )

                q_sp_embeds = np.concatenate(q_sp_embeds_list, axis=0)
                if args.hnsw:
                    q_sp_embeds = convert_hnsw_query(q_sp_embeds)

                D_, I_ = index.search(q_sp_embeds, args.beam_size)  # [bs*#beams*(curr_hop+1), #beams]
                # hop1size = bsize*args.beam_size*(curr_hop+1) #4
                # xq_bs2_hop1 = xq[-hop1size:]  #[4,hs]
                #D_, I_ = index.search(xq_bs2_hop1, args.beam_size)  # [bs*#beams*(curr_hop+1), #beams]
                curr_D = D_
                curr_I = I_
                newshape = [bsize] + [args.beam_size for i in range(curr_hop+2)]
                D_ = D_.reshape(newshape)  # [bsize, #beams, #beams, ...] where num of #beam dims = curr_hop+2
                I_ = I_.reshape(newshape)
                I_list.append(I_.copy())
#                D_ = D_.reshape(bsize, args.beam_size, args.beam_size)  
#                I_ = I_.reshape(bsize, args.beam_size, args.beam_size)
    
                # aggregate path scores
                #path_scores = np.expand_dims(D, axis=2) + D_  #For each orig question now have [bs, beamsize, beamsize] containing d+d_
                # D=q_only: [bs, #beams]  D_1=q+sp1: [bs, #beams, #beams]  D_2=q+sp1+sp2 [bs, #beams, #beams, #beams] ...
                # -> D: [bs, #beams, 1] + D_1: [bs, #beams, #beams] = path_scores: [bs, #beams, #beams] ie each successive axis contains cumulative score of prior hops 
                # -> [bs, #beams, #beams, 1] + [bs, #beams, #beams, #beams] = path_scores: [bs, #beams, #beams, #beams]
                # each element of path score is sum of distances to that point d + d_1 + d_2
                # D_1 = [bs, #beams, #beams] ie [0,0,0] = path score for sample 0, hop 0qonly beam 0, hop 1q+sp1 beam 0
                # D_2 = [bs, #beams, #beams, #beams] ie [0,0,0,0] = path score for sample 0, hop 0qonly beam 0, hop 1q+sp1 beam 0, hop 2q+sp1+sp2 beam 0
                path_scores = np.expand_dims(path_scores, axis=-1) + D_  #For each orig question now have [bs, beamsize, beamsize] containing d+d_
                
        
                if args.eval_stop:
                    stop_01_numpy = np.concatenate(stop_01_list, axis=0)
                    newshape = [bsize] + [args.beam_size for i in range(curr_hop+1)]
                    stop_01_numpy = stop_01_numpy.reshape(newshape) # [bs,  #beams, ...] 
                    # stop_01_numpy = array([[0, 1],
                    #                        [2, 3]])
                    # stop_on_hop: list of max_hops-1 np arrays [ [bs, #beams], [bs, #beams, #beams], .. ]  idx 0 = q+sp1
                    # stop_on_hop[0] = q+sp1 = [bs, #beams] ie [0, 0] = stop pred for sample 0, hop 1q+sp1 beam 0. path score= D_1[sample0, qonlybeam0, q+sp1beam0]
                    # stop_on_hop[1] = q+sp1+sp2 = [bs, #beams, #beams] ie [0, 0, 0] = stop pred for sample 0, hop 1q+sp1 beam 0, hop 2q+sp1+sp2 beam 0
                    stop_on_hop_list.append(stop_01_numpy.copy())
                    #stop_on_hop[:, curr_hop] = stop_01_numpy[:, 0]

            if args.hnsw:
                path_scores = - path_scores
            

            # start eval per batch
            
            for idx in range(bsize): #TJH Score, rank paths, return top k paths and eval vs gt
                search_scores = path_scores[idx]  # [#beams, #beams, ..]
                # search_scores = array([[41.651543, 41.626137], 
                #                        [42.186363, 41.819355]], dtype=float32)
                # ranked_coords = array([[1, 0], [1, 1], [0, 0], [0, 1]])
                beam_shapes = search_scores.shape #(args.beam_size, args.beam_size)
#                ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
#                                           (args.beam_size, args.beam_size))).transpose()  # [#beams * #beams, 2] = ranked coords in [#beams, #beams] matrix of scores
                # .ravel like flatten but returns view of original not copy
                # .argsort returns indices into flattened array
                # .unravel_index converts indices into flattened array into tuple of indices of 2d ([x1, x2, ..], [y1, y2, ..])
                # vstack condenses each tuple into [[y1, x1], [y2, x2], ..] 
                # returns sorted desc coords/indices into search_scores values
                # tst3d = np.random.randn(2,2,2)  # array([ [[ 1.42257859, -0.93570487], [-0.92932392, -1.34603146]], [[ 0.82341017, -0.1254692 ],  [-0.59356903, -1.34296959]]])
                # beam_shapes = tst3d.shape
                # tst3d_coords = np.vstack(np.unravel_index(np.argsort(tst3d.ravel())[::-1], beam_shapes)).transpose() [2*2*2=8, 3]
                # for (z, y, x) in tst3d_coords: print(f"[{z},{y},{x}]={tst3d[z, y, x]}")  #works! also works for 4d...
                ranked_coords = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], beam_shapes)).transpose()  # [#beams * #beams, 2] = ranked coords in [#beams, #beams] matrix of scores

                retrieved_titles = []       # [k * max_hops] list of all separate retrieved titles (docid_pidx) for this sample for each k
                hop1_titles = []            # [k * 1] list of all first hop retrieved titles for this sample for each k
                paths, path_titles = [], [] # [k, max_hops] list of lists of title seq/corpus idx seqs for this sample for each k
                stop_preds = []             # [k] list of stop preds for this sample for each k
                for k in range(args.topk):
                    path_ids = ranked_coords[k]  # [max_hops] = k+1th highest scoring path eg [2, 1, 3] means I[0][idx][2] is best 1st para then I[1][idx][1] is best 2nd para then I[2][idx][3] is best 3rd para if 3 hops                    
                    curr_path = []
                    curr_path_para_ids = []
                    curr_stop_preds = []
                    np_coords_I = [idx]
                    np_coords_stop = [idx]
                    for i, path_id in enumerate(path_ids):
                        np_coords_I.append( path_id )
                        hop_n_id = I_list[i][ tuple(np_coords_I) ]  #nparr[ tuple([idx, path_id[0], ...]) ] - must be tuple not list to work
                        retrieved_titles.append( unescape(id2doc[str(hop_n_id)][evidence_key]) )  
                        curr_path.append( str(hop_n_id) )  
                        curr_path_para_ids.append( unescape(id2doc[str(hop_n_id)][evidence_key]) )
                        if i == 0:
                            hop1_titles.append( unescape(id2doc[str(hop_n_id)][evidence_key]) ) # append 1st hop predicted para for each k
                        if i > 0:  # no stop pred for q_only so start at i=1
                            np_coords_stop.append( path_id )
                            hop_n_stop_pred = stop_on_hop_list[i-1][ tuple(np_coords_stop) ]
                            curr_stop_preds.append( hop_n_stop_pred )
                            
                    paths.append(curr_path)  # append [retrieved para corpus idxs] for each k
                    path_titles.append(curr_path_para_ids)  # append [ title or docid_paraidx ] for each k
                    stop_preds.append (curr_stop_preds)  # Appending stop pred for each k
                
                if args.only_eval_ans:
                    gold_answers = batch_ann[idx]["answer"]
                    concat_p = "yes no "
                    for p in paths:
                        concat_p += " ".join([unescape(id2doc[doc_id]["title"]) + " " + id2doc[doc_id]["text"] for doc_id in p])
                    metrics.append({
                        "question": batch_ann[idx]["question"],
                        "ans_recall": int(para_has_answer(gold_answers, concat_p, simple_tokenizer)),
                        "type": batch_ann[idx].get("type", "single")
                    })
                    
                else:
                    gold_answers = batch_ann[idx]["answer"]
                    concat_p = "yes no "  # make ans_recall 1.0 for all yes/no questions..
                    for p in paths:
                        concat_p += " ".join([unescape(id2doc[doc_id]["title"]) + " " + id2doc[doc_id]["text"] for doc_id in p])
                    ans_recall = int(para_has_answer(gold_answers, concat_p, simple_tokenizer))
                        
                    sp = [unescape(p) for p in batch_ann[idx]["sp"]]
                    
                    if args.eval_stop:
                        act_hops = len(sp)  # num gold paras = num of hops needed to answer this question
                        if act_hops == 0: # no para annotations, find act_hops another way to avoid nan in stop acc calc
                            if batch_ann[idx]["type"].strip() == '':
                                act_hops = 1
                            else:
                                if firsterr:
                                    logger.info(f"WARNING: unable to determine act_hops for sample from {batch_ann[idx]['src']}. stop accuracy will be incorrect. Update program logic to determine act_hops")
                                    firsterr = False
                                
                        stop_preds_np = np.array(stop_preds)
                        stop_target = np.zeros((args.max_hops-1), dtype=np.int64)
                        if act_hops-1 < stop_target.shape[0]:
                            stop_target[act_hops-1] = 1  #aim to stop on act_hop unless act_hop = max_hop 
                            
                        stop_acc = (stop_preds_np == stop_target).astype(np.float64)  # [topk, max_hops-1]
                        for i in range(args.max_hops-1): # ignore where hop > actual # hops for this sample 
                            for k in range(stop_acc.shape[0]):
                                if i > act_hops-1:
                                    stop_acc[k, i] = 0.0
                        correct_counts = stop_acc.sum()
                        act_hops_denom = act_hops if act_hops < args.max_hops else args.max_hops-1
                        act_hops_denom *= stop_acc.shape[0]
                        stop_accuracy_per_sample = float(correct_counts / act_hops_denom)  # Note: since averaging over topk possible for topk 2+ stop acc to be lower than topk=1..
                    else:
                        stop_preds_np = np.array([])
                        stop_target = np.array([])
                        stop_accuracy_per_sample = -1.0
                    
                    type_ = batch_ann[idx]["type"]
                    if type_.strip() == '':
                        type_ = batch_ann[idx]["src"] # was 'single hop'

                    if len(sp) > 0:  # if para-level annotationns exist ie not nq or tqa
                        p_recall, p_em = 0, 0
                        sp_covered = [sp_title in retrieved_titles for sp_title in sp]  # [len(sp)]
                        if np.sum(sp_covered) > 0:
                            p_recall = 1  # Any gold para is in k * max_hops retrieved titles
                        if np.sum(sp_covered) == len(sp_covered):  #works for variable # of hops
                            p_em = 1      #if len(sp)=2 both retrieved para in gold paras, if len(sp)=1, single retrived para in gold paras
                        path_covered = [int(set(p) == set(sp)) for p in path_titles]  # for hpqa equiv but wont work for act_hops < max_hops 
                        path_covered = np.sum(path_covered) > 0
                        recall_1 = 0
                        covered_1 = [sp_title in hop1_titles for sp_title in sp] # 1st retrieved para in gold paras. works for single hop 
                        if np.sum(covered_1) > 0: recall_1 = 1
                    else:           # no para level annotations
                        p_recall = -1
                        p_em = -1
                        recall_1 = -1
                        path_covered = -1
                        
                    metrics.append({
                                    "question": batch_ann[idx]["question"],
                                    "p_recall": p_recall,
                                    "p_em": p_em,
                                    "type": type_,
                                    'recall_1': recall_1,
                                    'path_covered': int(path_covered),
                                    'stop_acc': stop_accuracy_per_sample,
                                    'ans_recall': ans_recall
                                    })


                    # saving when there's no annotations
                    candidate_chains = []
                    for cpath in paths:
                        chain = [id2doc[cpath[i]] for i in range(len(cpath))]
                        candidate_chains.append(chain) 
#                        candidate_chains.append([id2doc[cpath[0]], id2doc[cpath[1]]])  
                    
                    retrieval_outputs.append({
                        "_id": batch_ann[idx]["_id"],
                        "type": type_,
                        "question": batch_ann[idx]["question"],
                        "candidate_chains": candidate_chains,
                        "gold_sp": batch_ann[idx]["sp"],
                        "answer": batch_ann[idx]["answer"],
                        "stop_target": stop_target.tolist(),
                        "stop_preds": stop_preds_np.tolist()
                        # "coverd_k": covered_k
                    })

    with open(os.path.join(args.output_dir, 'predicted_paras_val_test.jsonl'), "w") as out:
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
        logger.info(f'\tAvg PR: {np.mean( return_filtered_list([m["p_recall"] for m in metrics]) )}')
        logger.info(f'\tAvg P-EM: {np.mean( return_filtered_list([m["p_em"] for m in metrics]) )}')
        logger.info(f'\tAvg 1-Recall: {np.mean( return_filtered_list([m["recall_1"] for m in metrics]) )}')
        logger.info(f'\tPath Recall: {np.mean( return_filtered_list([m["path_covered"] for m in metrics]) )}')
        logger.info(f'\tAns Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        if args.eval_stop:
            logger.info(f'\tStop Acc: {np.mean([m["stop_acc"] for m in metrics])}')
        for t in type2items.keys():
            logger.info(f"{t} Questions num: {len(type2items[t])}")
            logger.info(f'\tAvg PR: {np.mean( return_filtered_list([m["p_recall"] for m in type2items[t]]) )}')
            logger.info(f'\tAvg P-EM: {np.mean( return_filtered_list([m["p_em"] for m in type2items[t]]) )}')
            logger.info(f'\tAvg 1-Recall: {np.mean( return_filtered_list([m["recall_1"] for m in type2items[t]]) )}')
            logger.info(f'\tPath Recall: {np.mean( return_filtered_list([m["path_covered"] for m in type2items[t]]) )}')
            logger.info(f'\tAns Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
            if args.eval_stop:
                logger.info(f'\tStop Acc: {np.mean([m["stop_acc"] for m in type2items[t]])}')





