# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
Description: encode text corpus into a store of dense vectors. 

Usage (adjust the batch size according to your GPU memory):

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file ${CORPUS_PATH} \
    --init_checkpoint ${MODEL_CHECKPOINT} \
    --embed_save_path ${SAVE_PATH} \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20 

"""

import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader

from mdr.retrieval.data.encode_datasets import EmDataset, em_collate
from mdr.retrieval.models.mhop_retriever import RobertaCtxEncoder
from mdr_config import encode_args
from utils import move_to_cuda, load_saved

def main():
    args = encode_args()
#    if args.fp16:
#        import apex
#        apex.amp.register_half_function(torch, 'einsum')

    if os.path.exists(args.embed_save_path) and os.listdir(args.embed_save_path):
        print(f"output directory {args.embed_save_path} already exists and is not empty.")
    if not os.path.exists(args.embed_save_path):
        os.makedirs(args.embed_save_path, exist_ok=True)


    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.embed_save_path, "encode_log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    if not args.predict_file:
        raise ValueError(
            "`predict_file` must be specified.")

    if args.update_id2doc_only:
        logger.info(f"--update_id2doc_only flag set so corpus id2doc.json file will be updated in {args.embed_save_path} but not embeddings in index.npy.")


    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "roberta" in args.model_name:
        model = RobertaCtxEncoder(bert_config, args)
    else:
        logger.info("Invalid model. Only Roberta is supported.")
        assert False, "Exiting due to invalid model name.."
        #model = CtxEncoder(bert_config, args)

    eval_dataset = EmDataset(tokenizer, args.predict_file, args.max_q_len, args.max_c_len, args.is_query_embed, args.embed_save_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=em_collate, pin_memory=True, num_workers=args.num_workers)
    
    if args.update_id2doc_only:
        logger.info(f"--update_id2doc_only flag set so corpus id2doc.json file updated in {args.embed_save_path} but not embeddings in index.npy. Exiting.")
        return

    assert args.init_checkpoint != ""
    model = load_saved(model, args.init_checkpoint, exact=False)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Begin encoding corpus...")
    with torch.cuda.amp.autocast(enabled=args.fp16):
        embeds = predict(model, eval_dataloader)
        
    logger.info(f"Finished encoding. Embedding dimensions: {embeds.size()}")
    logger.info(f"Saving index.npy to: {args.embed_save_path}")
    np.save(os.path.join(args.embed_save_path, 'index.npy'), embeds.cpu().numpy())  #TJH added os.path.join
    logger.info("Finished encoding corpus!")
    
    

def predict(model, eval_dataloader):
    if type(model) == list:
        model = [m.eval() for m in model]
    else:
        model.eval()

    embed_array = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            results = model(batch_to_feed)
            embed = results['embed'].cpu()
            embed_array.append(embed)

    print(f"Finished encoding batches, now concatenating {len(embed_array)} batches into single tensor..")
    embed_array = torch.cat(embed_array)

    model.train()
    return embed_array


if __name__ == "__main__":
    main()
