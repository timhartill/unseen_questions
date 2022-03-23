# Portions Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
@author: Tim Hartill

Adapted from MDR train_mhop.py and train_momentum.py
Description: train a multi-hop dense retrieval from pretrained RoBERTa encoder

Usage: Run from base/scripts dir: 
    bash mdr_train_mhop_retriever_varsteps.sh 
    or
    bash mdr_train_mhop_retriever_momentum_varsteps.sh  after updating --init-retriever with the prior trained ckpt to start from

Debug Args to add after running args = train_args()  
args.prefix='TEST'
args.fp16=True
args.do_train=True
args.predict_batch_size=100
args.train_batch_size=5
args.model_name='roberta-base'
args.learning_rate=2e-5
args.train_file='/home/thar011/data/mdr/hotpot/hotpot_train_with_neg_v0.json'
args.predict_file ='/home/thar011/data/mdr/hotpot/hotpot_dev_with_neg_v0.json'
args.seed=16
args.eval_period=-1
args.max_c_len= 300
args.max_q_len=70
args.max_q_sp_len=350
args.shared_encoder=True
args.warmup_ratio=0.1
args.use_var_versions = True
args.output_dir = '/large_data/thar011/out/mdr/logs'

args.gradient_accumulation_steps = 1

args.momentum=True
args.init_retriever = '/large_data/thar011/out/mdr/logs/03-23-2022/varinitialtest_-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt'

args.init_checkpoint='models/q_encoder.pt'
args.init_checkpoint='logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt'
"""
import logging
import os
import random
from datetime import date
from functools import partial

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr.retrieval.config import train_args
from mdr.retrieval.criterions import (mhop_eval, mhop_loss, mhop_loss_var, mhop_eval_var)
from mdr.retrieval.data.mhop_dataset import MhopDataset, mhop_collate, MhopDataset_var, mhop_collate_var
from mdr.retrieval.models.mhop_retriever import RobertaRetriever, RobertaRetriever_var, RobertaMomentumRetriever, RobertaMomentumRetriever_var
from mdr.retrieval.utils.utils import AverageMeter, move_to_cuda, load_saved


def main():
    args = train_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    date_curr = date.today().strftime("%m-%d-%Y")
    if args.momentum:
        model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-m{args.m}-k{args.k}-t{args.temperature}-ga{args.gradient_accumulation_steps}-var{args.use_var_versions}"
    else:    
        model_name = f"{args.prefix}-mom-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-shared{args.shared_encoder}-ga{args.gradient_accumulation_steps}-var{args.use_var_versions}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs","tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
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
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.use_var_versions:
        if args.momentum:
            model = RobertaMomentumRetriever_var(bert_config, args)
        else:
            model = RobertaRetriever_var(bert_config, args)
        eval_dataset = MhopDataset_var(tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
        collate_fc = partial(mhop_collate_var, pad_id=tokenizer.pad_token_id)
    else:
        if args.momentum:
            model = RobertaMomentumRetriever(bert_config, args)
        else:
            model = RobertaRetriever(bert_config, args)
        #model_nv = RobertaRetriever(bert_config, args)
        eval_dataset = MhopDataset(tokenizer, args.predict_file, args.max_q_len, args.max_q_sp_len, args.max_c_len)
        collate_fc = partial(mhop_collate, pad_id=tokenizer.pad_token_id)
        #collate_fc_nv = partial(mhop_collate, pad_id=tokenizer.pad_token_id)

    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    #model_nv.to(device)
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            from apex import amp
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train() #TJH model_nv.train()
        if args.use_var_versions:
            train_dataset = MhopDataset_var(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
            mloss = mhop_loss_var
        else:
            train_dataset = MhopDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
#            train_dataset_nv = MhopDataset(tokenizer, args.train_file, args.max_q_len, args.max_q_sp_len, args.max_c_len, train=True)
            mloss = mhop_loss
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)
#        train_dataloader_nv = DataLoader(train_dataset_nv, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc_nv, num_workers=args.num_workers, shuffle=True)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader):
                #TJH batch = next(iter(train_dataloader))
                #TJH batch_nv = next(iter(train_dataloader_nv))
                #TJH batch_nv = move_to_cuda(batch_nv)
                batch_step += 1
                batch = move_to_cuda(batch)
                #TJH: outputs = model(batch) #intermittent error with amp. To reproduce: run next line (should work), then run this line (fail), then run next line again (also fail)
                #TJH q_embeds = model.encode_q(batch['q_input_ids'][0], batch['q_mask'][0], batch.get("token_type_ids", None)) #intermittent Error with amp, no error without
                #TJH q_embeds_nv = model.encode_q(batch_nv['q_input_ids'], batch_nv['q_mask'], batch_nv.get("token_type_ids", None)) #Error with amp, no error without
                #TJH q_embeds_nv = model_nv.encode_q(batch_nv['q_input_ids'], batch_nv['q_mask'], batch_nv.get("token_type_ids", None))  # works
                loss = mloss(model, batch, args)  #works without amp
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())
            
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  #Note: Using amp get "Detected call of `lr_scheduler.step()` before `optimizer.step()`". Can ignore this. Explanation: if the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient overflow), the optimizer.step() will be skipped and you might get this warning.
                    model.zero_grad()
                    global_step += 1

                    tb_logger.add_scalar('batch_train_loss', loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss', train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        mrrs = predict(args, model, eval_dataloader, device, logger)
                        mrr = mrrs["mrr_avg"]
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (global_step, train_loss_meter.avg, mrr*100, epoch))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                            if args.momentum:
                                if n_gpu > 1: # TJH Added based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
                                    # Using DataParallel -> need to call model.module
                                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_best.pt"))
                                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_best.pt"))
                                else:
                                    torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_best.pt"))
                                    torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_best.pt"))
                            else:
                                torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrrs = predict(args, model, eval_dataloader, device, logger)
            mrr = mrrs["mrr_avg"]
            logger.info("Step %d Train loss %.2f MRR-AVG %.2f on epoch=%d" % (global_step, train_loss_meter.avg, mrr*100, epoch))
            for k, v in mrrs.items():
                tb_logger.add_scalar(k, v*100, epoch)
            if args.momentum:
                if n_gpu > 1: # TJH Added based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
                    # Using DataParallel -> need to call model.module
                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_last.pt"))
                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_last.pt"))
                else:
                    torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_last.pt"))
                    torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_last.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_last.pt"))

            if best_mrr < mrr:
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" % (best_mrr*100, mrr*100, epoch))
                if args.momentum:
                    if n_gpu > 1: # TJH Added based on https://github.com/facebookresearch/multihop_dense_retrieval/pull/14/commits/96a0df6620ab02b231a1448373da7f59f615dae1
                        # Using DataParallel -> need to call model.module
                        torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_best.pt"))
                        torch.save(model.module.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_best.pt"))
                    else:
                        torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_q_best.pt"))
                        torch.save(model.encoder_q.state_dict(), os.path.join(args.output_dir, f"checkpoint_k_best.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_best.pt"))
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")

def predict(args, model, eval_dataloader, device, logger):
    if args.use_var_versions:
        eval_func = mhop_eval_var
    else:
        eval_func = mhop_eval
    model.eval()
    #rrs_1, rrs_2 = [], [] # reciprocal rank
    rrs_all = {} # reciprocal rank
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch_to_feed)
            eval_results = eval_func(outputs, args)  # eg rrs={1: [1.0, 0.125, 0.2], 2: [0.125, 0.5], 3: []}
            for hop in eval_results:
                if rrs_all.get(hop) is None:
                    rrs_all[hop] = []
                rrs_all[hop] += eval_results[hop]
#            _rrs_1, _rrs_2 = eval_results["rrs_1"], eval_results["rrs_2"]
#            rrs_1 += _rrs_1
#            rrs_2 += _rrs_2
    mrrs_all = {'mrr_'+str(hop):np.mean(rrs_all[hop]) for hop in rrs_all if len(rrs_all[hop]) > 0}
    mrr_avg = np.mean([mrrs_all[hop] for hop in mrrs_all])
    mrrs_all["mrr_avg"] = mrr_avg
#    mrr_1 = np.mean(rrs_1)
#    mrr_2 = np.mean(rrs_2)
    logger.info(f"evaluated {len(rrs_all[1])} examples...")
    logger.info(f"MRRS: {mrrs_all}...")
#    logger.info(f'MRR-1: {mrr_1}')
#    logger.info(f'MRR-2: {mrr_2}')
    model.train()
#    return {"mrr_1": mrr_1, "mrr_2": mrr_2, "mrr_avg": (mrr_1 + mrr_2) / 2}
    return mrrs_all


if __name__ == "__main__":
    main()
