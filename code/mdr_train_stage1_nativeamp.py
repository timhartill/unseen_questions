"""
@author: Tim Hartill

Adapted from MDR train_mhop.py and train_momentum.py
Description: train a stage 1 reader from pretrained ELECTRA encoder

Usage: Run from base/scripts dir: 
    bash mdr_train_mhop_retriever_varsteps.sh 
    or
    bash mdr_train_mhop_retriever_momentum_varsteps.sh  after updating --init-retriever with the prior trained ckpt to start from

Debug Args to add after running args = train_args()  
args.prefix='TEST'
args.fp16=True
args.do_train=True
args.predict_batch_size=100
args.train_batch_size=12
args.model_name='google/electra-large-discriminator'
args.learning_rate=5e-5
args.train_file='/home/thar011/data/sentences/sent_train.jsonl'
args.predict_file ='/home/thar011/data/sentences/sent_dev.jsonl'
args.seed=42
args.eval_period=250
args.max_c_len= 512
args.max_q_len=70
args.warmup_ratio=0.1
args.output_dir = '/large_data/thar011/out/mdr/logs'
args.gradient_accumulation_steps = 1
args.use_adam=True
args.sp_weight = 1.0
args.sent_score_force_zero = True
args.debug = True

"""

import logging
import os
import json
import collections
import random
import time
from datetime import date
from functools import partial

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr_config import train_args
from reader.reader_dataset import Stage1Dataset, stage1_collate, AlternateSampler
from reader.reader_model import Stage1Model

from reader.hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
from mdr_basic_tokenizer_and_utils import get_final_text
from utils import move_to_cuda, load_saved, AverageMeter

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']

def main():
    args = train_args()

    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{date_curr}-rstage1-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-ga{args.gradient_accumulation_steps}"
    args.output_dir = os.path.join(args.output_dir, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(args.output_dir))

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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    #args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError( "Cannot use sequence length %d because the model was only trained up to sequence length %d" % (args.max_c_len, bert_config.max_position_embeddings))
    
    model = Stage1Model(bert_config, args)
    eval_dataset = Stage1Dataset(args, tokenizer, args.predict_file, train=False)
    collate_fc = partial(stage1_collate, pad_id=tokenizer.pad_token_id)

    # turned off num_workers for eval after too many open files error
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True)  #, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")
    #TJH batch = next(iter(eval_dataloader))

    if args.init_checkpoint != "":
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.use_adam:
            optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_main_metric = 0
        train_loss_meter = AverageMeter()
        model.train() #TJH model_nv.train()
        train_dataset = Stage1Dataset(args, tokenizer, args.train_file, train=True)
        train_sampler = AlternateSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
        #train_dataloader = DataLoader(train_dataset, pin_memory=True, collate_fn=collate_fc, num_workers=0, batch_sampler=torch.utils.data.BatchSampler(train_sampler, batch_size=args.train_batch_size, drop_last=False))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        #nan_max_log = 2
        
        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):
            #nan_count = 0
            model.debug_count = 3
            logger.info(f"Starting epoch {epoch}..")
            for batch in tqdm(train_dataloader):
                #TJH batch = next(iter(train_dataloader))
                #if batch_step == 611:
                #    logger.info("ERROR LOG Outputting batch 611:")
                #    logger.info(f"INDEX:{batch['index']}")
                #    logger.info(f"STARTS:{batch['net_inputs']['starts']}")
                #    logger.info(f"ENDS:{batch['net_inputs']['ends']}")
                #    logger.info(f"PARA OFFSETS:{batch['para_offsets']}")
                #    logger.info(f"ANSWERS:{batch['gold_answer']}")
                batch_step += 1
                batch_inputs = move_to_cuda(batch["net_inputs"])
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss = model( batch_inputs )
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                #if args.debug: #and loss.isnan().any():
                #    nan_count += 1
                #    if nan_count < nan_max_log:
                #        logger.info(f"DEBUG Ep:{epoch} bstep:{batch_step} Debug str follows..")
                #        logger.info(f"{outstr}")
                    
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
                train_loss_meter.update(loss.item())
            
                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                    # You may use the same value for max_norm here as you would without gradient scaling.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called, otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    scaler.update() # Updates the scale for next iteration.
                    scheduler.step()  #Note: Using amp get "Detected call of `lr_scheduler.step()` before `optimizer.step()`". Can ignore this. Explanation: if the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient overflow), the optimizer.step() will be skipped and you might get this warning.
                    model.zero_grad() 
                    global_step += 1

                    tb_logger.add_scalar('batch_train_loss', loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss', train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        metrics = predict(args, model, eval_dataloader, device, logger)
                        main_metric = metrics["sp_em"]  #TODO: set this to the right "main" metric originally "em"
                        logger.info("Step %d Train loss %.2f SP_EM %.2f on epoch=%d" % (global_step, train_loss_meter.avg, main_metric*100, epoch))

                        if best_main_metric < main_metric:
                            logger.info("Saving model with best SP_EM %.2f -> EM %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                            model = model.to(device)
                            best_main_metric = main_metric

            metrics = predict(args, model, eval_dataloader, device, logger)
            main_metric = metrics["sp_em"] # originally 'em'
            logger.info("Step %d Train loss %.2f SP_EM %.2f on epoch=%d" % (global_step, train_loss_meter.avg, main_metric*100, epoch))
            #if args.debug:
            #    logger.info(f"Cumulative Loss NaN count:{nan_count}")
            for k, v in metrics.items():
                tb_logger.add_scalar(k, v*100, epoch)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_last.pt"))

            if best_main_metric < main_metric:
                logger.info("Saving model with best SP_EM %.2f -> EM %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                best_main_metric = main_metric

        logger.info("Training finished!")

    elif args.do_predict:
        metrics = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {metrics}")
    elif args.do_test:
        eval_final(args, model, eval_dataloader, weight=0.8)  #TJH NOT UPDATED
    return


def predict(args, model, eval_dataloader, device, logger, fixed_thresh=None):
    """      model returns {
            'start_logits': start_logits,   # [bs, seq_len]
            'end_logits': end_logits,       # [bs, seq_len]
            'rank_score': rank_score,       # [bs,1] is para evidential 0<->1
            "sp_score": sp_score            # [bs, num_sentences] is sentence evidential [0.2,0.1,0.99..]
            }
    
        batch.keys(): dict_keys(['qids', 'passages', 'gold_answer', 'sp_gold', 'para_offsets', 
                                 'net_inputs', 'index', 'doc_tokens', 'tok_to_orig_index', 'wp_tokens', 'full'])
        batch['net_inputs'].keys(): dict_keys(['input_ids', 'attention_mask', 'paragraph_mask', 'label', 
                                               'sent_offsets', 'sent_labels', 'token_type_ids'])
    """
    model.eval()
    id2result = collections.defaultdict(list)
    id2answer = collections.defaultdict(list)
    id2gold = {}
    id2goldsp = {}
    id2fullpartial = {}
    id2index = {}
    for batch in tqdm(eval_dataloader):
        #TJH batch = next(iter(eval_dataloader))
        # batch_to_feed = batch["net_inputs"] 
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        batch_index = batch["index"] # index into eval_dataloader.dataset.data[idx] list
        batch_full = batch["full"]
        batch_labels = batch["net_inputs"]["label"].view(-1).tolist() # list [bs] = 1/0
        with torch.no_grad():
            outputs = model(batch_to_feed)  # dict_keys(['start_logits', 'end_logits', 'rank_score', 'sp_score'])
            scores = outputs["rank_score"]
            scores = scores.sigmoid().view(-1).tolist()  # added .sigmoid()  list [bs] = 0.46923
            sp_scores = outputs["sp_score"]    # [bs, max#sentsinbatch]
            sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)  #mask scores past end of # sents in sample
            batch_sp_scores = sp_scores.sigmoid()  # [bs, max#sentsinbatch]  [0.678, 0.5531, 0.0, 0.0, ...]
            outs = [outputs["start_logits"], outputs["end_logits"]]  # [ [bs, maxseqleninbatch], [bs, maxseqleninbatch] ]

        for qid, index, full, label, score in zip(batch_qids, batch_index, batch_full, batch_labels, scores):
            id2result[qid].append( (label, score) )   #[ (para label, para score) ] - originally appended pos + 5 negs...
            id2fullpartial[qid] = int(full)  # 1 = query+para = full path (to neg or pos), 0 = partial path
            id2index[qid] = index

        # answer prediction
        span_scores = outs[0][:, :, None] + outs[1][:, None]  # [bs, maxseqleninbatch, maxseqleninbatch]
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), args.max_ans_len)          # [maxseqleninbatch, maxseqleninbatch]
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))   # [maxseqleninbatch, maxseqleninbatch]
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)  # [bs, maxseqleninbatch, maxseqleninbatch]
        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]  # [bs]
        end_position = span_scores_masked.max(dim=2)[1].gather(1, start_position.unsqueeze(1)).squeeze(1) # [bs]
        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist() # [bs]
        para_offset = batch['para_offsets']  # [bs]
        start_position_ = list(np.array(start_position.tolist()) - np.array(para_offset))  #para masking adjusted to start after base question
        end_position_ = list(np.array(end_position.tolist()) - np.array(para_offset)) 

        for idx, qid in enumerate(batch_qids):               
            id2gold[qid] = batch["gold_answer"][idx]
            id2goldsp[qid] = batch["sp_gold"][idx]
            
            rank_score = scores[idx]
            if rank_score >= 0.5:
                para_pred = 1
            else:
                para_pred = 0
            start = start_position_[idx]
            end = end_position_[idx]
            span_score = answer_scores[idx]
            
            tok_to_orig_index = batch['tok_to_orig_index'][idx]
            doc_tokens = batch['doc_tokens'][idx]
            wp_tokens = batch['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            pred_str = get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False)

            # get the sp sentences [ [title1, 0], [title1, 2], ..]
            pred_sp = []
            sp_score = batch_sp_scores[idx].tolist()
            passage =  batch["passages"][idx][0]
            #sent_offset = batch['net_inputs']['sent_offsets'][idx].tolist()
            for sent_idx, sent_score in enumerate(sp_score):
                if sent_score >= 0.5:  #TODO configure this threshold
                    pred_sp.append([passage["title"], sent_idx])

#            passages = batch["passages"][idx]  
#            for passage, sent_offset in zip(passages, [0, len(passages[0]["sents"])]):
#                for idx, _ in enumerate(passage["sents"]):
#                    try:
#                        if sp_score[idx + sent_offset] >= 0.5:
#                            pred_sp.append([passage["title"], idx])
#                    except:
                        # logger.info(f"sentence exceeds max lengths")
#                        continue
            id2answer[qid].append({
                "pred_str": pred_str.strip(),
                "rank_score": rank_score,
                "para_pred": para_pred,
                "span_score": span_score,
                "pred_sp": pred_sp
            })
#    acc = []
#    for qid, res in id2result.items():
#        res.sort(key=lambda x: x[1], reverse=True)  # [(para label, para score)]  pointless sort since always 1 element?
#        acc.append(res[0][0] == 1)  # acc of para label derived using "ans_covered" - only relevant where using retrieval results
#    logger.info(f"evaluated {len(id2result)} questions...")
#    logger.info(f'chain ranking em: {np.mean(acc)}')

    best_em, best_f1, best_joint_em, best_joint_f1, best_sp_em, best_sp_f1, best_para_acc = 0, 0, 0, 0, 0, 0, 0
    best_res = None
#    if fixed_thresh:
#        lambdas = [fixed_thresh]
#    else:
#        # selecting threshhold on the dev data
#        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    for lambda_ in lambdas:
    ems, f1s, sp_ems, sp_f1s, joint_ems, joint_f1s, para_acc = [], [], [], [], [], [], []
    results = collections.defaultdict(dict)
    for qid, res in id2result.items():
        full = id2fullpartial[qid]
        index = id2index[qid]
        pos = index % 2 == 0
        sample = eval_dataloader.dataset.data[index]
        src = sample['src']
        num_hops = sample['num_hops']
        ans_res = id2answer[qid]  # now only one ans so sort does nothing..
        #ans_res.sort(key=lambda x: lambda_ * x["rank_score"] + (1 - lambda_) * x["span_score"], reverse=True)
        top_pred = ans_res[0]["pred_str"]
        top_pred_sp = ans_res[0]["pred_sp"]
        top_pred_para = ans_res[0]["para_pred"]

        results["answer"][qid] = top_pred
        results["sp"][qid] = top_pred_sp
        results["para"][qid] = top_pred_para
        
        # para evidentiality eval (accuracy)
        para_acc.append(int(top_pred_para == res[0][0]))

        # answer eval
        ems.append(exact_match_score(top_pred, id2gold[qid][0]))
        f1, prec, recall = f1_score(top_pred, id2gold[qid][0])
        #ems.append(exact_match_score(top_pred, id2gold[qid])) # not using multi-answer versions of exact match, f1
        #f1, prec, recall = f1_score(top_pred, id2gold[qid])            
        f1s.append(f1)

        # sentence eval incl para
        metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
        update_sp(metrics, top_pred_sp, id2goldsp[qid])
        sp_ems.append(metrics['sp_em'])
        sp_f1s.append(metrics['sp_f1'])
        # joint metrics
        joint_prec = prec * metrics["sp_prec"]
        joint_recall = recall * metrics["sp_recall"]
        if joint_prec + joint_recall > 0:
            joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        else:
            joint_f1 = 0.
        joint_em = ems[-1] * sp_ems[-1]
        joint_ems.append(joint_em)
        joint_f1s.append(joint_f1)

    #if best_joint_f1 < np.mean(joint_f1s):
    best_joint_f1 = np.mean(joint_f1s)
    best_joint_em = np.mean(joint_ems)
    best_sp_f1 = np.mean(sp_f1s)
    best_sp_em = np.mean(sp_ems)
    best_f1 = np.mean(f1s)
    best_em = np.mean(ems)
    best_para_acc = np.mean(para_acc)
    best_res = results

    #logger.info(f".......Using combination factor {lambda_}......")
    logger.info("------------------------------------------------")
    logger.info(f'answer em: {np.mean(ems)}, count: {len(ems)}')
    logger.info(f'answer f1: {np.mean(f1s)}, count: {len(f1s)}')
    logger.info(f'sp em: {np.mean(sp_ems)}, count: {len(sp_ems)}')
    logger.info(f'sp f1: {np.mean(sp_f1s)}, count: {len(sp_f1s)}')
    logger.info(f'joint em: {np.mean(joint_ems)}, count: {len(joint_ems)}')
    logger.info(f'joint f1: {np.mean(joint_f1s)}, count: {len(joint_f1s)}')
    logger.info(f'para acc: {np.mean(para_acc)}, count:{len(para_acc)}')
    #logger.info(f"Best joint F1 from combination {best_joint_f1}")
    if args.save_prediction != "":
        json.dump(best_res, open(f"{args.save_prediction}", "w"))

    model.train()
    return {"em": best_em, "f1": best_f1, "joint_em": best_joint_em, "joint_f1": best_joint_f1, 
            "sp_em": best_sp_em, "sp_f1": best_sp_f1, "para_acc": best_para_acc}


#################
# TODO TJH Unmodifed below, won't work ...
########################

def eval_final(args, model, eval_dataloader, weight=0.8, gpu=True):
    """
    for final submission
    """
    model.eval()
    id2answer = collections.defaultdict(list)
    encode_times = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch["net_inputs"]) if gpu else batch["net_inputs"]
        batch_qids = batch["qids"]
        with torch.no_grad():
            start = time.time()
            outputs = model(batch_to_feed)
            encode_times.append(time.time() - start)

            scores = outputs["rank_score"]
            scores = scores.view(-1).tolist()

            if args.sp_pred:
                sp_scores = outputs["sp_score"]
                sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)
                batch_sp_scores = sp_scores.sigmoid()

            # ans_type_predicted = torch.argmax(outputs["ans_type_logits"], dim=1).view(-1).tolist()
            outs = [outputs["start_logits"], outputs["end_logits"]]


        # answer prediction
        span_scores = outs[0][:, :, None] + outs[1][:, None]
        max_seq_len = span_scores.size(1)
        span_mask = np.tril(np.triu(np.ones((max_seq_len, max_seq_len)), 0), args.max_ans_len)
        span_mask = span_scores.data.new(max_seq_len, max_seq_len).copy_(torch.from_numpy(span_mask))
        span_scores_masked = span_scores.float().masked_fill((1 - span_mask[None].expand_as(span_scores)).bool(), -1e10).type_as(span_scores)
        start_position = span_scores_masked.max(dim=2)[0].max(dim=1)[1]
        end_position = span_scores_masked.max(dim=2)[1].gather(1, start_position.unsqueeze(1)).squeeze(1)
        answer_scores = span_scores_masked.max(dim=2)[0].max(dim=1)[0].tolist()
        para_offset = batch['para_offsets']
        start_position_ = list(
            np.array(start_position.tolist()) - np.array(para_offset))
        end_position_ = list(
            np.array(end_position.tolist()) - np.array(para_offset)) 

        for idx, qid in enumerate(batch_qids):
            rank_score = scores[idx]
            start = start_position_[idx]
            end = end_position_[idx]
            span_score = answer_scores[idx]
            tok_to_orig_index = batch['tok_to_orig_index'][idx]
            doc_tokens = batch['doc_tokens'][idx]
            wp_tokens = batch['wp_tokens'][idx]
            orig_doc_start = tok_to_orig_index[start]
            orig_doc_end = tok_to_orig_index[end]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_tokens = wp_tokens[start:end+1]
            tok_text = " ".join(tok_tokens)
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            pred_str = get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False)

            chain_titles = [_["title"] for _ in batch["passages"][idx]]

            # get the sp sentences
            pred_sp = []
            if args.sp_pred:
                sp_score = batch_sp_scores[idx].tolist()
                passages = batch["passages"][idx]
                for passage, sent_offset in zip(passages, [0, len(passages[0]["sents"])]):
                    for idx, _ in enumerate(passage["sents"]):
                        try:
                            if sp_score[idx + sent_offset] > 0.5:
                                pred_sp.append([passage["title"], idx])
                        except:
                            # logger.info(f"sentence exceeds max lengths")
                            continue
            id2answer[qid].append({
                "pred_str": pred_str.strip(),
                "rank_score": rank_score,
                "span_score": span_score,
                "pred_sp": pred_sp,
                "chain_titles": chain_titles
            })
    lambda_ = weight
    results = collections.defaultdict(dict)
    for qid in id2answer.keys():
        ans_res = id2answer[qid]
        ans_res.sort(key=lambda x: lambda_ * x["rank_score"] + (1 - lambda_) * x["span_score"], reverse=True)
        top_pred = ans_res[0]["pred_str"]
        top_pred_sp = ans_res[0]["pred_sp"]

        results["answer"][qid] = top_pred
        results["sp"][qid] = top_pred_sp
        results["titles"][qid] = ans_res[0]["chain_titles"]


    if args.save_prediction != "":
        json.dump(results, open(f"{args.save_prediction}", "w"))

    return results


if __name__ == "__main__":
    main()
