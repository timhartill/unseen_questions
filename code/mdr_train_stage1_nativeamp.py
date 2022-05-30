"""
@author: Tim Hartill

Adapted from MDR train_mhop.py and train_momentum.py
Description: train a stage 1 reader from pretrained ELECTRA encoder

Usage: Run from base/scripts dir: 
    mdr_train_stage1_reader_nativeamp.sh
    or
    mdr_eval_stage1_reader_nativeamp.sh

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
args.sent_score_force_zero = False
args.debug = True
args.sp_percent_thresh = 0.55

# for eval only:
args.do_train=False
args.do_predict=True

#05/27/2022 14:12:47 - INFO - __main__ - Step 18000 Train loss 1.90 SP_EM 85.46 on epoch=3
#05/27/2022 14:12:47 - INFO - __main__ - Saving model with best SP_EM 85.33 -> SP_EM 85.46 on epoch=3
args.init_checkpoint = '/large_data/thar011/out/mdr/logs/stage1_test3_hpqa_hover_fever_nosentforcezero_fullevalmetrics-05-26-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt'

args.save_prediction = 'stage1_dev_predictions_TEST.jsonl'
"""

import logging
import os
import json
import collections
import random
import time
import copy
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
from utils import move_to_cuda, load_saved, AverageMeter, saveas_jsonl

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
    logger.info(f"Output dir: {args.output_dir}")

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
        logger.info(f"Loading checkpoint: {args.init_checkpoint}")
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
                        logger.info(f"Starting predict on Batch Step {batch_step}  Global Step {global_step}  Epoch {epoch} ..")
                        metrics = predict(args, model, eval_dataloader, device, logger)
                        main_metric = metrics["sp_em"]  
                        logger.info("Bat Step %d Glob Step %d Train loss %.2f SP_EM %.2f on epoch=%d" % (batch_step, global_step, train_loss_meter.avg, main_metric*100, epoch))

                        if best_main_metric < main_metric:
                            logger.info("Saving model with best SP_EM %.2f -> SP_EM %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                            model = model.to(device)
                            best_main_metric = main_metric

            logger.info(f"End of Epoch {epoch}: Starting predict on Batch Step {batch_step}  Global Step {global_step}  Epoch {epoch} ..")
            metrics = predict(args, model, eval_dataloader, device, logger)
            main_metric = metrics["sp_em"] # originally 'em'
            logger.info("Bat Step %d Glob Step %d Train loss %.2f SP_EM %.2f on epoch=%d" % (batch_step, global_step, train_loss_meter.avg, main_metric*100, epoch))
            for k, v in metrics.items():
                tb_logger.add_scalar(k, v*100, epoch)
            logger.info(f'Saving checkpoint_last.pt end the end of epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_last.pt"))

            if best_main_metric < main_metric:
                logger.info("Saving model with best SP_EM %.2f -> SP_EM %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                best_main_metric = main_metric

        logger.info("Training finished!")

    elif args.do_predict:
        metrics = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"eval performance summary {metrics}")
    elif args.do_test:
        eval_final(args, model, eval_dataloader, weight=0.8)  #TJH NOT UPDATED
    return


def predict(args, model, eval_dataloader, device, logger, 
            sp_thresh=[0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]):
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
        
        Calculates the sp threshold as the highest sp_recall returning less than sp_percent_thresh of the sentences in the para
    """
    sp_percent_thresh = args.sp_percent_thresh # default 0.55
    model.eval()
    id2result = collections.defaultdict(list)  #  inputs / golds
    id2answer = collections.defaultdict(list)   # corresponding predictions
    for batch in tqdm(eval_dataloader):
        #TJH batch = next(iter(eval_dataloader))
        # batch_to_feed = batch["net_inputs"] 
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        batch_labels = batch["net_inputs"]["label"].view(-1).tolist() # list [bs] = 1/0
        batch_sp_labels = batch['net_inputs']['sent_labels'].tolist()
        batch_sp_offsets = batch['net_inputs']['sent_offsets'].tolist()
        with torch.no_grad():
            outputs = model(batch_to_feed)  # dict_keys(['start_logits', 'end_logits', 'rank_score', 'sp_score'])
            scores = outputs["rank_score"]
            scores = scores.sigmoid().view(-1).tolist()  # added .sigmoid()  list [bs] = 0.46923
            sp_scores = outputs["sp_score"]    # [bs, max#sentsinbatch]
            sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)  #mask scores past end of # sents in sample
            batch_sp_scores = sp_scores.sigmoid()  # [bs, max#sentsinbatch]  [0.678, 0.5531, 0.0, 0.0, ...]
            outs = [outputs["start_logits"], outputs["end_logits"]]  # [ [bs, maxseqleninbatch], [bs, maxseqleninbatch] ]

        for idx, (qid, label, sp_labels) in enumerate(zip(batch_qids, batch_labels, batch_sp_labels)):
            # full: 1 = query+para = full path (to neg or pos), 0 = partial path
            # index into eval_dataloader.dataset.data[idx] list
            # sp_gold from sp_gold_single = [ [title1, 0], [title1, 2], ..] restricted to just the para
            # act_hops = number of hops the query + next para cover eg 1=q_only->sp1, 2=q+sp1->sp2  if act_hops=orig num_hops then query + next para is fully evidential
            # sp_num = # sentences in this sample
            sp_num = len([o for o in batch_sp_offsets[idx] if o != 0])
            if sp_num == 0:
                logger.info(f"Warning: Sentence Offsets for idx {idx}  qid {qid} are all zero: {batch_sp_offsets[idx]}. The query may have filled entire sequence len and caused para truncation.")
            golds = {'para_label': label, 'sp_labels': sp_labels, 
                     'full': int(batch['full'][idx]), 'index': batch['index'][idx], 
                     'gold_answer': batch['gold_answer'][idx], 'sp_gold': batch['sp_gold'][idx],
                     'act_hops': int(batch['act_hops'][idx]), 'sp_num': sp_num}
            id2result[qid] = golds                    #.append( (label, score) )   #[ (para label, para score) ] - originally appended pos + 5 negs...

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
        start_position_ = list(np.array(start_position.tolist()) - np.array(para_offset))  #para masking adjusted to start after base question so can predict span in query sents
        end_position_ = list(np.array(end_position.tolist()) - np.array(para_offset)) 

        for idx, qid in enumerate(batch_qids):                          
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
            sp_score = batch_sp_scores[idx].tolist()
            passage =  batch["passages"][idx][0]
            #sent_offset = batch['net_inputs']['sent_offsets'][idx].tolist()
            pred_sp_dict = {}
            for thresh in sp_thresh:
                pred_sp = []
                for sent_idx, sent_score in enumerate(sp_score):
                    if sent_score >= thresh and sent_idx < id2result[qid]['sp_num']:  
                        pred_sp.append([passage["title"], sent_idx])
                if pred_sp == []:
                    pred_sp = [[]]
                pred_sp_dict[thresh] = pred_sp

            id2answer[qid] = {
                "rank_score": rank_score,       # para evidentiality score
                "para_pred": para_pred,         # 0/1 decision on whether para is evidential
                "pred_str": pred_str.strip(),   # predicted answer span string
                "span_score": span_score,       # answer confidence score 
                "pred_sp_dict": pred_sp_dict,   # {threshold val: predicted sentences [ [title1, 0], [title1, 2], ..] }
                "pred_sp_scores": sp_score      # evidentiality score of each sentence marker
            }

    
    #Calc best sp threshold, copy corresponding pred_sp at thresh -> pred sp  
    num_results = len(id2result)
    sp_metrics = {}
    for thresh in sp_thresh:
        metrics = {'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0, 'sp_percent': 0.0}
        for qid, res in id2result.items():
            ans_res = id2answer[qid]
            em, prec, recall = update_sp(metrics, ans_res['pred_sp_dict'][thresh], res['sp_gold'])
            if res['sp_num'] > 0:
                metrics['sp_percent'] += ( len(ans_res['pred_sp_dict'][thresh]) / res['sp_num'] )
            else:
                metrics['sp_percent'] += 1.0
        metrics['sp_em'] /= num_results
        metrics['sp_f1'] /= num_results
        metrics['sp_prec'] /= num_results
        metrics['sp_recall'] /= num_results
        metrics['sp_percent'] /= num_results
        sp_metrics[thresh] = copy.deepcopy(metrics)
        logger.info(f"sp threshold: {thresh} metrics: {metrics}")


    best_thresh = -1.0
    best_sp_recall = -1.0
    for thresh in sp_thresh:
        if sp_metrics[thresh]['sp_percent'] >= sp_percent_thresh:
            continue
        if sp_metrics[thresh]['sp_recall'] > best_sp_recall:  # take lowest thresh with same recall that is over the sentence % threshold
            best_thresh = thresh
            best_sp_recall = sp_metrics[thresh]['sp_recall']
    if best_thresh == -1:
        logger.info(f"Unable to determine best threshold selecting less than {sp_percent_thresh} of sentences. Setting theshold to 0.5")
        best_thresh = 0.5
        best_sp_recall = sp_metrics[best_thresh]['sp_recall']

    logger.info(f"Determined best sentence score thresh as {best_thresh} yielding mean sp_recall of {best_sp_recall} and selecting mean {sp_metrics[best_thresh]['sp_percent']} of sentences.")        
            
    
    out_list = []
    ems, f1s, sp_ems, sp_f1s, sp_precs, sp_recalls, joint_ems, joint_f1s, para_acc = [], [], [], [], [], [], [], [], []
    for qid, res in id2result.items():
        index = res['index']
        pos = index % 2 == 0
        sample = eval_dataloader.dataset.data[index]
        ans_res = id2answer[qid]
        ans_res['pred_sp'] = ans_res['pred_sp_dict'][best_thresh]  # select the sp pred from best thresh found
        para_acc.append(int(ans_res['para_pred'] == res['para_label']))  # para evidentiality eval (accuracy)
        # answer eval
        ems.append(exact_match_score(ans_res['pred_str'], res['gold_answer'][0])) # not using multi-answer versions of exact match, f1
        f1, prec, recall = f1_score(ans_res['pred_str'], res['gold_answer'][0])
        f1s.append(f1)
        # sentence eval incl para
        metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
        update_sp(metrics, ans_res['pred_sp'], res['sp_gold'])
        sp_ems.append(metrics['sp_em'])
        sp_f1s.append(metrics['sp_f1'])
        sp_precs.append(metrics['sp_prec'])
        sp_recalls.append(metrics['sp_recall'])
        # joint metrics
        joint_prec = prec * metrics['sp_prec']
        joint_recall = recall * metrics['sp_recall']
        if joint_prec + joint_recall > 0:
            joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        else:
            joint_f1 = 0.
        joint_em = ems[-1] * sp_ems[-1]
        joint_ems.append(joint_em)
        joint_f1s.append(joint_f1)
        
        out_sample = {}
        out_sample['question'] = sample['question'] 
        out_sample['context'] = sample['context_processed']['context']  # question + context = full untokenised input not incorprating truncation of q or c
        out_sample['ans'] = res["gold_answer"]
        out_sample['ans_pred'] = ans_res['pred_str']
        out_sample['ans_pred_score'] = ans_res['span_score']
        out_sample['sp'] = res['sp_gold']
        out_sample['sp_pred'] = ans_res['pred_sp']
        out_sample['sp_labels'] = res['sp_labels']
        out_sample['sp_scores'] = ans_res['pred_sp_scores']
        out_sample['para_gold'] = res['para_label']
        out_sample['para_pred'] = ans_res['para_pred']
        out_sample['para_score'] = ans_res['rank_score']
        out_sample['src'] = sample['src']
        out_sample['pos'] = pos
        out_sample['full'] = res['full']
        out_sample['act_hops'] = res['act_hops']
        out_sample['_id'] = qid
        out_sample['answer_em'] = int(ems[-1])
        out_sample['answer_f1'] = f1s[-1]
        out_sample.update(metrics)
        out_sample['joint_em'] = joint_em
        out_sample['joint_f1'] = joint_f1
        out_sample['para_acc'] = para_acc[-1]
        
        out_list.append(out_sample)

    best_joint_f1 = np.mean(joint_f1s)
    best_joint_em = np.mean(joint_ems)
    best_sp_f1 = np.mean(sp_f1s)
    best_sp_em = np.mean(sp_ems)
    best_sp_prec = np.mean(sp_precs)
    best_sp_recall = np.mean(sp_recalls)
    best_f1 = np.mean(f1s)
    best_em = np.mean(ems)
    best_para_acc = np.mean(para_acc)

    logger.info("------------------------------------------------")
    logger.info(f"Metrics over total eval set. n={len(ems)}")
    logger.info(f'answer em: {best_em}')
    logger.info(f'answer f1: {best_f1}')
    logger.info(f'sp em: {best_sp_em}')
    logger.info(f'sp f1: {best_sp_f1}')
    logger.info(f'sp prec: {best_sp_prec}')
    logger.info(f'sp recall: {best_sp_recall}')
    logger.info(f'joint em: {best_joint_em}')
    logger.info(f'joint f1: {best_joint_f1}')
    logger.info(f'para acc: {best_para_acc}')
    
    create_grouped_metrics(logger, out_list, group_key='src')
    create_grouped_metrics(logger, out_list, group_key='pos')
    create_grouped_metrics(logger, out_list, group_key='act_hops')
    create_grouped_metrics(logger, out_list, group_key='full')

    
    if args.save_prediction != "":
        saveas_jsonl(out_list, os.path.join(args.output_dir, args.save_prediction), update=25000)

    model.train()
    return {"em": best_em, "f1": best_f1, "joint_em": best_joint_em, "joint_f1": best_joint_f1, 
            "sp_em": best_sp_em, "sp_f1": best_sp_f1, "sp_prec": best_sp_prec, "sp_recall": best_sp_recall, "para_acc": best_para_acc}


def create_grouped_metrics(logger, sample_list, group_key='src',
                           metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc']):
    """ output metrics by group
    """
    grouped_metrics = {}
    for sample in sample_list:
        if grouped_metrics.get(sample[group_key]) is None:
            grouped_metrics[sample[group_key]] = {}
        for key in metric_keys:
            if grouped_metrics[sample[group_key]].get(key) is None:
                grouped_metrics[sample[group_key]][key] = []
            grouped_metrics[sample[group_key]][key].append( sample[key] )
    logger.info("------------------------------------------------")     
    logger.info(f"Metrics grouped by: {group_key}")
    logger.info("------------------------------------------------")
    for group in grouped_metrics:
        mgroup = grouped_metrics[group]
        logger.info(f"{group_key}: {group}")
        for key in metric_keys:
            n = len(mgroup[key])
            val = np.mean( mgroup[key] ) if n > 0 else -1
            logger.info(f'{key}: {val}  n={n}')
        logger.info("------------------------------------------------")
    return  
    
    


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
