"""
@author: Tim Hartill

Adapted from MDR train_mhop.py and train_momentum.py
Description: train a stage 2 reader from pretrained ELECTRA encoder

Usage: Run from base/scripts dir: 
    mdr_train_stage2_reader_nativeamp.sh
    or
    mdr_eval_stage2_reader_nativeamp.sh

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
args.sp_percent_thresh = 1.0
args.num_workers_dev = 10
args.ev_combiner = False

# for eval only:
args.do_train=False
args.do_predict=True

args.init_checkpoint = '/large_data/thar011/out/mdr/logs/stage2_test0_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-06-08-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best_BatStep 207999 GlobStep26000 Trainloss2.99 SP_EM64.00 epoch5 para_acc0.8416.pt'

args.save_prediction = 'stage2_dev_predictions_TEST.jsonl'
"""

import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import collections
import random
import time
import copy
from datetime import date
from functools import partial

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr_config import train_args
from reader.reader_dataset import Stage2Dataset, stage_collate, AlternateSampler
from reader.reader_model import StageModel, SpanAnswerer

from reader.hotpot_evaluate_v1 import f1_score, exact_match_score, update_sp
from mdr_basic_tokenizer_and_utils import get_final_text
from utils import move_to_cuda, load_saved, AverageMeter, saveas_jsonl, create_grouped_metrics

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']

def main():
    args = train_args()

    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{date_curr}-rstage2-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-ga{args.gradient_accumulation_steps}"
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
    
    model = StageModel(bert_config, args)  
    eval_dataset = Stage2Dataset(args, tokenizer, args.predict_file, train=False)
    collate_fc = partial(stage_collate, pad_id=tokenizer.pad_token_id)  

    # turned off num_workers for eval after too many open files error
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, 
                                 pin_memory=True, num_workers=args.num_workers_dev)
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
        train_dataset = Stage2Dataset(args, tokenizer, args.train_file, train=True)
        train_sampler = AlternateSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, 
                                      collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
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
                        main_metric = metrics["para_acc"]  
                        logger.info("Bat Step %d Glob Step %d Train loss %.2f para_acc %.2f on epoch=%d" % (batch_step, global_step, train_loss_meter.avg, main_metric*100, epoch))

                        if best_main_metric < main_metric:
                            logger.info("Saving model with best para_acc %.2f -> para_acc %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                            model = model.to(device)
                            best_main_metric = main_metric

            logger.info(f"End of Epoch {epoch}: Starting predict on Batch Step {batch_step}  Global Step {global_step}  Epoch {epoch} ..")
            metrics = predict(args, model, eval_dataloader, device, logger)
            main_metric = metrics["para_acc"] # originally 'em'
            logger.info("Bat Step %d Glob Step %d Train loss %.2f para_acc %.2f on epoch=%d" % (batch_step, global_step, train_loss_meter.avg, main_metric*100, epoch))
            for k, v in metrics.items():
                tb_logger.add_scalar(k, v*100, epoch)
            logger.info(f'Saving checkpoint_last.pt end the end of epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_last.pt"))

            if best_main_metric < main_metric:
                logger.info("Saving model with best para_acc %.2f -> para_acc %.2f on epoch=%d" % (best_main_metric*100, main_metric*100, epoch))
                torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                best_main_metric = main_metric

        logger.info("Training finished!")

    elif args.do_predict:
        metrics = predict(args, model, eval_dataloader, device, logger, use_fixed_thresh=False)
        logger.info(f"eval performance summary {metrics}")
    elif args.do_test:
        print("do_test not implemented..")  #TJH NOT UPDATED
    return


def predict(args, model, eval_dataloader, device, logger, 
            sp_thresh=[0.00001, 0.0001, 0.001, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
            ev_thresh=[0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.475, 0.4875, 0.5, 0.5125, 0.525, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            use_fixed_thresh=True):
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
        sp_thresh and ev_thresh must contain 0.5 or error..
        Calculates the sp threshold as the highest sp_recall returning less than sp_percent_thresh of the sentences in the para
    """
    sp_percent_thresh = args.sp_percent_thresh # default 1.0 in stage 2:  percentage of sentences to reduce to
    model.eval()
    id2result = collections.defaultdict(list)  #  inputs / golds
    id2answer = collections.defaultdict(list)   # corresponding predictions
    for batch in tqdm(eval_dataloader):
        #TJH batch = next(iter(eval_dataloader))
        # batch_to_feed = batch["net_inputs"]
        #batch = copy.deepcopy(batch_orig)
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        batch_labels = batch["net_inputs"]["label"].view(-1).tolist() # list [bs] = 1/0
        batch_sp_labels = batch['net_inputs']['sent_labels'].tolist()
        batch_sp_offsets = batch['net_inputs']['sent_offsets'].tolist()
        with torch.inference_mode():
            outputs = model(batch_to_feed)  # dict_keys(['start_logits', 'end_logits', 'rank_score', 'sp_score'])
            scores = outputs["rank_score"]
            scores = scores.sigmoid().view(-1).tolist()  # added .sigmoid()  list [bs] = 0.46923
            sp_scores = outputs["sp_score"]    # [bs, max#sentsinbatch]
            sp_scores = sp_scores.float().masked_fill(batch_to_feed["sent_offsets"].eq(0), float("-inf")).type_as(sp_scores)  #mask scores past end of # sents in sample
            batch_sp_scores = sp_scores.sigmoid()  # [bs, max#sentsinbatch]  [0.678, 0.5531, 0.0, 0.0, ...]
            outs = [outputs["start_logits"], outputs["end_logits"]]  # [ [bs, maxseqleninbatch], [bs, maxseqleninbatch] ]
            if args.ev_combiner:
                ev_scores = outputs["ev_logits"]

        for idx, (qid, label, sp_labels) in enumerate(zip(batch_qids, batch_labels, batch_sp_labels)):
            # full: 1 = query+para = full path (to neg or pos), 0 = partial path
            # index into eval_dataloader.dataset.data[idx] list
            # sp_gold from sp_gold_single = [ possentidx1, possentidx2, ...] s2: [idxs of pos sents]  s1: [ [title, possentidx1] , ...]
            # act_hops = number of hops the query + next para cover eg 1=q_only->sp1, 2=q+sp1->sp2  if act_hops=orig num_hops then query + next para is fully evidential
            # sp_num = # sentences in this sample
            sp_num = len([o for o in batch_sp_offsets[idx] if o != 0])
            if sp_num == 0:
                logger.info(f"Warning: Sentence Offsets for idx {idx}  qid {qid} are all zero: {batch_sp_offsets[idx]}. The query may have filled entire sequence len and caused para truncation.")
            golds = {'para_label': label, 'sp_labels': sp_labels, 
                     'full': int(batch['full'][idx]), 'index': batch['index'][idx], 
                     'gold_answer': batch['gold_answer'][idx], 'sp_gold': batch['sp_gold'][idx],
                     'act_hops': int(batch['act_hops'][idx]), 'sp_num': sp_num, 
                     'question': batch['question'][idx], 'context': batch['context'][idx]}
            id2result[qid] = golds                    #.append( (label, score) )   #[ (para label, para score) ] - originally appended pos + 5 negs...

        span_answerer = SpanAnswerer(batch, outs, batch['para_offsets'], batch['net_inputs']['insuff_offset'].tolist(), args.max_ans_len)

        for idx, qid in enumerate(batch_qids):

            # get the context full evidentiality accuracy at different ev score thresholds
            rank_score = scores[idx]
            para_pred_dict = {}
            for thresh in ev_thresh:
                if rank_score >= thresh:
                    para_pred = 1
                else:
                    para_pred = 0
                para_pred_dict[thresh] = para_pred
                
            if args.ev_combiner:
                ev_score = ev_scores[idx]
                ev_pred = int(ev_score.argmax())
                ev_score = ev_score.tolist()
            else:
                ev_score = [0.0, 0.0]
                ev_pred = -1                

            # get the positive sp sentences at difft sp score thresholds {thresh: [ sentidx1, sentidx4, ..]}
            sp_score = batch_sp_scores[idx].tolist()
            pred_sp_dict = {}
            for thresh in sp_thresh:
                pred_sp = []
                for sent_idx, sent_score in enumerate(sp_score):
                    if sent_score >= thresh and sent_idx < id2result[qid]['sp_num']:  # sp_scores past max sent offset already forced to zero above so probably extraneous
                        pred_sp.append(sent_idx)
                pred_sp_dict[thresh] = pred_sp

            id2answer[qid] = {
                "rank_score": rank_score,               # context evidentiality score
                "para_pred_dict": para_pred_dict,       # 0/1 decision on whether context is fully evidential  {thresh: 1/0}
                "pred_str": span_answerer.pred_strs[idx],             # predicted answer span string
                "span_score": span_answerer.span_scores[idx],         # answer "confidence" score 
                "insuff_score": span_answerer.insuff_scores[idx],     # insuff/[unused0] "confidence" score
                "pred_sp_dict": pred_sp_dict,           # {threshold val: predicted sentences [ sentidx1, sentidx4, ..] }
                "pred_sp_scores": sp_score,             # evidentiality score of each sentence marker
                "ev_pred": ev_pred,                     # 0/1 decision on fully evidential using evidence combiner head
                "ev_scores": ev_score                   # the raw ev logits [no-score, yes-score]
            }


    num_results = len(id2result) 
    #calc best ev threshold, copy corresponding pred -> para_pred
    ev_metrics = {}
    for thresh in ev_thresh:
        ev_acc = 0.0
        for qid, res in id2result.items():
            ans_res = id2answer[qid]
            ev_acc += int(ans_res['para_pred_dict'][thresh] == res['para_label'])  # context evidentiality eval (accuracy)
        ev_acc /= num_results
        ev_metrics[thresh] = ev_acc
        logger.info(f"context evidence threshold: {thresh} acc: {ev_acc}")

    best_ev_thresh = -1.0
    best_ev_acc = -1.0
    for thresh in ev_thresh:
        if ev_metrics[thresh] > best_ev_acc:  # take lowest thresh with same acc
            best_ev_thresh = thresh
            best_ev_acc = ev_metrics[thresh]
    logger.info(f"Determined best context evidence score thresh as {best_ev_thresh} yielding mean para_acc of {best_ev_acc}.")
    if use_fixed_thresh:  #during training use fixed threshold 0.5 and determine best ckpt based on para_acc
        best_ev_thresh = 0.5
        best_ev_acc = ev_metrics[best_ev_thresh]
        logger.info(f"Using fixed context evidence threshold: {best_ev_thresh} with para_acc: {best_ev_acc}")

        
    #Calc best sp threshold, copy corresponding pred_sp at thresh -> pred sp
    sp_metrics = {}
    for thresh in sp_thresh:
        metrics = {'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0, 'sp_percent': 0.0}
        for qid, res in id2result.items():
            ans_res = id2answer[qid]
            sp_gold = [str(s) for s in res['sp_gold']] if res['sp_gold'] != [] else [''] # update_sp dislikes ints
            sp_pred = [str(s) for s in ans_res['pred_sp_dict'][thresh]] if ans_res['pred_sp_dict'][thresh] != [] else ['']
            em, prec, recall = update_sp(metrics, sp_pred, sp_gold)
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
    if use_fixed_thresh:  #during training use fixed threshold 0.5 and determine best ckpt base on para_acc
        best_thresh = 0.5
        best_sp_recall = sp_metrics[best_thresh]['sp_recall']
        logger.info(f"Using fixed sp threshold: {best_thresh} with sp_recall: {best_sp_recall}")
    
    out_list = []
    ems, f1s, sp_ems, sp_f1s, sp_precs, sp_recalls, joint_ems, joint_f1s, para_acc, ev_acc = [], [], [], [], [], [], [], [], [], []
    for qid, res in id2result.items():
        index = res['index']
        pos = res['para_label'] == 1  #was index % 2 == 0 in stage 1
        sample = eval_dataloader.dataset.data[index]  # retrieve some extra info for output file
        ans_res = id2answer[qid]
        ans_res['pred_sp'] = ans_res['pred_sp_dict'][best_thresh]  # select the sp pred from best sp thresh found
        ans_res['para_pred'] = ans_res['para_pred_dict'][best_ev_thresh]  # select the ev pred from best ev thresh found
        para_acc.append(int(ans_res['para_pred'] == res['para_label']))  # context evidentiality eval (accuracy)
        ev_acc.append(int(ans_res['ev_pred'] == res['para_label']))  # context evidentiality eval (accuracy) using evidence combiner head
        # answer eval
        ems.append(exact_match_score(ans_res['pred_str'], res['gold_answer'][0])) # not using multi-answer versions of exact match, f1
        f1, prec, recall = f1_score(ans_res['pred_str'], res['gold_answer'][0])
        f1s.append(f1)
        # sentence eval incl para
        metrics = {'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0}
        sp_gold = [str(s) for s in res['sp_gold']] if res['sp_gold'] != [] else [''] # update_sp dislikes ints
        sp_pred = [str(s) for s in ans_res['pred_sp']] if ans_res['pred_sp'] != [] else ['']
        update_sp(metrics, sp_pred, sp_gold)
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
        out_sample['question'] = res['question'] 
        out_sample['context'] = res['context']  # question + context = full untokenised input not incorporating truncation of q or c
        out_sample['ans'] = res["gold_answer"]
        out_sample['ans_pred'] = ans_res['pred_str']
        out_sample['ans_pred_score'] = ans_res['span_score']
        out_sample['ans_insuff_score'] = ans_res['insuff_score']
        out_sample['sp'] = res['sp_gold']
        out_sample['sp_pred'] = ans_res['pred_sp']
        out_sample['sp_labels'] = res['sp_labels']
        out_sample['sp_scores'] = ans_res['pred_sp_scores']
        out_sample['sp_thresh'] = best_thresh
        out_sample['sp_pred_dict'] = ans_res['pred_sp_dict']
        out_sample['para_gold'] = res['para_label']
        out_sample['para_pred'] = ans_res['para_pred']
        out_sample['para_score'] = ans_res['rank_score']
        out_sample['para_thresh'] = best_ev_thresh
        out_sample['para_pred_dict'] = ans_res['para_pred_dict']
        out_sample['ev_pred'] = ans_res['ev_pred']
        out_sample['ev_scores'] = ans_res['ev_scores']
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
        out_sample['ev_acc'] = ev_acc[-1]
        
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
    if args.ev_combiner:
        best_ev_acc = np.mean(ev_acc)
    else:
        best_ev_acc = -1.0

    logger.info("------------------------------------------------")
    logger.info(f"Metrics over total eval set. n={len(ems)}")
    logger.info(f'answer em: {best_em}')
    logger.info(f'answer f1: {best_f1}')
    logger.info(f'At sp threshold {best_thresh}:')
    logger.info(f'sp em: {best_sp_em}')
    logger.info(f'sp f1: {best_sp_f1}')
    logger.info(f'sp prec: {best_sp_prec}')
    logger.info(f'sp recall: {best_sp_recall}')
    logger.info(f'joint em: {best_joint_em}')
    logger.info(f'joint f1: {best_joint_f1}')
    logger.info(f'At ev threshold {best_ev_thresh}:')
    logger.info(f'para acc: {best_para_acc}')
    if args.ev_combiner:
        logger.info(f'ev combiner acc: {best_ev_acc}')
    
    create_grouped_metrics(logger, out_list, group_key='src', metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc'])
    create_grouped_metrics(logger, out_list, group_key='pos', metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc'])
    #create_grouped_metrics(logger, out_list, group_key='act_hops', metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc'])
    #create_grouped_metrics(logger, out_list, group_key='full', metric_keys = ['answer_em', 'answer_f1', 'sp_em', 'sp_f1', 'sp_prec', 'sp_recall', 'joint_em', 'joint_f1', 'para_acc'])

    
    if args.save_prediction != "":
        saveas_jsonl(out_list, os.path.join(args.output_dir, args.save_prediction), update=25000)

    model.train()
    return {"em": best_em, "f1": best_f1, "joint_em": best_joint_em, "joint_f1": best_joint_f1, 
            "sp_em": best_sp_em, "sp_f1": best_sp_f1, "sp_prec": best_sp_prec, "sp_recall": best_sp_recall, 
            "para_acc": best_para_acc, "ev_acc": best_ev_acc}


  

if __name__ == "__main__":
    main()
