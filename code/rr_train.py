"""
@author: Tim Hartill

Description: train a rationale reranker model from pretrained ELECTRA encoder

Usage: Run from base/scripts dir: 
    rr_train_vXX.sh
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
args.train_file='/home/thar011/data/rationale_reranker/rr_train.jsonl'
args.predict_file ='/home/thar011/data/rationale_reranker/rr_dev.jsonl'
args.seed=42
args.eval_period=250
args.max_c_len= 512
args.max_q_len=70
args.warmup_ratio=0.1
args.output_dir = '/large_data/thar011/out/mdr/logs'
args.gradient_accumulation_steps = 1
args.use_adam=True
args.debug = True
args.num_workers_dev = 10

# for eval only:
args.do_train=False
args.do_predict=True

args.init_checkpoint = '/large_data/thar011/out/mdr/logs/stage2_test0_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-06-08-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best_BatStep 207999 GlobStep26000 Trainloss2.99 SP_EM64.00 epoch5 para_acc0.8416.pt'

args.save_prediction = 'stage2_dev_predictions_TEST.jsonl'
"""

import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import collections
import random
from datetime import date
from functools import partial

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs
from tqdm import tqdm
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from mdr_config import train_args
from rr_model_dataset import RRDataset, batch_collate, AlternateSampler
from rr_model_dataset import RRModel

from utils import move_to_cuda, load_saved, AverageMeter, saveas_jsonl, create_grouped_metrics

ADDITIONAL_SPECIAL_TOKENS = ['[unused0]', '[unused1]', '[unused2]', '[unused3]'] # Actually unused. Only using: [CLS] query [SEP] Rationale [SEP]

def main():
    args = train_args()

    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-{date_curr}-RR-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-ga{args.gradient_accumulation_steps}-nopair{args.no_pos_neg_pairing}-singlepos{args.single_pos_samples}-mcstrip{args.mc_strip_prob}"
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS)
    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError( "Cannot use sequence length %d because the model was only trained up to sequence length %d" % (args.max_c_len, bert_config.max_position_embeddings))
    
    model = RRModel(bert_config, args)  
    eval_dataset = RRDataset(args, tokenizer, args.predict_file, train=False)
    collate_fc = partial(batch_collate, pad_id=tokenizer.pad_token_id)  

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
        train_dataset = RRDataset(args, tokenizer, args.train_file, train=True)
        if args.no_pos_neg_pairing:
            logger.info("Using Random Sampling Strategy.")
            train_sampler = RandomSampler(train_dataset)
        else:
            logger.info("Using Paired Sampling Strategy.")
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
    else:
        print("arg not implemented. Use do_train or do_predict.")  #TJH NOT UPDATED
    return


def predict(args, model, eval_dataloader, device, logger, 
            ev_thresh=[0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.475, 0.4875, 0.5, 0.5125, 0.525, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            use_fixed_thresh=True):
    """      model returns {
            'rank_score': rank_score,       # [bs,1] is rationale evidential 0<->1
            }
    
        batch.keys(): dict_keys(['net_inputs', 'question', 'context', 'gold_answer', 'qids', 'para_offsets', 'index'])
        batch['net_inputs'].keys(): dict_keys(['input_ids', 'attention_mask', 'label', 'token_type_ids'])
        ev_thresh must contain 0.5 or error..
    """
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
        with torch.inference_mode():
            outputs = model(batch_to_feed)  # dict_keys(['rank_score'])
            scores = outputs["rank_score"]
            scores = scores.sigmoid().view(-1).tolist()  

        for idx, (qid, label) in enumerate(zip(batch_qids, batch_labels)):
            # index = index into eval_dataloader.dataset.data[idx] list
            golds = {'para_label': label,
                     'index': batch['index'][idx], 
                     'gold_answer': batch['gold_answer'][idx],
                     'question': batch['question'][idx], 'context': batch['context'][idx]}
            id2result[qid] = golds                    #.append( (label, score) )   #[ (para label, para score) ] - originally appended pos + 5 negs...

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
                
            id2answer[qid] = {
                "rank_score": rank_score,               # context evidentiality raw score
                "para_pred_dict": para_pred_dict,       # 0/1 decision on whether rationale is fully evidential at each threshold  {thresh: 1/0}
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

    
    out_list = []
    para_acc = []
    for qid, res in id2result.items():
        index = res['index']
        pos = res['para_label'] == 1  # sample with positive rationale = 1, neg = 0 
        sample = eval_dataloader.dataset.data[index]  # retrieve some extra info for output file
        ans_res = id2answer[qid]
        ans_res['para_pred'] = ans_res['para_pred_dict'][best_ev_thresh]  # select the ev pred from best ev thresh found or 0.5 if training
        para_acc.append(int(ans_res['para_pred'] == res['para_label']))  # rationale evidentiality eval (accuracy)
        
        out_sample = {}
        out_sample['question'] = res['question'] 
        out_sample['context'] = res['context']  # question + context = full untokenised input not incorporating truncation of q or c
        out_sample['ans'] = res["gold_answer"]
        out_sample['para_gold'] = res['para_label']
        out_sample['para_pred'] = ans_res['para_pred']            # 0/1 decision at best thresh (or 0.5 if training)
        out_sample['para_score'] = ans_res['rank_score']          # raw score
        out_sample['para_thresh'] = best_ev_thresh
        out_sample['para_pred_dict'] = ans_res['para_pred_dict']  # 0/1 decision at each threshold
        out_sample['src'] = sample['src']
        out_sample['pos'] = pos
        out_sample['_id'] = qid
        out_sample['para_acc'] = para_acc[-1]       
        out_list.append(out_sample)

    best_para_acc = np.mean(para_acc)

    logger.info("------------------------------------------------")
    logger.info(f"Metrics over total eval set. n={len(para_acc)}")
    logger.info(f'At ev threshold {best_ev_thresh}:')
    logger.info(f'para acc: {best_para_acc}')
    
    create_grouped_metrics(logger, out_list, group_key='src', metric_keys = ['para_acc'])
    create_grouped_metrics(logger, out_list, group_key='pos', metric_keys = ['para_acc'])
    
    if args.save_prediction != "":
        saveas_jsonl(out_list, os.path.join(args.output_dir, args.save_prediction), update=25000)

    model.train()
    return {"para_acc": best_para_acc}


  

if __name__ == "__main__":
    main()
