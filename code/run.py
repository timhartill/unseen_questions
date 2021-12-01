""" Functions for:
    Training, 
    Inference,
    Eval Metrics Calculation,
    Sentence Embedding Creation
    Train-Test Similarity Calculation     

Author: Tim Hartill

Portions adapted from https://github.com/allenai/unifiedqa
"""

import os
import numpy as np
from tqdm import tqdm
import datetime    
import json  
import shutil
import pickle

import torch
from transformers import BartTokenizer, BartConfig
from transformers import AutoTokenizer, AutoModelForPreTraining  
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from data import QAData
from unified_data import UnifiedQAData
from bart import MyBart
import eval_metrics  
from overlap_detector import UQADataset
from sentence_embeddings import Embedder, restate_qa_all
from utils import get_parsed_decomp_str, get_parsed_decomp_by_key, load_model

def run(args, logger):
    if args.do_train or args.calc_metrics:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
        if args.is_unifiedqa:
            dev_data = UnifiedQAData(logger, args, args.predict_file, False)
        else:
            dev_data = QAData(logger, args, args.predict_file, False)
    
        if not args.skip_inference:
            dev_data.load_dataset(tokenizer)
            dev_data.load_dataloader()

    if args.do_train:
        if args.is_unifiedqa:
            train_data = UnifiedQAData(logger, args, args.train_file, True)
        else:
            train_data = QAData(logger, args, args.train_file, True)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()
        
        if args.model == "facebook/bart-large":   
            my_model = MyBart
        else:
            my_model = AutoModelForPreTraining  # HF documentation indicates this gives right models for T5 and gpt2 as well as vanilla bart


        if args.checkpoint is not None:
            model = my_model.from_pretrained(args.model,
                                           state_dict=torch.load(args.checkpoint))  
            logger.info("Loading checkpoint from {}".format(args.checkpoint))       
        else:
            model = my_model.from_pretrained(args.model) 
            logger.info("No checkpoint loaded. Training from base pretrained model.")
            
        model.config.to_json_file(os.path.join(args.output_dir, "model_config.json"))  
        logger.info("Saved model config to {}".format(os.path.join(args.output_dir, "model-config.json")))
    
            
        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)
        if args.n_gpu>0:
            model.to(torch.device("cuda"))

        # Added from HF trainer.py
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.num_scheduler_steps)  #TJH added num_scheduler_steps param default 250k, was 100k
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        if args.checkpoint is not None and not os.path.exists(args.checkpoint):
            logger.info(f"Error running Predict: Specified checkpoint doesnt exist: Checkpoint={args.checkpoint}") 
            assert os.path.exists(args.checkpoint), "Exiting. Please remediate and restart."
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        if not os.path.exists(checkpoint):
            checkpoint = None
        logger.info(f"Running Predict. Checkpoint={checkpoint}")    
        tokenizer, model = load_model(model_name=args.model, checkpoint=checkpoint)
        inference_wrapper(tokenizer, model, args, logger, predict_file=args.predict_file)

    if args.do_predict_all:
        if args.checkpoint is not None and not os.path.exists(args.checkpoint):
            logger.info(f"Error running Predict All: Specified checkpoint doesnt exist: Checkpoint={args.checkpoint}") 
            assert os.path.exists(args.checkpoint), "Exiting. Please remediate and restart."
        checkpoint = os.path.join(args.output_dir, 'best-model.pt') if args.checkpoint is None else args.checkpoint
        if not os.path.exists(checkpoint):
            checkpoint = None
        logger.info(f"Running Predict All. Checkpoint={checkpoint}")    
        tokenizer, model = load_model(model_name=args.model, checkpoint=checkpoint)
        uqa_dir = args.predict_file
        if uqa_dir[-4:] == '.tsv':  # eg specified as '/data/thar011/data/unifiedqa/dev.tsv' like uqa train format
            uqa_dir = os.path.split(uqa_dir)[0]  # base uqa directory
        logger.info(f"Base dir: {uqa_dir}")
        ftype='dev'
        for ds in eval_metrics.dev_eval:
            args.prefix = ftype + '_' + ds + '_'
            out_file = os.path.join(args.output_dir, f"{args.prefix}predictions.json")
            if args.add_only_missing and os.path.exists(out_file):
                logger.info(f"Skipping Prediction for {ftype} data of {ds} as prediction file already exists")
                continue
            dspath = os.path.join(uqa_dir, ds, ftype+'.tsv')
            inference_wrapper(tokenizer, model, args, logger, predict_file=dspath)
        ftype='test'
        for ds in eval_metrics.test_eval:
            args.prefix = ftype + '_' + ds + '_'
            out_file = os.path.join(args.output_dir, f"{args.prefix}predictions.json")
            if args.add_only_missing and os.path.exists(out_file):
                logger.info(f"Skipping Prediction for {ftype} data of {ds} as prediction file already exists")
                continue
            dspath = os.path.join(uqa_dir, ds, ftype+'.tsv')
            inference_wrapper(tokenizer, model, args, logger, predict_file=dspath)         
        
    if args.calc_metrics:
        calc_metrics(args, logger, dev_data, predict_file=args.predict_file)

    if args.calc_metrics_all:
        tokenizer = load_model(model_name=args.model, loadwhat='tokenizer_only')
        results_file = os.path.join(args.output_dir, 'eval_metrics.json')
        if os.path.exists(results_file):
            results_dict = json.load(open(results_file))
        else:
            results_dict = {}
        already_calculated = list(results_dict.keys())
        uqa_dir = args.predict_file
        if uqa_dir[-4:] == '.tsv':  # eg specified as '/data/thar011/data/unifiedqa/dev.tsv' like uqa train format
            uqa_dir = os.path.split(uqa_dir)[0]  # base uqa directory
        logger.info(f"Running Calc Metrics All. Base dir: {uqa_dir}")
        ftype='dev'
        for ds in eval_metrics.dev_eval:
            args.prefix = ftype + '_' + ds + '_'
            if args.add_only_missing and ds in already_calculated:
                logger.info(f"Skipping calc metrics for {ftype} data of {ds} as key already exists in eval_metrics.json")
                continue
            dspath = os.path.join(uqa_dir, ds, ftype+'.tsv')
            calc_metrics_wrapper(tokenizer, args, logger, predict_file=dspath)        
        ftype='test'
        for ds in eval_metrics.test_eval:
            args.prefix = ftype + '_' + ds + '_'
            if args.add_only_missing and ds in already_calculated:
                logger.info(f"Skipping calc metrics for {ftype} data of {ds} as key already exists in eval_metrics.json")
                continue
            dspath = os.path.join(uqa_dir, ds, ftype+'.tsv')
            calc_metrics_wrapper(tokenizer, args, logger, predict_file=dspath)        

        
    if args.calc_similarity:
        calc_similarity(args, logger)
        
    if args.calc_similarity_numeric:
        calc_similarity_numeric(args, logger)
        
    if args.create_embeddings:
        create_sentence_embeddings(args, logger)
        
    if args.calc_similarity_embeddings:
        calc_similarity_embeddings(args, logger)
    
    return
        

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False       

    if args.checkpoint_step > 0:
        if (args.checkpoint_step // args.gradient_accumulation_steps) < args.num_scheduler_steps:
            for _ in range(args.checkpoint_step):
                global_step += 1
                if global_step % args.gradient_accumulation_steps == 0:  
                    scheduler.step()
        else:
            logger.info("Number of checkpoint_steps %d / gradient accum steps %d exceeds lr scheduler steps %d so ignoring checkpoint steps.." % (args.checkpoint_step, args.gradient_accumulation_steps, args.num_scheduler_steps))   #TJH Added
            

    def convert_to_single_gpu(state_dict):
        def _convert(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        return {_convert(key):value for key, value in state_dict.items()}

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            if args.verbose and global_step % 100 == 0:
                logger.info("Epoch %d   Global Step %d LR %1.6e" % (epoch, global_step, scheduler.get_last_lr()[0]))   #TJH Added
            global_step += 1
            if (global_step // args.gradient_accumulation_steps) >= args.num_scheduler_steps:
                logger.info("Epoch %d   Global Step %d Number of global steps exceeds lr scheduler steps. lr = 0 so exiting after final inference.." % (epoch, global_step))   #TJH Added
                stop_training = True
                
            batch = [b.to(torch.device("cuda")) for b in batch]
            if args.model != "facebook/bart-large":   # Standard pytorch loss used in hf models ignores -100 values in labels
                batch[2][batch[2]==train_data.tokenizer.pad_token_id] = -100                
            outputs = model(input_ids=batch[0], attention_mask=batch[1],
                         labels=batch[2], decoder_attention_mask=batch[3])  
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]  
            if args.n_gpu > 1:
                loss = loss.mean() 
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            if args.gradient_accumulation_steps > 1:   
                loss = loss / args.gradient_accumulation_steps    
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if (global_step % args.eval_period == 0) or stop_training:
                if args.skip_inference:
                    logger.info("Step %d (epoch %d) Train loss %.2f LR %1.6e" % (
                            global_step,
                            epoch,
                            np.mean(train_losses),
                            scheduler.get_last_lr()[0]))
                    save_config = args.__dict__
                    save_config['curr_train_loss'] = float(np.mean(train_losses))
                    save_config['curr_global_step'] = global_step
                    save_config['curr_lr'] = float(scheduler.get_last_lr()[0])
                    save_config['curr_time'] = str(datetime.datetime.now())
                    train_losses = []
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir,
                                                              "best-model-{}.pt".format(str(global_step).zfill(6))))
                    config_file = os.path.join(args.output_dir, 'best-model-config-{}.json'.format(str(global_step).zfill(6)))
                    with open(config_file, 'w') as f:
                        json.dump(save_config, f)

                else:
                    if args.verbose:
                        logger.info("Step %d Starting inference.." % (global_step)) 
                    model.eval()
                    curr_em, ems = inference(model if args.n_gpu==1 else model.module, dev_data, save_predictions=True, return_details=True)
                    logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d   LR %1.6e" % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_em*100,
                            epoch,
                            scheduler.get_last_lr()[0]))
                    save_config = args.__dict__
                    save_config['curr_train_loss'] = float(np.mean(train_losses))
                    save_config['curr_em'] = float(curr_em*100.0)
                    save_config['curr_global_step'] = global_step
                    save_config['curr_lr'] = float(scheduler.get_last_lr()[0])
                    save_config['curr_time'] = str(datetime.datetime.now())
                    if args.is_unifiedqa:
                        for i, dataset in enumerate(dev_data.unified_dataset):
                            save_config['em_' + dataset] = float(ems[i]*100.0)
                    
                    hist_file = os.path.join(args.output_dir, 'log-train-history.jsonl')
                    with open(hist_file, 'a') as f:
                        json.dump(save_config, f)
                        f.write('\n')
                        
                    train_losses = []

                    # Added Save current-model (always) in addition to best-model to faciliate resumption of training after interruption:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    if args.n_gpu > 1:
                        model_state_dict = convert_to_single_gpu(model_state_dict)
                    torch.save(model_state_dict, os.path.join(args.output_dir, "current-model.pt"))
                    logger.info("Saving current-model with %s: %.2f%% on epoch=%d, global_step=%d, LR=%1.6e" % \
                            (dev_data.metric, curr_em*100.0, epoch, global_step, scheduler.get_last_lr()[0]))
                    config_file = os.path.join(args.output_dir, 'current-model-config.json')
                    with open(config_file, 'w') as f:
                        json.dump(save_config, f)

                    if best_accuracy < curr_em:
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d, LR=%1.6e" % \
                                (dev_data.metric, best_accuracy*100.0, curr_em*100.0, epoch, global_step, scheduler.get_last_lr()[0]))
                        best_accuracy = curr_em
                        wait_step = 0
                        stop_training = False
                        config_file = os.path.join(args.output_dir, 'best-model-config.json')
                        with open(config_file, 'w') as f:
                            json.dump(save_config, f)
    
                    else:
                        wait_step += 1
                        logger.info("No improvement. Number of wait steps: %d of max wait steps: %d" % (wait_step, args.wait_step))
                        if wait_step >= args.wait_step:
                            stop_training = True
                            logger.info("Early Stopping due to no improvement after %d wait steps!" % (wait_step))   #TJH Added
                            break
                    if global_step > 0 and global_step % args.save_best_model_steps == 0:
                        try:
                            src = os.path.join(args.output_dir, "best-model.pt")
                            dest = os.path.join(args.output_dir, f"best-model-{str(global_step)}.pt")
                            logger.info(f"Saving best model after {global_step} steps to {dest} ...")
                            shutil.copyfile(src, dest)
                            src = os.path.join(args.output_dir, "best-model-config.json")
                            dest = os.path.join(args.output_dir, f"best-model-config-{str(global_step)}.json")
                            shutil.copyfile(src, dest)
                        except:
                            logger.info(f"Error saving best model after {global_step} steps. Skipping ...")
                                   
                model.train()
        if stop_training:
            break


def inference(model, dev_data, save_predictions=False, return_details=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    if dev_data.args.verbose:
        dev_data.dataloader = tqdm(dev_data.dataloader)
    for i, batch in enumerate(dev_data.dataloader):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_length=1,  #TJH: was min_lnegth
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    if return_details:
        ems = dev_data.evaluate(predictions)  # if unifiedqa: [dset1 pred mean, dset2 pred mean, ...] else [pred1, pred2, ...]
        return np.mean(ems), ems
    return np.mean(dev_data.evaluate(predictions))


def inference_wrapper(tokenizer, model, args, logger, predict_file):
    """ Run inference for single dataset """
    dev_data = QAData(logger, args, predict_file, False)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()
    ems = inference(model, dev_data, save_predictions=True, return_details=False)
    logger.info("%s: %s on %s data: %.2f" % (args.prefix, dev_data.metric, dev_data.data_type, np.mean(ems)*100))
    results_file = os.path.join(args.output_dir, 'eval_results.csv')  
    dtnow = datetime.datetime.now()
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write('WHEN,TYPE,PREFIX,METRIC,VALUE' + '\n')
    with open(results_file, 'a') as f:
        f.write(f'{dtnow},{dev_data.data_type},{args.prefix},{dev_data.metric},{np.mean(ems)*100}' + '\n')
    return


def calc_metrics_wrapper(tokenizer, args, logger, predict_file):
    """ Calc metrics for single dataset """
    dev_data = QAData(logger, args, predict_file, False)
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()    
    calc_metrics(args, logger, dev_data, predict_file)
    return


def calc_metrics(args, logger, dev_data, predict_file):
    """ Calculate metrics relevant to this dataset
    """
    ds_name = os.path.split(predict_file)[0]
    ds_name = ds_name.split('/')[-1]
    ds_attribs = eval_metrics.dataset_attribs.get(ds_name)
    if ds_attribs is None:
        logger.info(f"Error: Dataset {ds_name} needs to be added to dataset_attribs dict in eval_metrics.py. Exiting.")
        assert ds_attribs is not None, "Exiting.."
    ds_type = ds_attribs['type']
    comp_metrics = eval_metrics.metric_groups[ds_type]['compute']
    pref_metric = eval_metrics.metric_groups[ds_type]['prefer']
    if ds_attribs['prefer'] != '' and ds_attribs['prefer'] is not None:
        pref_metric = ds_attribs['prefer']
        
    predictions = dev_data.load_predictions()
    
    logger.info(f"Loading eval questions and answers for {ds_name} from {predict_file} ...")
    questions = []
    groundtruths = []
    for sample in dev_data.data:
        questions.append(sample['question'])
        groundtruths.append(sample['answer'][0])  #answer is stored as a list
        
    if ds_type == 'DC':
        logger.info(f"Dataset {ds_name} is type {ds_type} so parsing decompositions...")
        gt_decomp = get_parsed_decomp_str(groundtruths)
        pred_decomp = get_parsed_decomp_str(predictions)
        
        
    logger.info("Loading pre-tokenized data from {}".format(dev_data.preprocessed_path))
    with open(dev_data.preprocessed_path, "r") as f:
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
            metadata, word_starts, ners_ids = json.load(f)
        
    output_dict = {'prefer': pref_metric,
                   'type': ds_type,
                   'comp_metrics': comp_metrics,
                   'eval_file_type': dev_data.data_type,
                   'gt_file': predict_file,
                   'gt_file_tokenized': dev_data.preprocessed_path,
                   'groundtruths_tokenized': decoder_input_ids,
                   'groundtruths': groundtruths,
                   'predictions': predictions}
    
    if 'EM' in comp_metrics:
        logger.info("Calculating Exact Match Metric ...")
        scorer = eval_metrics.EM()
        score = scorer.compute_metric(predictions, groundtruths)
        results = {'score': score,
                   'scores': scorer.ems,
                   'newpreds': [],
                   'choices': []}
        output_dict['EM'] = results     
        logger.info(f"EM Accuracy for {ds_name} {dev_data.data_type}: {score}")

    
    if 'F1' in comp_metrics:
        logger.info("Calculating F1 Metric ...")
        scorer = eval_metrics.F1()
        score = scorer.compute_metric(predictions, groundtruths)
        results = {'score': score,
                   'scores': scorer.f1s,
                   'newpreds': [],
                   'choices': []}
        output_dict['F1'] = results        
        logger.info(f"F1 Accuracy for {ds_name} {dev_data.data_type}: {score}")
    
    if 'YN' in comp_metrics:
        logger.info("Calculating YN Accuracy Metric ...")
        scorer = eval_metrics.YN()
        score = scorer.compute_metric(predictions, groundtruths)
        results = {'score': score,
                   'scores': scorer.yn,
                   'newpreds': [],
                   'choices': []}
        output_dict['YN'] = results        
        logger.info(f"YN Accuracy for {ds_name} {dev_data.data_type}: {score}")
    
    if 'SS' in comp_metrics:
        logger.info("Calculating Multichoice Similarity Accuracy Metric ...")
        scorer = eval_metrics.StringSimilarity()
        score = scorer.compute_metric(predictions, groundtruths, questions, usesolver_preproc=False, use_f1=True)
        results = {'score': score,
                   'scores': scorer.simscores,
                   'newpreds': scorer.newpreds,
                   'choices': scorer.choices}
        output_dict['SS'] = results        
        logger.info(f"MC Similarity Accuracy for {ds_name} {dev_data.data_type}: {score}")
    
    if 'RL' in comp_metrics:
        logger.info("Calculating ROUGE-L Metric ...")
        scorer = eval_metrics.Rouge()
        score = scorer.compute_metric(predictions, groundtruths, norm=False)
        results = {'score': score,
                   'scores': [],
                   'newpreds': [],
                   'choices': []}
        output_dict['RL'] = results        
        logger.info(f"Rouge-L for {ds_name} {dev_data.data_type}: {score}")
        
    if 'SARIDA' in comp_metrics:    
        logger.info("Calculating SARI-DA Metric over decomps + decomp answers...")
        gt = get_parsed_decomp_by_key(gt_decomp, 'dalist')
        p = get_parsed_decomp_by_key(pred_decomp, 'dalist')
        scorer = eval_metrics.Sari()
        score = scorer.compute_metric(p, gt, questions)
        results = {'score': score,
                   'scores': scorer.saris,
                   'newpreds': [],
                   'choices': []}
        output_dict['SARIDA'] = results
        logger.info(f"SARIDA Accuracy for {ds_name} {dev_data.data_type}: {score}")

    if 'SARID' in comp_metrics:    
        logger.info("Calculating SARI-D Metric over decomps without decomp answers...")
        gt = get_parsed_decomp_by_key(gt_decomp, 'dlist')
        p = get_parsed_decomp_by_key(pred_decomp, 'dlist')
        scorer = eval_metrics.Sari()
        score = scorer.compute_metric(p, gt, questions)
        results = {'score': score,
                   'scores': scorer.saris,
                   'newpreds': [],
                   'choices': []}
        output_dict['SARID'] = results        
        logger.info(f"SARID Accuracy for {ds_name} {dev_data.data_type}: {score}")        

    if 'F1A' in comp_metrics:    
        logger.info("Calculating F1-A Metric over answers without decomps...")
        gt = get_parsed_decomp_by_key(gt_decomp, 'ans')
        p = get_parsed_decomp_by_key(pred_decomp, 'ans')
        scorer = eval_metrics.F1()
        score = scorer.compute_metric(p, gt)
        results = {'score': score,
                   'scores': scorer.f1s,
                   'newpreds': [],
                   'choices': []}
        output_dict['F1A'] = results        
        logger.info(f"F1A Accuracy for {ds_name} {dev_data.data_type}: {score}") 

    if 'F1DA' in comp_metrics:    
        logger.info("Calculating F1-DA Metric over answers without decomps...")
        gt = get_parsed_decomp_by_key(gt_decomp, 'alist')
        p = get_parsed_decomp_by_key(pred_decomp, 'alist')
        scorer = eval_metrics.F1()
        score = scorer.compute_metric(p, gt)
        results = {'score': score,
                   'scores': scorer.f1s,
                   'newpreds': [],
                   'choices': []}
        output_dict['F1DA'] = results
        logger.info(f"F1DA Accuracy for {ds_name} {dev_data.data_type}: {score}")

        
    results_file = os.path.join(args.output_dir, 'eval_metrics.json')
    if os.path.exists(results_file):
        results_dict = json.load(open(results_file)) 
    else:
        results_dict = {}
    results_dict[ds_name] = output_dict
    with open(results_file, 'w') as f:
        json.dump(results_dict, f)
    logger.info(f"Finished! Evaluation Metrics for {ds_name} {dev_data.data_type} saved into {results_file}")
    return


def calc_similarity(args, logger):
    """ Calculate BOW similarity between each train dataset and the eval datasets with results already in eval_metrics.json
    
    To test:
        run cli.py args stuff
        args.train_file = '/data/thar011/data/unifiedqa/train.tsv'
        args.output_dir = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd'
        args.is_unifiedqa = True
        args.mixture = 'unifiedqa'
        args.answer_thresh = 0.0
        logger = None
        trainset = 'narrativeqa'
        testset = 'drop' 
        manually run relevant lines below..        
    """
    logger.info("Calculating similarity between train and test sets..")
    logger.info(f"Answer threshold: {str(args.answer_thresh)}")
    logger.info(f"Results will be written into file: {os.path.join(args.output_dir, f'eval_test_train_similarities_thresh{str(args.answer_thresh)}.json')}")

    train_data = UnifiedQAData(logger, args, args.train_file, False)
    results_file = os.path.join(args.output_dir, 'eval_metrics.json')
    if os.path.exists(results_file):
        results_dict = json.load(open(results_file)) 
    else:
         logger.info(f"ERROR! Evaluation Metrics File {results_file} not found. Exiting..")
         assert os.path.exists(results_file)

    sim_results = {}
    for trainset in train_data.data.keys():
        logger.info(f"Calculating similarity for {trainset} against:")
        train_reformat = []
        for i in range(len(train_data.data[trainset]['question'])):  #reformat train to be same format as eval datasets below..
            train_reformat.append({'question': train_data.data[trainset]['question'][i],
                                   'answer': train_data.data[trainset]['answer'][i]} )
        sim_results[trainset] = {}    
        for testset in results_dict.keys():
            logger.info(f" ... {testset}")
            gt_file = results_dict[testset]['gt_file']
            dev_data = QAData(logger, args, gt_file, False)
            prefmetric = results_dict[testset]['prefer']
            if prefmetric == 'RL':  # No sample-level results for RL
                prefmetric = 'F1'
            scores = results_dict[testset][prefmetric]['scores']    
            result_score, result_detail = UQADataset().run_similarity_comparer(train_reformat, dev_data.data, 
                                                                               answer_thresh=args.answer_thresh, question_thresh=0.0, 
                                                                               use_cosine='bow')
            sim_results[trainset][testset] = {'sim_scores': result_score, 
                                              'sim_details': result_detail, 
                                              'test_metric': prefmetric, 
                                              'test_scores': scores}

    results_file = os.path.join(args.output_dir, f'eval_test_train_similarities_thresh{str(args.answer_thresh)}.json')
    logger.info(f"Finished calculating test-train similarities. Saving results into {results_file}..")
    with open(results_file, 'w') as f:
        json.dump(sim_results, f)
    logger.info(f"Finished saving results into {results_file}!")
    return


def calc_similarity_numeric(args, logger):
    """ Calculate "F1" similarity between synthetic_textual and synthetic_numeric train datasets vs DROP with results already in eval_metrics.json
    To test:
        run cli.py args stuff
        args.train_file = '/data/thar011/data/unifiedqa/train.tsv'
        args.output_dir = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd'
        args.is_unifiedqa = True
        args.mixture = 'unifiedqa,synthetic_textual,synthetic_numeric'
        logger = None
        trainset = 'synthetic_textual'
        testset = 'drop' 
        manually run relevant lines below..        
    """
    logger.info("Calculating similarity between train synthetic_textual and synthetic_numeric datasets and DROP test set..")
    logger.info(f"Answer threshold: {str(args.answer_thresh)}")
    logger.info(f"Results will be written into file: {os.path.join(args.output_dir, f'eval_test_train_similarities_numeric_thresh{str(args.answer_thresh)}.json')}")
    train_data = UnifiedQAData(logger, args, args.train_file, False)
    results_file = os.path.join(args.output_dir, 'eval_metrics.json')
    if os.path.exists(results_file):
        results_dict = json.load(open(results_file)) 
    else:
         logger.info(f"ERROR! Evaluation Metrics File {results_file} not found. Exiting..")
         assert os.path.exists(results_file)
    results_file_out = os.path.join(args.output_dir, f'eval_test_train_similarities_numeric_thresh{str(args.answer_thresh)}.json')
    logger.info(f"Output will be saved into {results_file_out}")

    testset = 'drop'
    sim_results = {}
    for trainset in ['synthetic_textual', 'synthetic_numeric']:
        sim_results[trainset] = {}
        logger.info(f"Calculating similarity for {trainset} against DROP")
        train_reformat = []
        for i in range(len(train_data.data[trainset]['question'])):  #reformat train to be same format as eval datasets below..
            train_reformat.append({'question': train_data.data[trainset]['question'][i],
                                   'answer': train_data.data[trainset]['answer'][i]} )
        gt_file = results_dict[testset]['gt_file']
        dev_data = QAData(logger, args, gt_file, False)
        prefmetric = results_dict[testset]['prefer']
        if prefmetric == 'RL':  # No sample-level results for RL
            prefmetric = 'F1'
        scores = results_dict[testset][prefmetric]['scores']
        result_score, result_detail = UQADataset().run_similarity_comparer(train_reformat, dev_data.data, 
                                                                           answer_thresh=args.answer_thresh, question_thresh=0.0, 
                                                                           use_cosine='f1')
        sim_results[trainset][testset] = {'sim_scores': result_score, 
                                          'sim_details': result_detail, 
                                          'test_metric': prefmetric, 
                                          'test_scores': scores}

        logger.info(f"Finished calculating F1 similarities for {trainset} vs DROP. Saved results into {results_file_out}..")
        with open(results_file_out, 'w') as f:
            json.dump(sim_results, f)
    logger.info(f"Finished saving results into {results_file_out}!")
    return


def create_sentence_embeddings(args, logger):
    """ Create sentence embeddings for each train dataset and each eval dataset
        Save them in the respective data directories
    To test:
        run cli.py args stuff
        args.train_file = '/data/thar011/data/unifiedqa/train.tsv'
        args.output_dir = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd'
        args.is_unifiedqa = True
        args.mixture = 'unifiedqa,synthetic_textual,synthetic_numeric'
        args.predict_batch_size=20
        logger = None
        trainset = 'narrativeqa'
        testset = 'drop' 
        args.add_only_missing = True
        args.use_question_only = False
        manually run relevant lines below..        
    """
    logger.info("Creating train and eval sentence embeddings...")
    train_data = UnifiedQAData(logger, args, args.train_file, False)
    results_file = os.path.join(args.output_dir, 'eval_metrics.json')
    if os.path.exists(results_file):
        results_dict = json.load(open(results_file)) 
    else:
         logger.info(f"ERROR! Evaluation Metrics File {results_file} not found. Exiting..")
         assert os.path.exists(results_file)
    out_dir_base = args.train_file.replace('/train.tsv', '/')
    s = Embedder()
         
    #create sentence embeddings from train datasets...     
    if args.use_question_only:
        out_emb_file_train = 'train_emb_qonly.pkl'
    elif args.reformat_question_ssvise:
        out_emb_file_train = 'train_emb_ssvise.pkl'
    else:
        out_emb_file_train = 'train_emb.pkl'
    for i, trainset in enumerate(train_data.data.keys()):
        out_dir = os.path.join(out_dir_base, trainset)
        out_file = os.path.join(out_dir, out_emb_file_train)
        if args.add_only_missing and os.path.exists(out_file):
            logger.info(f"Skipping Calculating embeddings for train data of {trainset} as embedding file already exists")
            continue
        ssvised = train_data.selfsupervised[i]
        logger.info(f"Calculating embeddings for train data of {trainset} self-supervised:{ssvised}:")
        questions = train_data.data[trainset]['question']
        answers = train_data.data[trainset]['answer']
        if args.use_question_only:  # not used
            questions = s.strip_context(questions)    
        questions = [q.strip() for q in questions]
        answers = [a.strip() for a in answers]
        if args.reformat_question_ssvise:
            questions = restate_qa_all(questions, answers)
        if args.do_lowercase:  #roberta is case sensitive so don't use this flag with sroberta..
            questions = [q.lower() for q in questions]
            answers = [a.lower() for a in answers]
        num_samples = len(questions)
        questions_emb = []
        answers_emb = []
        for i in range(0, num_samples, args.predict_batch_size):
            enc = s.encode_inputs(questions[i:i+args.predict_batch_size])  #NB for list[5:50] on a 10 element list python returns items 5:10 only hence partial batch at end is picked up
            emb = s.get_embeddings(enc)
            questions_emb += emb
            if not ssvised and not args.reformat_question_ssvise:  # don't create embeddings for non-existent self supervised dataset answers..
                enc = s.encode_inputs(answers[i:i+args.predict_batch_size])
                emb = s.get_embeddings(enc)
                answers_emb += emb
            if i % 1000 == 0:
                print(f'{trainset}: Processed: {i}')
            
        questions_emb = np.array(questions_emb)
        answers_emb = np.array(answers_emb)
        with open(out_file, "wb") as f:
            pickle.dump({'question': questions_emb, 'answer': answers_emb}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Embeddings for training data of {trainset} written to {out_file}')

    #create sentence embeddings from eval datasets
    for testset in results_dict.keys():
        gt_file = results_dict[testset]['gt_file']
        dev_data = QAData(logger, args, gt_file, False)
        if gt_file.endswith('test.tsv'):
            if args.use_question_only:
                out_emb_file_test = 'test_emb_qonly.pkl'
            elif args.reformat_question_ssvise:
                out_emb_file_test = 'test_emb_ssvise.pkl'
            else:
                out_emb_file_test = 'test_emb.pkl'
        else:
            if args.use_question_only:
                out_emb_file_test = 'dev_emb_qonly.pkl'
            elif args.reformat_question_ssvise:
                out_emb_file_test = 'dev_emb_ssvise.pkl'
            else:
                out_emb_file_test = 'dev_emb.pkl'
        out_dir = os.path.join(out_dir_base, testset)
        out_file = os.path.join(out_dir, out_emb_file_test)
        if args.add_only_missing and os.path.exists(out_file):
            logger.info(f"Skipping Calculating embeddings for eval data of {testset} as embedding file already exists")
            continue
        logger.info(f"Calculating embeddings for eval data of {testset}:")
            
        questions = []
        answers = []
        for sample in dev_data.data:
            questions.append( sample['question'] )
            answers.append( sample['answer'][0] )
        if args.use_question_only:
            questions = s.strip_context(questions)                
        questions = [q.strip() for q in questions]
        answers = [a.strip() for a in answers]
        if args.reformat_question_ssvise:
            questions = restate_qa_all(questions, answers)
        if args.do_lowercase:
            questions = [q.lower() for q in questions]
            answers = [a.lower() for a in answers]
            
        num_samples = len(questions)
        questions_emb = []
        answers_emb = []
        ssvised = False
        if answers[0] == '':
            ssvised = True
        for i in range(0, num_samples, args.predict_batch_size):
            enc = s.encode_inputs(questions[i:i+args.predict_batch_size])
            emb = s.get_embeddings(enc)
            questions_emb += emb
            if not ssvised and not args.reformat_question_ssvise:
                enc = s.encode_inputs(answers[i:i+args.predict_batch_size])
                emb = s.get_embeddings(enc)
                answers_emb += emb
            if i % 1000 == 0:
                print(f'{testset}: Processed: {i}')
               
        questions_emb = np.array(questions_emb)
        answers_emb = np.array(answers_emb)
        with open(out_file, "wb") as f:
            pickle.dump({'question': questions_emb, 'answer': answers_emb}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Embeddings for eval data of {testset} written to {out_file}')
    logger.info('Finished Creating Embeddings!')
    return


def calc_similarity_embeddings(args, logger):
    """ Calculate the cosine similarity between train and eval dataset using sentence embeddings
    Note can compute cos sim for [30k,1024] against [2.6m,1024] in 768GB RAM but 60K vs 2.6m exceeds 768GB ram
        
        To test:
        run cli.py args stuff
        args.train_file = '/data/thar011/data/unifiedqa/train.tsv'
        args.output_dir = '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd'
        args.is_unifiedqa = True
        args.mixture = 'unifiedqa,synthetic_textual,synthetic_numeric'
        args.answer_thresh = -100.1
        args.use_question_only = True
        logger = None
        trainset = 'narrativeqa'
        testset = 'drop' 
        args.add_only_missing = True
        args.use_question_only = False
        manually run relevant lines below..
    """
    logger.info("Calculating sentence embedding cosine similarity between train and test sets..")
    logger.info(f"Answer threshold: {str(args.answer_thresh)}")
    if args.use_question_only:
        results_file_out = os.path.join(args.output_dir, f'eval_test_train_similarities_semb_qonly_thresh{str(args.answer_thresh)}.json')
    else:    
        results_file_out = os.path.join(args.output_dir, f'eval_test_train_similarities_semb_thresh{str(args.answer_thresh)}.json')
    logger.info(f"Output will be saved into {results_file_out}")

    train_data = UnifiedQAData(logger, args, args.train_file, False)
    results_file = os.path.join(args.output_dir, 'eval_metrics.json')
    if os.path.exists(results_file):
        results_dict = json.load(open(results_file)) 
    else:
         logger.info(f"ERROR! Evaluation Metrics File {results_file} not found. Exiting..")
         assert os.path.exists(results_file)
    out_dir_base = args.train_file.replace('/train.tsv', '/')

    sim_results = {}
    if os.path.exists(results_file_out):
        logger.info(f"Loading existing sim_results from {results_file_out}")
        sim_results = json.load(open(results_file_out))
    else:
        logger.info(f"No existing sim_results file {results_file_out}. Starting from empty file.")

    changed = False       
    for i, trainset in enumerate(train_data.data.keys()):
        ssvised = train_data.selfsupervised[i]
        logger.info(f"Calculating embedding similarity for {trainset} against:")
        train_reformat = []
        for i in range(len(train_data.data[trainset]['question'])):  #reformat train to be same format as eval datasets below..
            train_reformat.append({'question': train_data.data[trainset]['question'][i],
                                   'answer': train_data.data[trainset]['answer'][i]} )
            
        out_dir = os.path.join(out_dir_base, trainset)
        if args.use_question_only:
            out_file = os.path.join(out_dir, 'train_emb_qonly.pkl')
        else:
            out_file = os.path.join(out_dir, 'train_emb.pkl')
        with open(out_file, "rb") as f:
            train_emb = pickle.load(f)
        
        if sim_results.get(trainset) is None:
            sim_results[trainset] = {}
            changed = True
            
        for testset in results_dict.keys():
            if testset == 'narrativeqa':
                logger.info(f" ... skipping {testset}")  #insufficient memory and not needed
            else:    
                if args.add_only_missing and (sim_results[trainset].get(testset) is not None):
                    logger.info(f" ... skipping {testset} as it already exists..")
                    continue
                gt_file = results_dict[testset]['gt_file']
                dev_data = QAData(logger, args, gt_file, False)
                prefmetric = results_dict[testset]['prefer']
                if prefmetric == 'RL':  # No sample-level results for RL
                    prefmetric = 'F1'
                scores = results_dict[testset][prefmetric]['scores']
                if gt_file.endswith('test.tsv'):
                    if args.use_question_only:
                        emb_file_test = 'test_emb_qonly.pkl'
                    elif ssvised and args.reformat_question_ssvise:  # if trainset is ssvised then optionally use reformatted eval ssvise-style embeddings
                        emb_file_test = 'test_emb_ssvise.pkl'
                    else:
                        emb_file_test = 'test_emb.pkl'
                else:
                    if args.use_question_only:
                        emb_file_test = 'dev_emb_qonly.pkl'
                    elif ssvised and args.reformat_question_ssvise:  # if trainset is ssvised then optionally use reformatted eval ssvise-style embeddings
                        emb_file_test = 'dev_emb_ssvise.pkl'
                    else:
                        emb_file_test = 'dev_emb.pkl'
                out_dir = os.path.join(out_dir_base, testset)
                out_file = os.path.join(out_dir, emb_file_test)
                logger.info(f" ... {testset} using {out_file}")

                with open(out_file, "rb") as f:
                    test_emb = pickle.load(f)
    
                result_score, result_detail = UQADataset().run_similarity_comparer(train_reformat, dev_data.data, 
                                                                                   answer_thresh=args.answer_thresh, question_thresh=0.0, 
                                                                                   use_cosine='emb', train_emb=train_emb, test_emb=test_emb)
                sim_results[trainset][testset] = {'sim_scores': result_score, 
                                                  'sim_details': result_detail, 
                                                  'test_metric': prefmetric, 
                                                  'test_scores': scores}
                changed = True

    if changed:
        logger.info(f"Finished calculating test-train embedding similarities. Saving results into {results_file_out}..")
        with open(results_file_out, 'w') as f:
            json.dump(sim_results, f)
        logger.info(f"Finished saving results into {results_file_out}!")
    else:
        logger.info(f"No updates so {results_file_out} not updated...")
    return








