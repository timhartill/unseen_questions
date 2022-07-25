# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

from run import run

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--train_file", default="data/train.tsv")
    parser.add_argument("--predict_file", default="data/dev.tsv")
    parser.add_argument("--output_dir", default=None, type=str)  #TJH , required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_predict_all", action='store_true',
                        help="Output predictions for all datasets in dataset_attributes.py dev_eval and test_eval.")
    parser.add_argument("--is_unifiedqa", action='store_true',
                        help="If set and --do_train will train on the mixture specified in --mixture.")
    parser.add_argument("--skip_inference", action='store_true')

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument('--checkpoint_step', type=int, default=0)
    parser.add_argument("--do_lowercase", action='store_true')

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true')

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", default=10000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)
    
    # Added parameters
    parser.add_argument("--num_scheduler_steps", default=250000, type=int,
                        help="Linear lr decay to zero over this number of scheduler steps (scheduler step = global_steps // grad accumulation steps).")
    parser.add_argument("--save_best_model_steps", default=50000, type=int,
                        help="Save the best model to date as eg. best-model-50000.pt every 50000 steps.")
    parser.add_argument("--model", type=str, default="facebook/bart-large",
                        help="The model to train or predict from.")
    parser.add_argument("--mixture", type=str, default="unifiedqa",
                        help="The mixture of datasets to train on. Format: --mixture unifiedqa,extradataset1,extradataset2")
    parser.add_argument("--calc_metrics", action='store_true',
                        help="Calculate additional dataset-specific metrics beyond EM. --do_predict must have been run previously or specifed here as well.")
    parser.add_argument("--calc_metrics_all", action='store_true',
                        help="Calculate additional dataset-specific metrics beyond EM for each eval dataset. --do_predict_all must have been run previously or specifed here as well.")
    parser.add_argument("--strip_single_quotes", action='store_true',
                        help="Strip closed 'single quote pairs' but not single possessive type quotes eg I'm")
    parser.add_argument("--indiv_digits", action='store_true',
                        help="Tokenize 123.4 as ['1', '2', '3', '.', '4'] rather than eg ['123','.4'] ")
    parser.add_argument("--norm_numbers", action='store_true',
                        help="Normalise numbers before tokenisation eg 1,678.54 becomes 1678.54 and .90 becomes 0.9, 009 becomes 9 and twenty becomes 20 unless --norm_10e also set")
    parser.add_argument("--norm_10e", action='store_true',
                        help="if --norm_numbers and this is set use form '- 4 10e1 3 10e0 2 10e-1 1 10e-2' instead of -43.21 for all identifed numbers")
    parser.add_argument("--error_based_sampling", action='store_true',
                        help="In multitask training sample tasks based on oversampling those with higher dev error. Default is uniform sampling")
    parser.add_argument("--error_based_ssvise_prob", default=0.5, type=float,
                        help="For error based sampling the probability of choosing a self supervised dataset")
    parser.add_argument("--calc_similarity", action='store_true',
                        help="Calculate similarity between train datasets specified by  --mixture and eval datasets specified in eval_metrics.json.")
    parser.add_argument("--calc_similarity_numeric", action='store_true',
                        help="Calculate F1 similarity between synthetic_textual and synthetic_numeric train datasets vs DROP")
    parser.add_argument("--answer_thresh", default=-100.1, type=float,
                        help="For similarity calc this is the threshold on answer similarities to be considered a match")
    parser.add_argument("--create_embeddings", action='store_true',
                        help="Create sentence embeddings for train datasets specified by  --mixture and eval datasets specified in eval_metrics.json. --predict_batch_size must be specified")
    parser.add_argument("--use_question_only", action='store_true',
                        help="Create sentence embeddings only using question part of the input context.")    
    parser.add_argument("--reformat_question_ssvise", action='store_true',
                        help="Create sentence embeddings from question + answer reformatted to be similar to self-supervised training format.")
    parser.add_argument("--calc_similarity_embeddings", action='store_true',
                        help="Calculate cosine similarity between sentence embeddings between train datasets specified by  --mixture and eval datasets specified in eval_metrics.json. --create_embeddings must be run before running this.")
    parser.add_argument("--add_only_missing", action='store_true',
                        help="If true only missing sentence embedding files are added.")
    parser.add_argument("--ssm_prob", default=0.5, type=float,
                        help="If sample is self supervised and named entities have been found, the prob of masking using SSM vs WWSC.")
    parser.add_argument("--wwsc_toks_to_mask", default=0.11, type=float,
                        help="If sample is self supervised and doing WWSC, the proportion of tokens to mask (will adjusted upwards to whole word boundary).")
    parser.add_argument("--wwsc_avg_span_len", default=2, type=int,
                        help="If sample is self supervised and doing WWSC, the mean token span length (will adjusted upwards to whole word boundary).")
    parser.add_argument("--wwsc_span_len_sd", default=0.75, type=float,
                        help="If sample is self supervised and doing WWSC, the std deviation on token span length.")
    parser.add_argument("--add_mask_char", type=str, default="_",
                        help="Add a character after <mask> e.g. make it <mask>_. Specify NONE to exclude this.")
    parser.add_argument("--add_mask_ctr", action='store_true',
                        help="Add a counter to <mask> e.g. make it <mask>_2")

    parser.add_argument("--gen_explanations_all", action='store_true',
                        help="Generate explanations e for datasets specified in dataset_attibutes.create_datasets_dynamic, append as q[+mc]+e->a and save as new uqa-formatted datasets in dataset_attributes.UQA_DIR")
    parser.add_argument("--dont_save_train_token_file", action='store_true',
                        help="If set the preprocessed token file for train mixtures won't be saved to disk or loaded from disk")



    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=2000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    assert args.output_dir is not None and args.output_dir != '', "Output directory must be specified using --output_dir"
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict \
        and not args.calc_metrics and not args.calc_similarity \
        and not args.calc_similarity_numeric and not args.create_embeddings and not args.calc_similarity_embeddings \
        and not args.do_predict_all and not args.calc_metrics_all and not args.gen_explanations_all:
        raise ValueError("At least one of `do_train` or `do_predict` or `do_predict_all` or 'calc_metrics' or 'calc_metrics_all' or calc_similarity or calc_similarity_numeric or create_embeddings or calc_similarity_embeddings or gen_explanations_all must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict or args.do_predict_all:
        if not args.predict_file:
            raise ValueError("If `do_predict(_all)` is True, then `predict_file` must be specified.")
    if args.calc_metrics or args.calc_metrics_all:
        if not args.predict_file:
            raise ValueError("If `calc_metrics(_all)` is True, then `predict_file` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    run(args, logger)

if __name__=='__main__':
    main()
