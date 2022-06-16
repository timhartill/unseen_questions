# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import argparse


def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str, default="../data/nq-with-neg-train.txt")
    parser.add_argument("--predict_file", type=str, default="../data/nq-with-neg-dev.txt")
    parser.add_argument("--num_workers", default=10, type=int, help="number of dataloader processes. 0 means run on main thread.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action="store_true", help="for final test submission")

    # model
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--init_checkpoint", type=str, help="Initial checkpoint to load weights from.", default="")
    parser.add_argument("--max_c_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=70, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_q_sp_len", default=50, type=int)
    parser.add_argument("--max_ans_len", default=35, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html - NOT USED in py native amp implementation")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--sent-level", action="store_true")
    parser.add_argument("--rnn-retriever", action="store_true")
    parser.add_argument("--predict_batch_size", default=512,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--shared-encoder", action="store_true")
    parser.add_argument("--save_prediction", default="", type=str)

    # multi vector scheme
    parser.add_argument("--multi-vector", type=int, default=1)
    parser.add_argument("--scheme", type=str, help="how to get the multivector, layerwise or tokenwise", default="none")

    # momentum
    parser.add_argument("--momentum", action="store_true", help="If true, perform momentum training.")
    parser.add_argument("--init_retriever", type=str, default="", help="Ckpt to load for Momentum training.")
    parser.add_argument("--k", type=int, default=38400, help="memory bank size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum")


    # NQ multihop trial
    parser.add_argument("--nq-multi", action="store_true", help="train the NQ retrieval model to recover from error cases")
    
    #TJH Added
    parser.add_argument("--use_var_versions", action="store_true", help="Use the generic variable step '..._var' versions.")
    parser.add_argument("--debug", action="store_true", help="If set prints extra debug info in criterions")
    parser.add_argument("--max_hops", type=int, default=2, help="Maximum number of hops in train-dev samples.")
    parser.add_argument("--num_negs", type=int, default=2, help="Number of adversarial negatives to include for each sample in training.")
    parser.add_argument("--query_use_sentences", action="store_true", help="Use the gold or predicted sentences within a paragraph in the query instead of the entire paragraph.")
    parser.add_argument("--query_add_titles", action="store_true", help="If --query_use_sentences then prepend sentences with para title in query encoding.")
    parser.add_argument("--random_multi_seq", action="store_true", help="If training type multi para sequencing, randomize para seq in each step.")   
    parser.add_argument("--sent_score_force_zero", action="store_true", help="Stage 1 reader training : Zero sentence scores where sent or para label is zero.")    
    parser.add_argument("--sp_percent_thresh", type=float, default=0.55, help="maximum mean fraction of sentences in para for a given sp score threshold to take in order for that thresh to be selected.")
    parser.add_argument("--num_workers_dev", default=0, type=int, help="number of dev dataloader processes. 0 means run on main thread.")
    parser.add_argument("--ev_combiner", action="store_true", help="reader training : Add evidence combining head to model and return extra ev_score key.")    
    parser.add_argument('--stop-drop', type=float, default=0.0, help="Dropout on stop head.")
    parser.add_argument("--eval_stop", action="store_true", help="Retriever: Add stop head and/or evaluate stop prediction accuracy in addition to evaluating para retrieval.")   

    return parser


def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints, logs etc will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_checkpoints_steps", default=20000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=2500)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--reduction", default="none", type=str,
                        help="type of reduction to apply in cross-entropy loss - 'sum', 'mean' or 'none' gives sum over mean per hop.")
    parser.add_argument("--retrieve_loss_multiplier", default=1.0, type=float,
                        help="Retrieve loss multiplier. Final loss will be stop_loss + retrieve_loss_multiplier*retrieve_loss")
    parser.add_argument("--sp-weight", default=0, type=float, help="weight of the sentence relevance prediction loss")

    return parser.parse_args()


def encode_args():
    parser = common_args()
    parser.add_argument('--embed_save_path', type=str, default="", help="Directory to save into. Will be created if doesnt exist.")
    parser.add_argument('--is_query_embed', action="store_true")
    args = parser.parse_args()
    return args


def eval_args():
    parser = common_args()
    parser.add_argument('--index_path', type=str, default=None, help="index.npy file containing para embeddings [num_paras, emb_dim]")
    parser.add_argument('--corpus_dict', type=str, default=None, help="id2doc.json file containing dict with key id -> title+txt")
    parser.add_argument('--topk', type=int, default=2, help="topk paths/para sequences to return. Must be <= beam-size^num_steps")
    parser.add_argument('--beam_size', type=int, default=5, help="Number of beams each step (number of nearest neighbours to append each step.).")
    parser.add_argument('--gpu_faiss', action="store_true", help="Put Faiss index on visible gpu(s).")
    parser.add_argument('--gpu_model', action="store_true", help="Put q encoder on gpu 0 of the visible gpu(s).")
    parser.add_argument('--save_index', action="store_true",help="Save index if hnsw option chosen")
    parser.add_argument('--only_eval_ans', action="store_true")
    parser.add_argument('--hnsw', action="store_true", help="Non-exhaustive but fast and relatively accurate. Suitable for FAISS use on cpu.")
    parser.add_argument('--strict', action="store_true")  #TJH Added - load ckpt in 'strict' mode
    parser.add_argument('--exact', action="store_true")  #TJH Added - filter ckpt in 'exact' mode
    return parser.parse_args()


    



