# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import argparse


def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--predict_file", type=str, default="")
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
    parser.add_argument("--predict_batch_size", default=512, type=int, help="Total batch size for predictions.")
    parser.add_argument("--shared-encoder", action="store_true")
    parser.add_argument("--save_prediction", default="", type=str)

    # multi vector scheme
    #parser.add_argument("--multi-vector", type=int, default=1)
    #parser.add_argument("--scheme", type=str, help="how to get the multivector, layerwise or tokenwise", default="none")

    # momentum
    parser.add_argument("--momentum", action="store_true", help="If true, perform momentum training.")
    parser.add_argument("--init_retriever", type=str, default="", help="Ckpt to load for Momentum training.")
    parser.add_argument("--k", type=int, default=38400, help="memory bank size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum")


    # NQ multihop trial
    #parser.add_argument("--nq-multi", action="store_true", help="train the NQ retrieval model to recover from error cases")
    
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
    parser.add_argument("--output_dir", default="./logs", type=str, help="The output directory where the model checkpoints, logs etc will be written.")
    parser.add_argument("--sp-weight", default=1.0, type=float, help="weight of the sentence relevance prediction loss")
    parser.add_argument('--prefix', type=str, default="eval")

    return parser


def train_args():
    parser = common_args()
    # optimization
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
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
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=2500)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--reduction", default="none", type=str,
                        help="type of reduction to apply in cross-entropy loss - 'sum', 'mean' or 'none' gives sum over mean per hop.")
    parser.add_argument("--retrieve_loss_multiplier", default=1.0, type=float,
                        help="Retrieve loss multiplier. Final loss will be stop_loss + retrieve_loss_multiplier*retrieve_loss")

    return parser.parse_args()


def encode_args():
    parser = common_args()
    parser.add_argument('--embed_save_path', type=str, default="", help="Directory to save into. Will be created if doesnt exist.")
    parser.add_argument('--is_query_embed', action="store_true")
    parser.add_argument('--update_id2doc_only', action="store_true")

    args = parser.parse_args()
    return args


def eval_args():
    parser = common_args()
    parser.add_argument('--index_path', type=str, default=None, help="index.npy file containing para embeddings [num_paras, emb_dim]")
    parser.add_argument('--corpus_dict', type=str, default=None, help="id2doc.json file containing dict with key id -> title+txt")
    parser.add_argument('--topk', type=int, default=2, help="Retrievel eval: topk paths/para sequences to return. Must be <= beam-size^num_steps. Iterator: max num of sents returned from stage 1")
    parser.add_argument('--topk_stage2', type=int, default=5, help="Iterator: max num of sents returned from stage 2")
    parser.add_argument('--beam_size', type=int, default=5, help="Number of beams each step (number of nearest neighbours to return each retrieval step.).")
    parser.add_argument('--gpu_faiss', action="store_true", help="Put Faiss index on visible gpu(s).")
    parser.add_argument('--gpu_model', action="store_true", help="Put q encoder on gpu 0 of the visible gpu(s).")
    parser.add_argument('--save_index', action="store_true",help="Save index if hnsw option chosen")
    parser.add_argument('--only_eval_ans', action="store_true")
    parser.add_argument('--hnsw', action="store_true", help="Non-exhaustive but fast and relatively accurate. Suitable for FAISS use on cpu.")
    parser.add_argument('--hnsw_buffersize', type=int, default=20000000, help="Buffer size (ie num docs to load into hnsw index in an iteration), if building hnsw index. 20000000 fits in ~700GB RAM. 40000000 is enough for all english wikipedia in 1 pass but requires at least 800GB RAM, maybe a bit more")
    parser.add_argument('--strict', action="store_true", help="load ckpt in 'strict' mode")
    parser.add_argument('--exact', action="store_true", help="filter ckpt in 'exact' mode")
    parser.add_argument("--model_name_stage", default='google/electra-large-discriminator', type=str, help="stage 1 rereranker model name")
    parser.add_argument("--init_checkpoint_stage1", default='', type=str, help="stage 1 rereranker model checkpoint")
    parser.add_argument("--init_checkpoint_stage2", default='', type=str, help="stage 2 rereranker model checkpoint")
    parser.add_argument('--s1_use_para_score', action="store_true", help="Stage 1: Use s1 para score + s1 sent score (vs s1 sent score only) in selecting topk sentences.")
    parser.add_argument('--s1_para_sent_ratio', default=0.5, type=float, help="Stage 1 iter select: Ratio for formula s1_para_sent_ratio*s1 para score + (1-s1_para_sent_ratio)*s1 sent score (vs s1 sent score only) in selecting topk sentences.")
    parser.add_argument('--s1_para_sent_ratio_final', default=0.5, type=float, help="Stage 1 final select: Ratio for formula s1_para_sent_ratio_final*s1 para score + (1-s1_para_sent_ratio_final)*max(s1 sent score) (vs s1 sent score only) in selecting final R@20 topk sentences.")
    parser.add_argument('--s2_use_para_score', action="store_true", help="Stage 2: Use s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting s2 topk sentences.")
    parser.add_argument('--s2_para_sent_ratio_final', default=0.5, type=float, help="Stage 2 final select: Ratio for formula s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting final topk sentences.")
    parser.add_argument('--s2_para_sent_ratio', default=0.5, type=float, help="Stage 2 iter select: Ratio for formula s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting topk sentences.")
    parser.add_argument("--s2_sp_thresh", default=0.1, type=float, help="Min stage 2 sent score for selection (Note: A minimum of s2_min_take sentences will be taken even if no sentences meet this value).")
    parser.add_argument('--s2_min_take', type=int, default=2, help="Min number of sentences to select from stage 2")
    parser.add_argument("--stop_ev_thresh", default=99.0, type=float, help="Stop iterating if s2ev_score >= this (s2ev_score between 0 & 1).")
    parser.add_argument("--stop_ansconfdelta_thresh", default=99999.0, type=float, help="Stop iterating if s2_ans_conf_delta >= this (Note: s2_ans_conf_delta not between 0 & 1. If 0.0 ans pred is insuff evidence).")
    parser.add_argument("--stop_lowerev", action="store_true", help="Stop iterating if current hop s2ev_score < last hop s2ev_score.")
    parser.add_argument("--output_dataset", default='', type=str, help="Full path to output tsv-formatted files to. Typically /parent/.../unifiedqa/newdatasetname/train|dev|test.tsv ")
    parser.add_argument('--resume_dir', type=str, default=None, help="Path to log dir containing samples_with_context.jsonl to resume adding to.")
    parser.add_argument('--ctx_topk_paras', type=int, default=-1, help="Number of paras to include in final context build. -1 means include all.")
    parser.add_argument('--ctx_gold_sents_only', action="store_true", help="If set only sentences from s2 included in final context. Otherwise 1 sentence before/after each s2 sent is included.")    
    
    return parser.parse_args()


def llm_args():
    parser = common_args()
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Max number of new tokens to generate excluding input prompt")
    parser.add_argument('--max_seq_len_in', type=int, default=1024, help="Max number of input tokens.")
    parser.add_argument('--max_memory', type=int, default=-1, help="Max mem in GB to allocate on each visible GPU. -1=auto.")
    parser.add_argument('--max_memory_buffer', type=int, default=6, help="#GB to subtract from max mem on each gpu if max_memory=auto.")
    parser.add_argument("--output_dataset", default='', type=str, help="Full path to output tsv-formatted files to. Typically /parent/.../unifiedqa/newdatasetname/train|dev|test.tsv ")
    parser.add_argument("--template_file", default='', type=str, help="template file name without path (if specified will enable single mode). UQA_DIR/prompts/ will be prepended.")
    parser.add_argument('--resume_dir', type=str, default=None, help="Path to log dir containing samples_with_context_llm.jsonl to resume adding to.")
    parser.add_argument('--generate_train', action="store_true", help="Eval mode: Generate rationales for train.tsvs specified in TRAIN_SETS.")
    parser.add_argument('--generate_dev', action="store_true", help="Eval mode: Generate rationales for dev.tsvs specified in TRAIN_SETS.")
    parser.add_argument('--generate_eval', action="store_true", help="Eval mode: Generate rationales for dev.tsvs and test.tsvs specified in EVAL_SETS_DEV|TEST.")
    parser.add_argument('--max_samples', type=int, default=-1, help="Max samples to generate rationales for. -1=all.")
    parser.add_argument('--do_sample', action="store_true", help="If True do topk or top p sampling instead of beam/greedy search")
    parser.add_argument("--top_k", default=0, type=int, help="If do_sample=True, only sample from the top_k most likely words. 50 is often an ok value")
    parser.add_argument("--top_p", default=0.0, type=float, help="Nucleus sampling: If do_sample=True, only sample from the top words whose collective prob exceeds p so lower p = fewer but higher prob words to choose from. 0.92 is often an ok value")
    parser.add_argument("--temperature", default=1.0, type=float, help="If do_sample=True, the lower the temperature the higher the chances of choosing high-prob words. eg 0.7")   
    parser.add_argument('--num_beams', type=int, default=1, help="Number of beams if do_sample=False (1=greedy, >1 = beam search).")
    parser.add_argument('--num_return_sequences', type=int, default=1, help="Number of sequences to return. Must be <= num_beams in beam search.")
    
    return parser.parse_args()
    



