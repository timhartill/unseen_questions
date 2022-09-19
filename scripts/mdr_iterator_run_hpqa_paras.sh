# Run iterator ie retrieve/stage1/stage2 on previously trained models/encoded corpus
# Will build hnsw index if this option selected and index doesnt exist

# MDR: Per issue 8: We used a large beam size for the leaderboard submission. We used beam size 200 and top250 passage sequences for answer extraction.

# --gpu_model: puts the model on single gpu (0 of visible gpus). Without --gpu_model keeps model on cpu
# --gpu_faiss: puts the faiss index on gpu. Without keeps on cpu. Overruled by hnsw if that is set
# --hnsw: Convert faiss index to hnsw and run on cpu (if not already built). Will load existing hnsw index /index_path/index_hnsw.index if it exists instead of building it 
#         hnsw buffer size 20,000,000 takes ~35 hours and 500GB RAM to build index. 2nd pass of 15M vectors takes longer than 1st pass of 20M vectors.
# --save_index: if HNSW option chosen, if index built here then saves the index to the --index_path dir as index_hnsw.index

# after training (base or base+momentum), run mdr_encode_datasets then mdr_eval_mhop_retrieval_nativeamp.sh then run this sh AFTER setting:
# index_path, corpus_dict to the location that mdr_encode_datasets outputted to
# model_path, output_dir to location of the ckpt dir that mdr_encode_datasets was run on. Note for momentum can pick either checkpoint_q_best.pt or checkpoint_k_best.pt as they are identical..


# predict_file: /home/thar011/data/DPR/bqa_nq_tqa_qas_val.jsonl \  #use with bqa wiki dump ie gold sp = wiki doc id _ para idx
# /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val_with_spfacts.jsonl  # use with hpqa abstracts ie gold sp = title
# bqa eval file: /home/thar011/data/beerqa/beerqa_qas_val.jsonl
# hover eval file: /home/thar011/data/baleen_downloads/hover/hover_qas_val.jsonl 

# beam_size: # paras to retrieve each hop
# topk: max # sents to return from stage 1 ie prior s2 sents + selected s1 sents <= 9
# topk_stage2: max num sents to return from stage 2
# s1_use_para_score: Stage 1: if set add s1 para rank score to each s1 sent score before selecting topk s1 sentences
# s1_para_sent_ratio: Stage 1 iter select: Ratio for formula s1_para_sent_ratio*s1 para score + (1-s1_para_sent_ratio)*s1 sent score (vs s1 sent score only) in selecting topk sentences.
# s1_para_sent_ratio_final: Stage 1 final select: Ratio for formula s1_para_sent_ratio_final*s1 para score + (1-s1_para_sent_ratio_final)*max(s1 sent score) (vs s1 sent score only) in selecting final R@20 topk sentences.
# s2_use_para_score: Stage 2: if set add s1 para rank score to each s2 sent score before selecting topk_stage2 s2 sentences: s2_para_sent_ratio*s1 para score + (1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only)
# s2_para_sent_ratio 0.5 : Stage 2 iter select: Ratio for formula s2_para_sent_ratio*s1 para score+(1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting topk sentences.
# s2_para_sent_ratio_final: Stage 2 final select: Ratio for formula s2_para_sent_ratio*s1 para score+(1-s2_para_sent_ratio)*s2 sent score (vs s2 sent score only) in selecting final topk sentences.
                          # Set to -1.0 to empirically determine a ratio
# fp16: use fp16 on all models if set
# max_q_len: max len of question excluding sents added to query by iterator
# max_q_sp_len: retriever max input seq length 
# max_c_len: stage models max input seq length
# max_ans_len: stage models max answer seq length
# predict_batch_size: batch size for stage 1 model, set small so can fit all models on 1 gpu
# s2_sp_thresh: s2 sent score min for selection (unless < s2_min_take sents over this score in which case will take all)
# s2_min_take: Min number of sentences to select from stage 2
# max_hops: max number of retrieve->s1->s2 iterations on each sample
# stop_ev_thresh: stop iterating if s2_ev_score >= this thresh. Set > 1.0 to ignore. 0.6 = best per s2 train eval but set higher to make stopping because of this conservative
# stop_ansconfdelta_thresh = 18.0     # stop if s2_ans_conf_delta >= this thresh. Set to large number eg 99999.0 to ignore. Set high to make stopping because of this very rare
# args.stop_lowerev = False           # Stop iterating if current hop s2ev_score < last hop s2ev_score. 
# query_use_sentences = True      # Retriever only: If true use title: sents form in query otherwise use full para text (note s1 query always uses title | sents form)
# query_add_titles = True         # Retriever only: Prepend query paras with para title (only if using paras, if using sents always prepending title regardless)
                                  # Note only add this if retriever was trained with titles prepended - FALSE for hpqa abstracts since many paras begin with a paraphrase of the title

# encoded corpus: hpqa_varinitialtest2_03_66_be24_no_momentum  1gpu +varsteps no momentum using para query encoding
#    --index_path /large_data/thar011/out/mdr/encoded_corpora/hpqa_varinitialtest2_03-26_bs24_no_momentum/index.npy \
#    --corpus_dict /large_data/thar011/out/mdr/encoded_corpora/hpqa_varinitialtest2_03-26_bs24_no_momentum/id2doc.json \
#    --model_name roberta-base \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/03-26-2022/varinitialtest2_-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue/checkpoint_best.pt \

# encoded corpus: hpqa_mominitialvar_-03-28_bs24_with_momentum  1gpu +varsteps +momentum using para query encoding mom training from above 03-26-2022 ckpt
#    --index_path /large_data/thar011/out/mdr/encoded_corpora/hpqa_mominitialvar_-03-28_bs24_with_momentum/index.npy \
#    --corpus_dict /large_data/thar011/out/mdr/encoded_corpora/hpqa_mominitialvar_-03-28_bs24_with_momentum/id2doc.json \
#    --model_name roberta-base \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/mominitialvar_-03-28-2022-mom-seed16-bsz50-fp16True-lr1e-05-decay0.0-warm0.1-valbsz100-m0.999-k76800-t1.0-ga1-varTrue/checkpoint_q_best.pt \

# encoded corpus: /hpqa_mdr_orig_ckpt_8gpu_bs150 mdr original ckpt
#    --index_path /large_data/thar011/out/mdr/encoded_corpora/hpqa_mdr_orig_ckpt_8gpu_bs150/wiki_index.npy \
#    --corpus_dict /large_data/thar011/out/mdr/encoded_corpora/hpqa_mdr_orig_ckpt_8gpu_bs150/id2doc.json \
#    --model_name roberta-base \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/hpqa_mdr_orig_ckpt_8gpu_bs150/q_encoder.pt \

# encoded corpus: full wikipedia (bqa dump) mencoder trained on bqa minus squad +nq +tqa
#    --index_path /large_data/thar011/out/mdr/encoded_corpora/bqa_nosquad_nq_tqa_test3-04-14_bs24_no_momentum_cenone_ckpt_best/index.npy \
#    --corpus_dict /large_data/thar011/out/mdr/encoded_corpora/bqa_nosquad_nq_tqa_test3-04-14_bs24_no_momentum_cenone_ckpt_best/id2doc.json \
#    --model_name roberta-base \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/bqa_nosquad_nq_tqa_test3-04-14-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \



#    --s1_use_para_score \
#     --stop_ev_thresh 0.91 \
#    --stop_ansconfdelta_thresh 18.0
#     --hnsw \
#    --gpu_faiss \
#     --stop_lowerev \

# predict files:
# /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val_with_spfacts.jsonl \
# /home/thar011/data/baleen_downloads/hover/hover_qas_val_with_spfacts.jsonl \
# /home/thar011/data/strategyqa/strategyqa_aristotle_qas_val_with_spfacts.jsonl \
# $UQA_DIR/aristotle_od_ans/test.tsv
# $UQA_DIR/csqa2/dev.tsv
# $UQA_DIR/creak_od_ans/dev.tsv
# $UQA_DIR/creak_od_ans_impl_rels/dev.tsv    $UQA_DIR/creak_od_ans_impl_rels/train.tsv
# $UQA_DIR/csqa2_impl_rels/dev.tsv           $UQA_DIR/csqa2_impl_rels/train.tsv
# $UQA_DIR/creak_initial_context/dev.tsv
# $UQA_DIR/nq_open_od_ans/dev.tsv            $UQA_DIR/nq_open_od_ans/train.tsv
# $UQA_DIR/hpqa_od_ans/dev.tsv            $UQA_DIR/hpqa_od_ans/train.tsv
# $UQA_DIR/musique_qa/dev.tsv            $UQA_DIR/musique_qa/train.tsv

# HPQA Abstracts encoded against best retriever:
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250-09-02-2022/index.npy
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250-09-02-2022/id2doc.json

# HPQA Abstracts +genericsKB encoded against best retriever:
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_hpqaabsgenerics_6gpubs250-09-02-2022/index.npy
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_hpqaabsgenerics_6gpubs250-09-02-2022/id2doc.json

# Full Wiki encoded against best retriever:
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/index.npy
# $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/id2doc.json


cd ../code

python mdr_searchers.py \
    --prefix ITER_fullwiki_mudev_test55_b60_h4_hpqahovnqmubs250_mom \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_qa_fullwiki_bs60/dev.tsv \
    --predict_file $UQA_DIR/musique_qa/dev.tsv \
    --index_path $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/index.npy \
    --corpus_dict $LDATA/out/mdr/encoded_corpora/hover_hpqa_nq_mu_paras_test12_mom_fullwiki_6gpubs250-09-02-2022/id2doc.json \
    --model_name roberta-base \
    --init_checkpoint $LDATA/out/mdr/logs/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250_hgx2-09-02-2022-mom-seed16-bsz250-fp16True-lr1e-05-decay0.0-warm0.1-valbsz100-m0.999-k76800-t1.0-ga1-varTrue-cenone/checkpoint_q_best.pt \
    --model_name_stage google/electra-large-discriminator \
    --init_checkpoint_stage1 $LDATA/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --init_checkpoint_stage2 $LDATA/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --gpu_model \
    --hnsw \
    --hnsw_buffersize 40000000 \
    --save_index \
    --beam_size 60 \
    --predict_batch_size 160 \
    --query_add_titles \
    --topk 9 \
    --topk_stage2 5 \
    --s1_use_para_score \
    --s1_para_sent_ratio 0.5 \
    --s1_para_sent_ratio_final -1.0 \
    --s2_use_para_score \
    --s2_para_sent_ratio 0.5 \
    --s2_para_sent_ratio_final -1.0 \
    --max_hops 4 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --max_c_len 512 \
    --max_ans_len 35 \
    --s2_sp_thresh 0.10 \
    --s2_min_take 2 \
    --stop_ev_thresh 1.01 \
    --stop_ansconfdelta_thresh 99999.0



