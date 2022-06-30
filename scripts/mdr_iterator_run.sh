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
# s1_use_para_score: if set add s1 para rank score to each sent score before selecting topk sentences
# fp16: use fp16 on all models if set
# max_q_len: max len of question excluding sents added to query by iterator
# max_q_sp_len: retriever max input seq length 
# max_c_len: stage models max input seq length
# max_ans_len: stage models max answer seq length
# predict_batch_size: batch size for stage 1 model, set small so can fit all models on 1 gpu
# s2_sp_thresh: s2 sent score min for selection (unless < 2 sents over this score in which case will take both)
# max_hops: max number of retrieve->s1->s2 iterations on each sample
# stop_ev_thresh: stop iterating if s2_ev_score >= this thresh. Set > 1.0 to ignore. 0.6 = best per s2 train eval but set higher to make stopping because of this conservative
# stop_ansconfdelta_thresh = 18.0     # stop if s2_ans_conf_delta >= this thresh. Set to large number eg 99999.0 to ignore. Set high to make stopping because of this very rare


cd ../code

python mdr_searchers.py \
    --prefix ITER_hpqaabst_hpqaeval_test0 \
    --output_dir /large_data/thar011/out/mdr/logs \
    --predict_file /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val_with_spfacts.jsonl \
    --index_path /large_data/thar011/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/index.npy \
    --corpus_dict /large_data/thar011/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/id2doc.json \
    --model_name roberta-base \
    --init_checkpoint /large_data/thar011/out/mdr/logs/hpqa_sent_annots_test1-04-18-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
    --model_name_stage google/electra-large-discriminator \
    --init_checkpoint_stage1 /large_data/thar011/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --init_checkpoint_stage2 /large_data/thar011/out/mdr/logs/stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix-06-14-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --gpu_model \
    --hnsw \
    --save_index \
    --beam_size 100 \
    --topk 9 \
    --topk_stage2 5 \
    --s1_use_para_score \
    --max_hops 2 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --max_c_len 512 \
    --max_ans_len 35 \
    --predict_batch_size 26 \
    --s2_sp_thresh 0.10 \
    --stop_ev_thresh 0.91 \
    --stop_ansconfdelta_thresh 18.0




