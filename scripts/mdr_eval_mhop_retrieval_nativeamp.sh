# eval retrieve on previouly trained model

# MDR: Per issue 8: We used a large beam size for the leaderboard submission. We used beam size 200 and top250 passage sequences for answer extraction.

# --gpu_model: puts the model on single gpu (0 of visible gpus). Without --gpu_model keeps model on cpu
# --gpu_faiss: puts the faiss index on gpu. Without keeps on cpu.
# --hnsw: Convert faiss index to hnsw and run on cpu. Will load existing hnsw index /index_path/index_hnsw.index if it exists instead of building it 
# --save_index: if HNSW option chosen, saves the index to the --index_path dir as index_hnsw.index (if didnt exist before)

# bqa eval file: /home/thar011/data/beerqa/beerqa_qas_val.jsonl
# bqa_nq_tqa eval file: /home/thar011/data/DPR/bqa_nq_tqa_qas_val.jsonl
# hover eval file: /home/thar011/data/baleen_downloads/hover/hover_qas_val.jsonl 



# with bs 100 beam size 1, topk 1 faiss on cpu & model on gpu (takes ~4gb and neglible impact on other jon running there since most time is with faiss) takes ~2.5 hrs

# bs 100 beam 1 topk 1 Uses 21.2GB on one GPU: Approx 18.4GB of this is the FAISS index over ~5M paras (on disk wiki_index.npy=15GB and wiki_id2doc.json = 3.1GB)
# bs 100 beam 2 topk 2: Uses 23.4GB GPU - only slightly more than beam/topk 1
# bs 100 beam 4 topk 4: Uses 27.8GB GPU - 8GB more than beam/topk 1
# bs 100 beam 20 topk 20: GPU OOM
# bs 10  beam 20 topk 20: 23.1GB GPU Used. Inference time goes from 3mins@topk1->15mins
# bs 10  beam 50 topk 50: 29.7GB GPU used. Inference time: 25 mins

#hnsw buffer size 20,000,000 takes ~35 hours and 500GB RAM to build index. 2nd pass of 15M vectors takes longer than 1st pass of 20M vectors.

# after training (base or base+momentum), run mdr_encode_datasets then run this sh AFTER setting:
# index_path, corpus_dict to location when mdr_encode_datasets output to
# model_path, output_dir to location of the ckpt dir that mdr_encode_datasets was run on. Note for momentum can pick either checkpoint_q_best.pt or checkpoint_k_best.pt as they are identical..


# predict_file: /home/thar011/data/DPR/bqa_nq_tqa_qas_val.jsonl \  #use with bqa wiki dump ie gold sp = wiki doc id _ para idx
# /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val.json  # use with hpqa abstracts ie gold sp = title


cd ../code

python mdr_eval_mhop_retrieval_nativeamp.py \
    --predict_file $LDATA/out/mdr/encoded_corpora/hotpot/hotpot_qas_val.json \
    --index_path $LDATA/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/index.npy \
    --corpus_dict $LDATA/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best/id2doc.json \
    --init_checkpoint $LDATA/out/mdr/logs/hpqa_sent_annots_test1-04-18-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
    --predict_batch_size 100 \
    --beam_size 1 \
    --topk 1 \
    --model_name roberta-base \
    --gpu_model \
    --hnsw \
    --save_index \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 400 \
    --use_var_versions \
    --max_hops 2 \
    --fp16 \
    --output_dir $LDATA/out/mdr/logs/hpqa_sent_annots_test1-04-18-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/test2_eval_beam1_topk1_ckpt_best_hnsw_evalfile_hotpot_qas_val



