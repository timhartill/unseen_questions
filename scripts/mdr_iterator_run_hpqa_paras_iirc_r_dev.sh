# Run iterator ie retrieve/stage1/stage2 on previously trained models/encoded corpus
# Will build hnsw index if this option selected and index doesnt exist

# iirc train


cd ../code



echo "BUILD CONTEXT FOR IIRC INITIAL CONTEXT DEV..."
python mdr_searchers.py \
    --prefix ITER_fullwiki_us_iircictrain_test90_b60_h4_hpqahovnqmubs250_mom \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/iirc_initial_context_fullwiki_bs60/dev.tsv \
    --predict_file $UQA_DIR/iirc_initial_context/dev.tsv \
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




