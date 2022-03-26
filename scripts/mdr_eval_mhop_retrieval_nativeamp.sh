# eval retrieve on previouly trained model
# Per issue 8: We used a large beam size for the leaderboard submission. We used beam size 200 and top250 passage sequences for answer extraction.

# bs 100 beam 1 topk 1 Uses 21.2GB on one GPU: Approx 18.4GB of this is the FAISS index over ~5M paras (on disk wiki_index.npy=15GB and wiki_id2doc.json = 3.1GB)
# bs 100 beam 2 topk 2: Uses 23.4GB GPU - only slightly more than beam/topk 1
# bs 100 beam 4 topk 4: Uses 27.8GB GPU - 8GB more than beam/topk 1
# bs 100 beam 20 topk 20: GPU OOM
# bs 10  beam 20 topk 20: 23.1GB GPU Used. Inference time goes from 3mins@topk1->15mins
# bs 10  beam 50 topk 50: 29.7GB GPU used. Inference time: 25 mins

cd ../code

python mdr_eval_mhop_retrieval_nativeamp.py \
    --eval_data /home/thar011/data/mdr/hotpot/hotpot_qas_val.json \
    --index_path /home/thar011/data/mdr/hpqa_novar1gpu_03-24_bs24_no_momentum/index.npy \
    --corpus_dict /home/thar011/data/mdr/hpqa_novar1gpu_03-24_bs24_no_momentum/id2doc.json \
    --model_path /large_data/thar011/out/mdr/logs/03-24-2022/novar1gpu_-mom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varFalse/checkpoint_best.pt \
    --batch_size 100 \
    --beam_size 1 \
    --topk 1 \
    --model_name roberta-base \
    --gpu \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --use_var_versions \
    --fp16 \
    --output_dir /large_data/thar011/out/mdr/logs/03-24-2022/novar1gpu_-mom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varFalse



