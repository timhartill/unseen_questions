# eval retrieve on previouly trained model

# after training (base or base+momentum), run mdr_encode_datasets then run this sh AFTER setting:
# index_path, corpus_dict to location when mdr_encode_datasets output to
# model_path, output_dir to location of the ckpt dir that mdr_encode_datasets was run on

# MDR: Per issue 8: We used a large beam size for the leaderboard submission. We used beam size 200 and top250 passage sequences for answer extraction.

# --gpu_model: puts the model on single gpu (0 of visible gpus). Without --gpu_model keeps model on cpu
# --gpu_faiss: puts the faiss index on gpu. Without keeps on cpu.

# with bs 100 beam size 1, topk 1 faiss on cpu & model on gpu (takes ~4gb and neglible impact on other jon running there since most time is with faiss) takes ~2.5 hrs

# bs 100 beam 1 topk 1 Uses 21.2GB on one GPU: Approx 18.4GB of this is the FAISS index over ~5M paras (on disk wiki_index.npy=15GB and wiki_id2doc.json = 3.1GB)
# bs 100 beam 2 topk 2: Uses 23.4GB GPU - only slightly more than beam/topk 1
# bs 100 beam 4 topk 4: Uses 27.8GB GPU - 8GB more than beam/topk 1
# bs 100 beam 20 topk 20: GPU OOM
# bs 10  beam 20 topk 20: 23.1GB GPU Used. Inference time goes from 3mins@topk1->15mins
# bs 10  beam 50 topk 50: 29.7GB GPU used. Inference time: 25 mins

cd ../code

python mdr_eval_mhop_retrieval_nativeamp.py \
    --eval_data /home/thar011/data/mdr/hotpot/hotpot_qas_val.json \
    --index_path /home/thar011/data/mdr/hpqa_varinitialtest2_03-26_bs24_no_momentum/index.npy \
    --corpus_dict /home/thar011/data/mdr/hpqa_varinitialtest2_03-26_bs24_no_momentum/id2doc.json \
    --model_path /large_data/thar011/out/mdr/logs/03-26-2022/varinitialtest2_-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue/checkpoint_best.pt \
    --batch_size 100 \
    --beam_size 1 \
    --topk 1 \
    --model_name roberta-base \
    --gpu_model \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --use_var_versions \
    --fp16 \
    --output_dir /large_data/thar011/out/mdr/logs/03-26-2022/varinitialtest2_-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue



