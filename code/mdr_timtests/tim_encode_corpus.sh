#CUDA_VISIBLE_DEVICES=0,1,2,3 

python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file /data/thar011/gitrepos/multihop_dense_retrieval/data/hpqa_raw_tim/hpqa_abstracts_tim.jsonl \
    --init_checkpoint logs/01-17-2022/tim_-seed16-bsz100-fp16True-lr1e-05-decay0.0-warm0-valbsz100-m0.999-k76800-t1.0/checkpoint_q_best.pt \
    --embed_save_path data/hpqa_tim_momentum \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file /data/thar011/gitrepos/multihop_dense_retrieval/data/hpqa_raw_tim/hpqa_abstracts_tim.jsonl \
    --init_checkpoint logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path data/hpqa_tim_no_momentum \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


