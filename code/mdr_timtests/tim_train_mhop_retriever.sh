#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export CUDA_VISIBLE_DEVICES=1,2,4,5

python scripts/train_mhop.py \
    --do_train \
    --prefix tim_ \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 100 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file data/hotpot/hotpot_train_with_neg_v0.json \
    --predict_file data/hotpot/hotpot_dev_with_neg_v0.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --shared-encoder \
    --warmup-ratio 0.1
    
