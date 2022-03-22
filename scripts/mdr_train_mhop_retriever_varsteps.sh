#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# Orig non-var step: mdr hpqa train/dev files with train/pred bs 100 on 4 GPUS takes 44GB on 1st gpu and 43.1GB on remaining 3. Takes 13.5hrs for 50 epochs. best at 48:
#export CUDA_VISIBLE_DEVICES=1,2,4,5

# bs100 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
# bs75 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
#export CUDA_VISIBLE_DEVICES=1,3,5,2,6

cd ../code

python mdr_train_mhop.py \
    --do_train \
    --prefix varinitialtest_ \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 75 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file /home/thar011/data/mdr/hotpot/hotpot_train_with_neg_v0.json \
    --predict_file /home/thar011/data/mdr/hotpot/hotpot_dev_with_neg_v0.json \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --shared-encoder \
    --use_var_versions \
    --output_dir /large_data/thar011/out/mdr/logs \
    --num_train_epochs 50 \
    --warmup-ratio 0.1
    
