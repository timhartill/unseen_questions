#CUDA_VISIBLE_DEVICES=0 
# Stage 1 reader

#    --sent_score_force_zero \

cd ../code

python mdr_train_stage1_nativeamp.py \
    --do_train \
    --prefix stage1_test1_hpqa_hover_fever_nosentforcezero \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file /home/thar011/data/sentences/sent_train.jsonl \
    --predict_file /home/thar011/data/sentences/sent_dev.jsonl \
    --seed 42 \
    --eval-period 250 \
    --max_c_len 512 \
    --max_q_len 70 \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --sp-weight 1.0 \
    --output_dir /large_data/thar011/out/mdr/logs \
    --num_train_epochs 10 \
    --debug \
    --warmup-ratio 0.1





    