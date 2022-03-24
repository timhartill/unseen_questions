# Continue training of q_encoder with momentum
# Original non-varstep version:bs 100 On 4 GPUS GPU loading is 18GB each and training/eval 1 epoch takes ~12mins
#varstep version bs 50 1 GPU takes ~36GB. 1 epoch approx 30mins

cd ../code

python mdr_train_mhop.py \
    --do_train \
    --prefix mominitialvar_ \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 50 \
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
    --gradient_accumulation_steps 1 \
    --use_var_versions \
    --output_dir /large_data/thar011/out/mdr/logs \
    --momentum \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init_retriever /large_data/thar011/out/mdr/logs/03-23-2022/varinitialtest_-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --num_train_epochs 50 \
    --warmup-ratio 0.1
    
