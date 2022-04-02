#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# Orig non-var step: mdr hpqa train/dev files with train/pred bs 100 on 4 GPUS takes 44GB on 1st gpu and 43.1GB on remaining 3. Takes 13.5hrs for 50 epochs. best at 48:
#export CUDA_VISIBLE_DEVICES=1,2,4,5

# bs100 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
# bs75 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
#export CUDA_VISIBLE_DEVICES=2
# bs50 on 1 gpu with 42.5GB free fails (also fails in this config for orig mdr version without _var routines)
# bs24 on one gpu with 42.5GB free (6639mb taken) just fits! (initially gets up to 42GB taken then falls to ~38GB taken as fp16 scaling kicks in) (pred bs 100 here works). Stopped after 26 epochs. Best=epoch 21: MRRS: {'mrr_1': 0.9389304208307683, 'mrr_2': 0.9643570317425275, 'mrr_avg': 0.951643726286648} NOTE loss NAN!: Step 82918 Train loss nan MRR-AVG 95.16 on epoch=21. Last=MRRS: {'mrr_1': 0.9364797357582563, 'mrr_2': 0.9660725050908262, 'mrr_avg': 0.9512761204245412} Step 101763 Train loss nan MRR-AVG 95.13 on epoch=26
# with torch.cuda.amp: bs25 starts on 1 gpu using ~43GB. No gradient overflow msgs or scaler scale msgs and seems to run slightly faster approx 41 mins per epoch..


cd ../code

python mdr_train_mhop_nativeamp.py \
    --do_train \
    --prefix stoptest1hpqa \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 24 \
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
    --reduction sum \
    --retrieve_loss_multiplier 1.0 \
    --stop-drop 0.0 \
    --debug \
    --output_dir /large_data/thar011/out/mdr/logs \
    --num_train_epochs 50 \
    --warmup-ratio 0.1
    
