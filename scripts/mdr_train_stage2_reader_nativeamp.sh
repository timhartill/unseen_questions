#CUDA_VISIBLE_DEVICES=0 
# Stage 1 reader

#    --sent_score_force_zero \  mask padding in sent preds/labels after loss calc  use sp_weight=1.0 with this option 
#    --sp_percent_thresh \      maximum mean fraction of sentences for a given sp score threshold to take in order for that thresh to be selected. Stage 2 can take all sents so set to 1.0

cd ../code

python mdr_train_stage2_nativeamp.py \
    --do_train \
    --prefix stage2_test3_hpqa_hover_fever_new_sentMASKforcezerospweight1_fevernegfix \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file $HDATA/data/sentences/sent_train.jsonl \
    --predict_file $HDATA/data/sentences/sent_dev.jsonl \
    --seed 42 \
    --eval-period 2000 \
    --max_c_len 512 \
    --max_q_len 70 \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --sp-weight 1.0 \
    --output_dir $LDATA/out/mdr/logs \
    --save_prediction stage2_dev_predictions.jsonl \
    --num_train_epochs 7 \
    --sent_score_force_zero \
    --sp_percent_thresh 1.0 \
    --num_workers_dev 10 \
    --warmup-ratio 0.1





    
