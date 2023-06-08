#CUDA_VISIBLE_DEVICES=0 
# RR Model training using merged ratioanles + iter contexts train filea


#    --single_pos_samples \  splits samples with multiple pos_para entries into separate samples each with single pos_para. Effectively skews sampling toward QASC and others with multiple pos_paras. 
#    --no_pos_neg_pairing \  default is put a pos and a neg in same batch. Adding this flag will cause pos and neg samples to be randomly sampled
#    --debug \


cd ../code

python rr_train.py \
    --do_train \
    --prefix RR_test8_mcstrip0.5_WITHsinglepossplit_withsharednormal_additer_addxtrarelrats \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 24 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file $HDATA/data/rationale_reranker/rr_train_rat_iterctxtsv3_merged_extrarelevancerats.jsonl \
    --predict_file $HDATA/data/rationale_reranker/rr_dev_rat_iterctxtsv3_merged_extrarelevancerats.jsonl \
    --seed 42 \
    --eval-period 2000 \
    --max_c_len 512 \
    --max_q_len 70 \
    --mc_strip_prob 0.5 \
    --single_pos_samples \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --output_dir $LDATA/out/mdr/logs \
    --save_prediction rr_dev_predictions.jsonl \
    --num_train_epochs 12 \
    --num_workers_dev 10 \
    --warmup-ratio 0.1





    
