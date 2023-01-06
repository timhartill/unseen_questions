#CUDA_VISIBLE_DEVICES=0 
# Stage 1 reader

#    --sent_score_force_zero \

cd ../code

python mdr_train_stage1_nativeamp.py \
    --do_train \
    --prefix stage1_sciencoderv2_test1 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file $HDATA/data/scifact/data/scifact_orig_train_with_neg_and_sent_annots.jsonl \
    --predict_file $HDATA/data/scifact/data/scifact_orig_dev_with_neg_and_sent_annots.jsonl \
    --init_checkpoint $LDATA/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --seed 42 \
    --eval-period -1 \
    --max_c_len 512 \
    --max_q_len 70 \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --sp-weight 1.0 \
    --output_dir $LDATA/out/mdr/logs \
    --save_prediction stage1_dev_predictions.jsonl \
    --num_train_epochs 7 \
    --sent_score_force_zero \
    --sp_percent_thresh 0.55 \
    --num_workers_dev 10 \
    --debug \
    --warmup-ratio 0.1





    
