#CUDA_VISIBLE_DEVICES=0 
# Stage 1 reader eval only
# NOTE: Will output to a new output dir determined by prefix and the date, not to the output dir of the ckpt


cd ../code

python mdr_train_stage1_nativeamp.py \
    --do_predict \
    --prefix TESTANSstage1_evalonly_test7_fromtest5_hpqa_hover_fever_fromnosentforcezero_spthreshtune \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --predict_file $HDATA/data/sentences/sent_dev.jsonl \
    --init_checkpoint $LDATA/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --seed 42 \
    --max_c_len 512 \
    --max_q_len 70 \
    --output_dir $LDATA/out/mdr/logs \
    --sp_percent_thresh 0.55 \
    --num_workers_dev 10 \
    --save_prediction stage1_dev_predictions.jsonl




    
