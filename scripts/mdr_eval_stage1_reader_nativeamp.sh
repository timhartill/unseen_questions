#CUDA_VISIBLE_DEVICES=0 
# Stage 1 reader eval only
# NOTE: Will output to a new output dir determined by prefix and the date, not to the output dir of the ckpt


cd ../code

python mdr_train_stage1_nativeamp.py \
    --do_predict \
    --prefix stage1_evalonly_test2_hpqa_hover_fever_nosentforcezero \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --predict_file /home/thar011/data/sentences/sent_dev.jsonl \
    --init_checkpoint /large_data/thar011/out/mdr/logs/stage1_test1_hpqa_hover_fever_nosentforcezero-05-24-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --seed 42 \
    --max_c_len 512 \
    --max_q_len 70 \
    --output_dir /large_data/thar011/out/mdr/logs \
    --save_prediction stage1_dev_predictions.jsonl




    
