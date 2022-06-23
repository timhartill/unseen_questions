#CUDA_VISIBLE_DEVICES=0 
# Stage 2 reader eval only
# NOTE: Will output to a new output dir determined by prefix and the date, not to the output dir of the ckpt
# Eval based on tuned sp thresh and tuned ev thresh
# pytorch too many open files error when num_workers_dev > 0 fixed by changing multiprocessing sharing strategy


cd ../code

python mdr_train_stage2_nativeamp.py \
    --do_predict \
    --prefix TESTANSstage2_evalonly_test2_fromtest0_hpqa_hover_fever_fromnosentforcezero_spthreshtune_evthreshtune \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --predict_file /home/thar011/data/sentences/sent_dev.jsonl \
    --init_checkpoint "/large_data/thar011/out/mdr/logs/stage2_test0_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-06-08-2022-rstage2-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best_BatStep 207999 GlobStep26000 Trainloss2.99 SP_EM64.00 epoch5 para_acc0.8416.pt" \
    --seed 42 \
    --max_c_len 512 \
    --max_q_len 70 \
    --output_dir /large_data/thar011/out/mdr/logs \
    --sp_percent_thresh 1.0 \
    --num_workers_dev 10 \
    --save_prediction stage2_dev_predictions.jsonl




    
