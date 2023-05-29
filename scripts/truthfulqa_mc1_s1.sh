#CUDA_VISIBLE_DEVICES=0 
# EVAL RR Model on TruthfulQA MC1
# here using rr model: RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5


cd ../code

python rr_eval_truthfulqa.py \
    --prefix TRUTHFULQA_MC1_TEST2_S1_Para_ReRanker \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/stage1_test5_hpqa_hover_fever_new_sentMASKforcezerospweight1_fullevalmetrics-05-29-2022-rstage1-seed42-bsz12-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --model_type s1 \
    --predict_file $HDATA/data/truthfulqa/mc_task.json \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10



