#CUDA_VISIBLE_DEVICES=0 
# EVAL RR Models on TruthfulQA MC1 and SQA Rat Rel with rr model test 8 (extra relevance samples added)


cd ../code

python rr_eval_truthfulqa.py \
    --prefix TRUTHFULQA_MC1_TEST4_RR_on_rrtest8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test8_mcstrip0.5_WITHsinglepossplit_withsharednormal_additer_addxtrarelrats-06-08-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposTrue-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --model_type rr \
    --predict_file $HDATA/data/truthfulqa/mc_task.json \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_eval_rationalerelevance_sqafacts.py \
    --prefix RRSQA_RAT_REL_TEST4_RR_on_rrtest8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test8_mcstrip0.5_WITHsinglepossplit_withsharednormal_additer_addxtrarelrats-06-08-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposTrue-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --model_type rr \
    --predict_file $UQA_DIR/strategy_qa_bigbench_expl_ans/dev.tsv \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10



