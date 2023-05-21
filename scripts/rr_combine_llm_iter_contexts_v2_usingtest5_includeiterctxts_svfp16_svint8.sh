#CUDA_VISIBLE_DEVICES=0 
# Score llm & iter rationales/contexts using rr model and output tsv datasets based on rr scores  Stablevicuna fp16 and int8
# here using rr model: RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5



cd ../code

echo "Doing SV FP16..."

python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST6_CSQA_svfp16 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/commonsenseqa_llm_expl_svfp16/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_csqa_test66_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset commonsenseqa_svfp16 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST7_SQA_svfp16 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/strategy_qa_bigbench_llm_expl_svfp16/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_sqabb_test64_b150_h4_hpqahovnqmubs250_mom-09-30-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset strategy_qa_bigbench_svfp16 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST8_ARCDA_svfp16 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/arc_da_od_ans_llm_expl_svfp16/test.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_arcdatst_test70_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset arc_da_od_ans_svfp16 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST9_IIRC_svfp16 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/iirc_initial_context_llm_expl_svfp16/test.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_iircictst_test69_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset iirc_initial_context_svfp16 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST10_MU_DEV_svfp16 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/musique_mu_dev_odv2_llm_expl_svfp16/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_mudev_test71_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset musique_mu_dev_odv2_svfp16 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


echo "Now doing INT8..."

python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST11_CSQA_svint8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/commonsenseqa_llm_expl_svint8/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_csqa_test66_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset commonsenseqa_svint8 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST12_SQA_svint8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/strategy_qa_bigbench_llm_expl_svint8/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_sqabb_test64_b150_h4_hpqahovnqmubs250_mom-09-30-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset strategy_qa_bigbench_svint8 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST13_ARCDA_svint8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/arc_da_od_ans_llm_expl_svint8/test.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_arcdatst_test70_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset arc_da_od_ans_svint8 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST14_IIRC_svint8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/iirc_initial_context_llm_expl_svint8/test.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_iircictst_test69_b150_h4_hpqahovnqmubs250_mom-10-01-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset iirc_initial_context_svint8 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10


python rr_merge_iter_output_eval.py \
    --prefix RR_LLM_ITER_MERGEV2_TEST15_MU_DEV_svint8 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --init_checkpoint $LDATA/out/mdr/logs/RR_test5_mcstrip0.5_notsinglepossplit_withsharednormal_additer-05-01-2023-RR-seed42-bsz24-fp16True-lr5e-05-decay0.0-warm0.1-valbsz100-ga8-nopairFalse-singleposFalse-mcstrip0.5/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --llm_file $UDATA/data/unifiedqa/musique_mu_dev_odv2_llm_expl_svint8/dev.tsv \
    --iter_file $LDATA/out/mdr/logs/ITER_fullwiki_us_mudev_test71_b150_h4_hpqahovnqmubs250_mom-10-02-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl \
    --base_dataset musique_mu_dev_odv2_svint8 \
    --max_c_len 512 \
    --max_q_len 70 \
    --num_workers_dev 10




