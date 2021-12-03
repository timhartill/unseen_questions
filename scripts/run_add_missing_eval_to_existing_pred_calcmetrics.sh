#!/bin/bash
# add information for incremental eval datasets to existing eval_metrics.json files
# (0) Add each new eval or train dataset to the dataset_attribs dict etc in dataset_attributes.py following instns there, then update this script to add any new output dirs 
# (1) If needed run rungen_expl_bartlarge_pick_ckpt150k.sh to generate any new dynamic expl datasets (create q[+mc]e->a from models trained to do q->e)
# (2) run predictions + (3) calc metrics - by running this script  
# (4) Once this done, create sentence embeddings for each new dataset by running runsembeddings_bart_indivdigits_tdnd_V7.sh 
# (5) Then update similarity json file for new datasets to existing model outputs by running runsim_for_sembeddings_bart_indivdigits_tdnd_V7.sh
# /data/thar011/out/unifiedqa_bart_large_v3                             script: runevalall_v2_bartlarge_pick_bestmodel.sh
# /data/thar011/out/unifiedqa_2gputest_from_uqackpt              	 script: runevalall_v2_bartlarge_pick_bestmodel.sh
# /data/thar011/out/unifiedqa_bart_large_v4indiv_digits        script: runevalall_v2_bartlarge_pick_ckpt150k_indivdigits.sh
# /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td     script: runevalall_v2_bartlarge_pick_ckpt150k_indivdigits.sh
# /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd     script: runevalall_v2_bartlarge_pick_ckpt150k_indivdigits.sh
# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd   script: runevalall_v2_bartlarge_pick_ckpt150k_indivdigits.sh
# /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e           script: runevalall_v2_bartlarge_pick_ckpt150k_nnorm10e.sh
# /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd      script: runevalall_v2_bartlarge_pick_ckpt150k_nnorm10e.sh
# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd_200k  script: runevalall_v2_bartlarge_pick_ckpt200k_indivdigits.sh
# /data/thar011/out/unifiedqa_allenai_bartlarge_eval                    script: in this script
# /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos               script: in this script
# /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos                script: in this script
# /data/thar011/out/unifiedqa_t5_base                                   script: in this script
# /data/thar011/out/unifiedqa_t5base_290ksteps                          script: in this script
# numerous bart-large which use best-model-150000 steps checkpoint with max output seq 130 and ssm 1.0 no extra mask char 	script: runevalall_v2_bartlarge_pick_ckpt150k.sh

# Note 1: Add new output dirs to appropriate section after running the eval all script for it to ensure the output dir is kept updated with new eval datasets..
# Note 2: Run this from scripts subdirectory..


echo "Update older existing BART outputs which use best-model without a particular checkpoint..."

for out in "/data/thar011/out/unifiedqa_bart_large_v3" "/data/thar011/out/unifiedqa_2gputest_from_uqackpt"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_bartlarge_pick_bestmodel.sh $out
done


echo "Update existing BART outputs which use best-model-150000 steps checkpoint and indiv_digits setting..."

for out in "/data/thar011/out/unifiedqa_bart_large_v4indiv_digits" "/data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td" "/data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd" "/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_bartlarge_pick_ckpt150k_indivdigits.sh $out
done


#echo "Update existing BART outputs which use best-model-200000 steps indiv_digits checkpoint..."

#for out in "/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd"
#do
#    echo "Updating eval for $out ..."
#    bash runevalall_v2_bartlarge_pick_ckpt200k_indivdigits.sh $out
#done


echo "Update existing BART outputs which use best-model-150000 steps checkpoint with max output seq 130 and ssm 1.0 no extra mask char ..."

for out in "/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v3_no_facts" "/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v2_dev_in_train" "/data/thar011/out/unifiedqa_bart_large_s5_v2_sqafacts_dev_in_train_only" "/data/thar011/out/unifiedqa_bart_large_s3_v1_cwwv" "/data/thar011/out/unifiedqa_bart_large_s3_v2_cwwv_atomic" "/data/thar011/out/unifiedqa_bart_large_s4_v1_qasc_dev_facts" "/data/thar011/out/unifiedqa_bart_large_s4_v2_cwwv_premask_atomic_premask" "/data/thar011/out/unifiedqa_bart_large_s4_v3_cwwv_ssvise_atomic_ssvise" "/data/thar011/out/unifiedqa_bart_large_s6_v3_musique_qa_only" "/data/thar011/out/unifiedqa_bart_large_s6_v4_musique_qa_plus_all_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v5_musique_qa_decomp_ans_plus_all_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v6_musique_qa_paras_plus_all_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v7_musique_qa_decomp_ans_only" "/data/thar011/out/unifiedqa_bart_large_s6_v8_musique_qa_decomp_ans_plus_new_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v9_musique_qa_plus_qa_decomp_ans_plus_all_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v10_musique_qa_plus_qa_decomp_ans_plus_new_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v11_musique_qa_paras_plus_qa_paras_decomp_ans_plus_new_decomps" "/data/thar011/out/unifiedqa_bart_large_s6_v12_musique_qa_paras_plus_qa_paras_decomp_ans" "/data/thar011/out/unifiedqa_bart_large_s6_v13_musique_qa_plus_qa_decomp_ans_full_plus_new_decomps_full" "/data/thar011/out/unifiedqa_bart_large_s6_v14_musique_qa_paras_plus_qa_paras_decomp_ans_full" "/data/thar011/out/unifiedqa_bart_large_s7_v1_uqa_sqa_mqa_expl_mswq_explans_msw"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_bartlarge_pick_ckpt150k.sh $out
done


echo "Update existing BART outputs which use best-model-150000 steps checkpoint and --norm_numbers --norm_10e settings..."

for out in "/data/thar011/out/unifiedqa_bart_large_v12_nnorm10e" "/data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_bartlarge_pick_ckpt150k_nnorm10e.sh $out
done


cd ../code

echo "Updating eval for unifiedqa_allenai_bartlarge_eval ..."
python cli.py --output_dir /data/thar011/out/unifiedqa_allenai_bartlarge_eval \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --checkpoint /data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt \
        --model facebook/bart-large \
        --do_predict_all --calc_metrics_all --add_only_missing



echo "Updating eval for /data/thar011/out/unifiedqa_t5_base ..."
python cli.py --output_dir /data/thar011/out/unifiedqa_t5_base \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model t5-base \
        --do_predict_all --calc_metrics_all --add_only_missing \
        --strip_single_quotes



echo "Running eval for /data/thar011/out/unifiedqa_t5base_290ksteps ..."
python cli.py --output_dir /data/thar011/out/unifiedqa_t5base_290ksteps \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model t5-base \
        --checkpoint /data/thar011/out/unifiedqa_t5_base/best-model-290000.pt \
        --do_predict_all --calc_metrics_all --add_only_missing \
        --strip_single_quotes



echo "Updating eval for /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos..."
python cli.py --output_dir /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model allenai/unifiedqa-t5-base \
        --do_predict_all --calc_metrics_all --add_only_missing \
        --strip_single_quotes



echo "Updating eval for /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos..."
python cli.py --output_dir /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model allenai/unifiedqa-t5-large \
        --do_predict_all --calc_metrics_all --add_only_missing \
        --strip_single_quotes



echo Finished!

