#!/bin/bash
# Generate explanations for selected datasets, add the explanations as context and write then out the a new dataset subdir
# Creates new explanation datasets for existing eval datasets specified in dataset_attributes.py create_datasets_dynamic
# with dataset subdir UQA_DIR/sourcedataset__dyn_expl_ans_explgeneratormodeloutputdir_timestamp and filename either dev.tsv or test.tsv
# usage: run with /data/thar011/out/unifiedqa_bart_large_s7_v1_uqa_sqa_mqa_expl_mswq_explans_msw
# or: /data/thar011/out/unifiedqa_bart_large_s7_v2_expl_mswq

cd ../code

echo "Running Explanation dataset generation using best model at 150k steps in $1 ..."

python cli.py --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --max_output_length 130 \
        --ssm_prob 1.0 \
        --add_mask_char NONE \
        --gen_explanations_all



