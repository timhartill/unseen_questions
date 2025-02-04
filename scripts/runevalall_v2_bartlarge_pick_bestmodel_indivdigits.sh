#!/bin/bash
# Improved bart-large eval script using --do_predict_all and --calc_metrics_all
# use for bart-large on base uqa datasets when not selecting a particular checkpoint
# run predictions on multiple dev and test sets using test set where available, dev set otherwise

#         --fp16 \  Almost the same as tf32 on inference

cd ../code

echo "Running Eval using ind digit tokenization for best model in $1 ..."

python cli.py --output_dir $1 \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model facebook/bart-large \
        --indiv_digits \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --max_output_length 130 \
        --do_predict_all --calc_metrics_all --add_only_missing



