#!/bin/bash
# Improved t5-large eval script using --do_predict_all and --calc_metrics_all
# use for t5-large on base uqa datasets when not selecting a particular checkpoint
# run predictions on multiple dev and test sets using test set where available, dev set otherwise

cd ../code

echo "Running T5-large Eval using ind digit tokenization for best model in $1 ..."

python cli.py --output_dir $1 \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --predict_batch_size 30 \
        --do_lowercase \
        --verbose \
        --model t5-large \
        --strip_single_quotes \
        --indiv_digits \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --max_output_length 130 \
        --fp16 \
        --do_predict_all --calc_metrics_all --add_only_missing



