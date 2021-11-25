#!/bin/bash
# Improved bart-large eval script using --do_predict_all and --calc_metrics_all
# use for bart-large on base uqa datasets when not selecting a particular checkpoint
# run predictions on multiple dev and test sets using test set where available, dev set otherwise

cd ../code

echo "Running Eval for best model in $1 ..."

python cli.py --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model facebook/bart-large \
        --do_predict_all --calc_metrics_all --add_only_missing


