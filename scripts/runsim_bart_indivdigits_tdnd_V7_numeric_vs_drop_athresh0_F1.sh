# generate data file for calculating similarity between train sets specified in mixture and train_file and test sets specified in output dir eval_metrics.json

cd ../code

python cli.py --calc_similarity_numeric --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --append_another_bos --do_lowercase \
        --verbose \
        --answer_thresh 0.0 \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric

