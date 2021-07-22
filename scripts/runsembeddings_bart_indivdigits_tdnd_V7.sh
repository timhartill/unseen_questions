# generate sentence embeddings for calculating similarity between train sets specified in mixture and train_file and test sets specified in output dir eval_metrics.json. Can specify --do_lowercase to force to lowercase before calculating embeddings. Must specify --predict_batch_size


cd ../code

python cli.py --create_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --predict_batch_size 20 \
        --add_only_missing \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric

