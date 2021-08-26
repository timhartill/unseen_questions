# calculate similarity using cosine similarity over sentence embeddings between train sets specified in mixture and train_file and test sets specified in output dir eval_metrics.json. Must run --create_embeddings before running this function


cd ../code


python cli.py --calc_similarity_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --answer_thresh -100.1 \
        --add_only_missing \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric,strategy_qa,cwwv,atomic,atomic,qasc_dev_facts_selfsvised,qasc_facts_selfsvised,strategy_qa_facts_dev_in_train_selfsvised

