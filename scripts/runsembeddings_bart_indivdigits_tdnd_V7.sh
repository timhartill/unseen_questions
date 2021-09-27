# generate sentence embeddings for calculating similarity between train sets specified in mixture and train_file and test sets specified in output dir eval_metrics.json. Can specify --do_lowercase to force to lowercase before calculating embeddings. Must specify --predict_batch_size


cd ../code
echo "Creating sentence embeddings with native question format format..."
python cli.py --create_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --predict_batch_size 20 \
        --add_only_missing \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric,strategy_qa,cwwv,atomic,qasc_dev_facts_selfsvised,qasc_facts_selfsvised,strategy_qa_facts_selfsvised,strategy_qa_facts_dev_in_train_selfsvised,musique_qa,musique_qa_paras,musique_decomp_all_dev_in_train,musique_decomp_new_dev_in_train,musique_decomp_train,musique_mu_dev_decomp


echo "Creating sentence embeddings with qa reformatted into self-supervised format..."
python cli.py --create_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --predict_batch_size 20 \
        --add_only_missing \
        --reformat_question_ssvise \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric,strategy_qa,cwwv,atomic,qasc_dev_facts_selfsvised,qasc_facts_selfsvised,strategy_qa_facts_selfsvised,strategy_qa_facts_dev_in_train_selfsvised,musique_qa,musique_qa_paras,musique_decomp_all_dev_in_train,musique_decomp_new_dev_in_train,musique_decomp_train,musique_mu_dev_decomp


