# Calculate similarity using cosine similarity over sentence embeddings between train sets specified in mixture and train_file and test sets specified in output dir eval_metrics.json. 
# Similarity results file saved to ../outputdir/eval_test_train_similarities_semb_thresh-100.1.json - manually back up this file before running this script unless recreating from scratch...
# Must run --create_embeddings before running this function using eg runsembeddings_bart_indivdigits_tdnd_V7.sh
# add the --add_only_missing flag to only add new training or eval datasets that don't already exist in the similarity file vs doing everything from scratch
# add the --reformat_question_ssvise flag to use eval samples reformatted to self-supervised-like format for comparison against self supervised training datasets

cd ../code


python cli.py --calc_similarity_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --answer_thresh -100.1 \
        --add_only_missing \
        --reformat_question_ssvise \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric,strategy_qa,cwwv,atomic,qasc_dev_facts_selfsvised,qasc_facts_selfsvised,strategy_qa_facts_selfsvised,strategy_qa_facts_dev_in_train_selfsvised,musique_qa,musique_qa_paras,musique_decomp_all_dev_in_train,musique_decomp_new_dev_in_train,musique_decomp_train,musique_mu_dev_decomp,musique_qa_full,musique_qa_paras_full


