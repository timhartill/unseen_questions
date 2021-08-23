# generate sentence embeddings for calculating similarity between train sets specified in mixture and train_file and test sets specified in output dir 
# specifying --use_question_only creates the question embedding from the actual question part of the input only (ie the string up to the first \\n) NOT USED


cd ../code


python cli.py --create_embeddings --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --verbose \
        --predict_batch_size 20 \
        --add_only_missing \
        --use_question_only \
        --mixture unifiedqa,synthetic_textual,synthetic_numeric,strategy_qa,cwwv,atomic

