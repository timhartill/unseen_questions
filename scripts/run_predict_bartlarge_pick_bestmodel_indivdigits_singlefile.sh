#!/bin/bash

# run predictions on single train | dev | test file and write them out to output_dir in form prefix_predictions.json


# WARNING: SET --prefix to something unique to avoid overwriting any existing prediction files in --output_dir

cd ../code

echo "Running prediction using ind digit tokenization for best model in output_dir ..."

python cli.py --output_dir $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2 \
        --prefix train_creak_fullwiki_bs150_implrel_origq_ \
        --predict_file $UDATA/data/unifiedqa/creak_fullwiki_bs150_implrel_origq/train.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model facebook/bart-large \
        --indiv_digits \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --max_output_length 130 \
        --do_predict



