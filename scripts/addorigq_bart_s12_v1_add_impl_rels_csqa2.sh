# Add implicit relations to questions and save as new dataset

cd ../code

python replace_implicit_relation_question_with_original.py \
        --orig_dataset csqa2 \
        --iter_dataset csqa2_fullwiki_bs150_implrel \
        --output_dataset csqa2_fullwiki_bs150_implrel_origq
        

