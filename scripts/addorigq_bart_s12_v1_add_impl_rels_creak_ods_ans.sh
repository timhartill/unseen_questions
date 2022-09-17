# Add implicit relations to questions and save as new dataset

cd ../code

python replace_implicit_relation_question_with_original.py \
        --orig_dataset creak_od_ans \
        --iter_dataset creak_fullwiki_bs150_implrel \
        --output_dataset creak_fullwiki_bs150_implrel_origq
        

