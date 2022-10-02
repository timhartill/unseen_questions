# Add implicit relations to questions and save as new dataset for unseen eval datasets with (sometimes with) implicit questions

cd ../code

echo "Creating new dataset without impl rels in question for SQA..."
python replace_implicit_relation_question_with_original.py \
        --orig_dataset strategy_qa_bigbench_od_ans \
        --iter_dataset strategy_qa_bigbench_fullwiki_bs150_implrel \
        --output_dataset strategy_qa_bigbench_fullwiki_bs150_implrel_origq

        
echo "Creating new dataset without impl rels in question for CommonsenseQA..."
python replace_implicit_relation_question_with_original.py \
        --orig_dataset commonsenseqa \
        --iter_dataset commonsenseqa_fullwiki_bs150_implrel \
        --output_dataset commonsenseqa_fullwiki_bs150_implrel_origq




