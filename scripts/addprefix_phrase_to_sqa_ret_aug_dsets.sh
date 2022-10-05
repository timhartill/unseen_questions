# Add implicit relations to questions and save as new dataset for unseen eval datasets with (sometimes with) implicit questions

cd ../code

echo "Creating new dataset with YN prepended phrase for SQA impl rels..."
python add_prefix_to_questions.py \
        --orig_dataset strategy_qa_bigbench_fullwiki_bs150_implrel \
        --output_dataset strategy_qa_bigbench_fullwiki_bs150_implrel_yn

        
echo "Creating new dataset with YN prepended phrase for SQA impl rels with orig question..."
python add_prefix_to_questions.py \
        --orig_dataset strategy_qa_bigbench_fullwiki_bs150_implrel_origq \
        --output_dataset strategy_qa_bigbench_fullwiki_bs150_implrel_origq_yn


echo "Creating new dataset with YN prepended phrase for SQA without impl rels..."
python add_prefix_to_questions.py \
        --orig_dataset strategy_qa_bigbench_fullwiki_bs150_noimplrel \
        --output_dataset strategy_qa_bigbench_fullwiki_bs150_noimplrel_yn


echo "Finished!"

