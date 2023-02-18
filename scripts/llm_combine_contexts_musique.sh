# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset musique_qa_full_llm_expl \
        --iter_dataset musique_qa_fullwiki_bs60 \
        --output_dataset musique_qa_full_llm_expl_fullwiki_bs60
        

