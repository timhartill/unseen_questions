# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset commonsenseqa_llm_expl \
        --iter_dataset commonsenseqa_fullwiki_bs150_noimplrel \
        --output_dataset commonsenseqa_llm_expl_fullwiki_bs150_noimplrel
        

