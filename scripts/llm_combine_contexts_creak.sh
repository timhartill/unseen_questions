# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset creak_bigbench_llm_expl \
        --iter_dataset creak_fullwiki_bs150_noimplrel \
        --output_dataset creak_llm_expl_fullwiki_bs150_noimplrel
        

