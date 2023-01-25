# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset csqa2_llm_expl \
        --iter_dataset csqa2_fullwiki_bs150_noimplrel \
        --output_dataset csqa2_llm_expl_fullwiki_bs150_noimplrel
        

