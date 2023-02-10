# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset iirc_initial_context_llm_expl \
        --iter_dataset iirc_initial_context_fullwiki_bs150 \
        --output_dataset iirc_initial_context_llm_expl_fullwiki_bs150
        

