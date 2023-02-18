# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset nq_open_llm_expl \
        --iter_dataset nq_open_fullwiki_bs60 \
        --output_dataset nq_open_llm_expl_fullwiki_bs60
        

