# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset hover_llm_expl \
        --iter_dataset hover_fullwiki_bs60 \
        --output_dataset hover_llm_expl_fullwiki_bs60
        

