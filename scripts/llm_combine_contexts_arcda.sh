# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset arc_da_od_ans_llm_expl \
        --iter_dataset arc_da_od_ans_fullwiki_bs150 \
        --output_dataset arc_da_od_ans_llm_expl_fullwiki_bs150
        

