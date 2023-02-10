# Add implicit relations to questions and save as new dataset

cd ../code

python combine_llm_retrieved_contexts.py \
        --llm_dataset musique_mu_dev_odv2_llm_expl \
        --iter_dataset musique_mu_dev_odv2_fullwiki_bs150 \
        --output_dataset musique_mu_dev_odv2_llm_expl_fullwiki_bs150
        

