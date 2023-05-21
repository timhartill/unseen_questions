# Add implicit relations to questions and save as new dataset

cd ../code

echo "Doing fp16..."

python combine_llm_retrieved_contexts.py \
        --llm_dataset arc_da_od_ans_llm_expl_svfp16 \
        --iter_dataset arc_da_od_ans_fullwiki_bs150 \
        --output_dataset arc_da_od_ans_llm_expl_fullwiki_bs150_svfp16


python combine_llm_retrieved_contexts.py \
        --llm_dataset commonsenseqa_llm_expl_svfp16 \
        --iter_dataset commonsenseqa_fullwiki_bs150_noimplrel \
        --output_dataset commonsenseqa_llm_expl_fullwiki_bs150_noimplrel_svfp16


python combine_llm_retrieved_contexts.py \
        --llm_dataset iirc_initial_context_llm_expl_svfp16 \
        --iter_dataset iirc_initial_context_fullwiki_bs150 \
        --output_dataset iirc_initial_context_llm_expl_fullwiki_bs150_svfp16


python combine_llm_retrieved_contexts.py \
        --llm_dataset musique_mu_dev_odv2_llm_expl_svfp16 \
        --iter_dataset musique_mu_dev_odv2_fullwiki_bs150 \
        --output_dataset musique_mu_dev_odv2_llm_expl_fullwiki_bs150_svfp16


python combine_llm_retrieved_contexts.py \
        --llm_dataset strategy_qa_bigbench_llm_expl_svfp16 \
        --iter_dataset strategy_qa_bigbench_fullwiki_bs150_noimplrel \
        --output_dataset strategy_qa_bigbench_llm_expl_fullwiki_bs150_noimplrel_svfp16


echo "Now doing int8..."

python combine_llm_retrieved_contexts.py \
        --llm_dataset arc_da_od_ans_llm_expl_svint8 \
        --iter_dataset arc_da_od_ans_fullwiki_bs150 \
        --output_dataset arc_da_od_ans_llm_expl_fullwiki_bs150_svint8


python combine_llm_retrieved_contexts.py \
        --llm_dataset commonsenseqa_llm_expl_svint8 \
        --iter_dataset commonsenseqa_fullwiki_bs150_noimplrel \
        --output_dataset commonsenseqa_llm_expl_fullwiki_bs150_noimplrel_svint8


python combine_llm_retrieved_contexts.py \
        --llm_dataset iirc_initial_context_llm_expl_svint8 \
        --iter_dataset iirc_initial_context_fullwiki_bs150 \
        --output_dataset iirc_initial_context_llm_expl_fullwiki_bs150_svint8


python combine_llm_retrieved_contexts.py \
        --llm_dataset musique_mu_dev_odv2_llm_expl_svfp16 \
        --iter_dataset musique_mu_dev_odv2_fullwiki_bs150 \
        --output_dataset musique_mu_dev_odv2_llm_expl_fullwiki_bs150_svint8


python combine_llm_retrieved_contexts.py \
        --llm_dataset strategy_qa_bigbench_llm_expl_svint8 \
        --iter_dataset strategy_qa_bigbench_fullwiki_bs150_noimplrel \
        --output_dataset strategy_qa_bigbench_llm_expl_fullwiki_bs150_noimplrel_svint8

