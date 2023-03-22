# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#    --rand_order \


cd ../code

echo "Outputting WORLDTREE DEV..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T32v2_YN_WORLDTREE_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 4 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T33v2_YN_WORLDTREE_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 4 \
    --max_memory_buffer 12 \
    --max_memory -1







