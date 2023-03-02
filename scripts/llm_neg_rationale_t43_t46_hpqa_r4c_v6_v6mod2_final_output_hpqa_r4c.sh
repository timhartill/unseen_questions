# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#    --rand_order \


cd ../code

echo "Outputting HPQA_R4C..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T43_HPQA_R4C_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hpqa_r4c_expl_ans_0/dev.tsv \
    --template_file neg_rationale_hpqa_r4c_v6_onlyneg_theanswermustbe.txt \
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
    --prefix LLM_NEGRAT_T44_HPQA_R4C_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hpqa_r4c_expl_ans_0/dev.tsv \
    --template_file neg_rationale_hpqa_r4c_v6_2_onlyneg_theanswermustbe_multisentence.txt \
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
    --prefix LLM_NEGRAT_T45_HPQA_R4C_TRAIN_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hpqa_r4c_expl_ans_0/train.tsv \
    --template_file neg_rationale_hpqa_r4c_v6_onlyneg_theanswermustbe.txt \
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
    --prefix LLM_NEGRAT_T46_HPQA_R4C_TRAIN_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hpqa_r4c_expl_ans_0/train.tsv \
    --template_file neg_rationale_hpqa_r4c_v6_2_onlyneg_theanswermustbe_multisentence.txt \
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




