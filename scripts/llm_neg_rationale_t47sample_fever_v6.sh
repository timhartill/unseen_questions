# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#     --query_no_mc \


cd ../code

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T47_FEVER_debug10onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --generate_dev \
    --query_no_context \
    --query_no_nl_mc_options \
    --rand_order \
    --debug \
    --debug_count 10 \
    --max_samples 10 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1




