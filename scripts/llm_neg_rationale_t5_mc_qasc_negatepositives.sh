# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#     --query_no_mc \


cd ../code

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T5_MCFOCUS_QASC_ONLY_debug10onv8 \
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
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


