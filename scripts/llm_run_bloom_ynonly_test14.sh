# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \

cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST14_SPANYN_csqa2_hpqa_madeup \
    --output_dir $LDATA/out/mdr/logs \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 30 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


