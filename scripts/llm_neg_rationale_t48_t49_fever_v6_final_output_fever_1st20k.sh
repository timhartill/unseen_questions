# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#    --rand_order \


cd ../code

echo "Outputting FEVER..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T48_FEVER_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file fever_expl_ans/dev.tsv \
    --template_file neg_rationale_fever_v6_onlyneg.txt \
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
    --prefix LLM_NEGRAT_T49_FEVER_TRAIN_onv6_sample_1st20K \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file fever_expl_ans/train.tsv \
    --template_file neg_rationale_fever_v6_onlyneg.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples 20000 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 4 \
    --max_memory_buffer 12 \
    --max_memory -1





