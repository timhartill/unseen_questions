# Abduce DROP rationale + answeronly from Bloom int8 with max_seq_len_in 1472
# echo $UDATA/ckpts/stable-vicuna-13b


cd ../code



echo "BLOOM INT8 ..."


python llm_infer_output.py \
    --prefix LLM_TEST66_bloomINT8_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/drop_llm_expl/dev.tsv \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_bloom.txt \
    --model_name bigscience/bloom \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1



python llm_infer_output.py \
    --prefix LLM_TEST67_bloomINT8_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_bloom_answeronly.txt \
    --model_name bigscience/bloom \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1

