# Abduce DROP rationale + answeronly from StableVicuna int8 + fp16 redo with max_seq_len_in 1472
# echo $UDATA/ckpts/stable-vicuna-13b


cd ../code

echo "SV FP16..."

python llm_infer_output.py \
    --prefix LLM_TEST62_SVFP16_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/drop_llm_expl_svfp16/dev.tsv \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --fp16 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST63_SVFP16_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_stablevicuna_answeronly.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --fp16 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "SV INT8 only..."


python llm_infer_output.py \
    --prefix LLM_TEST64_SVINT8_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/drop_llm_expl_svint8/dev.tsv \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
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
    --prefix LLM_TEST65_SVINT8_single_drop_dev_on_drop3shot \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file drop/dev.tsv \
    --template_file generic_drop3shot_withinstruction_stablevicuna_answeronly.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1

    
