# Abduce rationale from StableVicuna int8
# echo $UDATA/ckpts/stable-vicuna-13b


cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST47_SVINT8_single_csqa_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/commonsenseqa_llm_expl_svint8_v2/dev.tsv \
    --predict_file commonsenseqa/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST48_SVFP16_single_csqa_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/commonsenseqa_llm_expl_svfp16_v2/dev.tsv \
    --predict_file commonsenseqa/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --query_no_nl_mc_options \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --fp16 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1



