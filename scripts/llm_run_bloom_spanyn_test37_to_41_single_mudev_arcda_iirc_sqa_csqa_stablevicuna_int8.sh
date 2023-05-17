# Abduce rationale from StableVicuna int8
# echo $UDATA/ckpts/stable-vicuna-13b


cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST37_SVINT8_SPANYN_musique_mu_dev_odv2_dev_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_odv2_llm_expl_svint8/dev.tsv \
    --predict_file musique_mu_dev_odv2/dev.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST38_SVINT8_SPANYN_arc_da_od_ans_test_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_da_od_ans_llm_expl_svint8/test.tsv \
    --predict_file arc_da_od_ans/test.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST39_SVINT8_SPANYN_iirc_initial_context_test_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/iirc_initial_context_llm_expl_svint8/test.tsv \
    --predict_file iirc_initial_context/test.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST40_SVINT8_single_sqa_on_ynv3 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/strategy_qa_bigbench_llm_expl_svint8/dev.tsv \
    --predict_file strategy_qa_bigbench_od_ans/dev.tsv \
    --template_file generic_csqa2_ynmadeup_weicot_withinstructionv3_stablevicuna.txt \
    --model_name $UDATA/ckpts/stable-vicuna-13b \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST41_SVINT8_single_csqa_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/commonsenseqa_llm_expl_svint8/dev.tsv \
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


