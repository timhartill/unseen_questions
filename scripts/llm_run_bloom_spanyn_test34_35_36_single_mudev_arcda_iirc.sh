# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \

cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST34_SPANYN_musique_mu_dev_odv2_dev_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_odv2_llm_expl/dev.tsv \
    --predict_file musique_mu_dev_odv2/dev.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST35_SPANYN_arc_da_od_ans_test_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_da_od_ans_llm_expl/test.tsv \
    --predict_file arc_da_od_ans/test.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST36_SPANYN_iirc_initial_context_test_using_muv2 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/iirc_initial_context_llm_expl/test.tsv \
    --predict_file iirc_initial_context/test.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


