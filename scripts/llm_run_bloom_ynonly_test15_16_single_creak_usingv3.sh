# Abduce rationale from LLM and save as tsv format
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --resume_dir $LDATA/out/mdr/logs/LLM_TEST7_MCONLY_csqa_dev_all_on_finalv4-01-13-2023-LLM-bigscience-bloom- \

cd ../code

echo "Generating Rationales for CREAK dev..."
python llm_infer_output.py \
    --prefix LLM_TEST15_single_creak_dev_on_ynv3 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/creak_llm_expl/dev.tsv \
    --predict_file creak_od_ans/dev.tsv \
    --template_file generic_csqa2_ynmadeup_weicot_withinstructionv3.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 6 \
    --max_memory -1


echo "Generating Rationales for CREAK train..."
python llm_infer_output.py \
    --prefix LLM_TEST16_single_creak_train_on_ynv3 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/creak_llm_expl/train.tsv \
    --predict_file creak_od_ans/train.tsv \
    --template_file generic_csqa2_ynmadeup_weicot_withinstructionv3.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 6 \
    --max_memory -1




