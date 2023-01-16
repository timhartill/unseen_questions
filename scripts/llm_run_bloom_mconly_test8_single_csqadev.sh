# Abduce rationale from LLM and save as tsv format
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third


cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST9_single_csqa_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/commonsenseqa_llm_expl/dev.tsv \
    --predict_file commonsenseqa/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --resume_dir $LDATA/out/mdr/logs/LLM_TEST7_MCONLY_csqa_dev_all_on_finalv4-01-13-2023-LLM-bigscience-bloom- \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1152 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 6 \
    --max_memory -1



