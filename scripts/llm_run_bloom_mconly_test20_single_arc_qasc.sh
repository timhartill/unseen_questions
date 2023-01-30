# Abduce rationale from LLM and save as tsv format
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third


cd ../code

echo "Generating Rationales for ARC-EASY...."

python llm_infer_output.py \
    --prefix LLM_TEST20_single_arc_easy_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_easy_llm_expl/dev.tsv \
    --predict_file arc_easy/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST21_single_arc_easy_train_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_easy_llm_expl/train.tsv \
    --predict_file arc_easy/train.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST22_single_arc_easy_test_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_easy_llm_expl/test.tsv \
    --predict_file arc_easy/test.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


echo "Generating Rationales for ARC-HARD..."

python llm_infer_output.py \
    --prefix LLM_TEST23_single_arc_hard_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_hard_llm_expl/dev.tsv \
    --predict_file arc_hard/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST24_single_arc_hard_train_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_hard_llm_expl/train.tsv \
    --predict_file arc_hard/train.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST25_single_arc_hard_test_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/arc_hard_llm_expl/test.tsv \
    --predict_file arc_hard/test.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


echo "Generating Rationales for QASC..."

python llm_infer_output.py \
    --prefix LLM_TEST26_single_qasc_dev_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/qasc_llm_expl/dev.tsv \
    --predict_file qasc/dev.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST27_single_qasc_train_on_finalv4 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/qasc_llm_expl/train.tsv \
    --predict_file qasc/train.tsv \
    --template_file generic_csmadeup_weicot_anschoices_choicetextonlysqastyle_addhpqacsqa2_instructionv4.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1280 \
    --debug \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 9 \
    --max_memory -1






