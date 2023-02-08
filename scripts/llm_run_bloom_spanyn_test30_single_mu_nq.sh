# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \

cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST30_SPANYN_musique_qa_full_dev_using_muv2_all382 \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_qa_full_llm_expl/dev.tsv \
    --predict_file musique_qa_full/dev.tsv \
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
    --prefix LLM_TEST31_SPANYN_musique_qa_full_train_using_muv2_10krandord \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_qa_full_llm_expl/train.tsv \
    --predict_file musique_qa_full/train.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples 10000 \
    --rand_order \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1



python llm_infer_output.py \
    --prefix LLM_TEST32_SPANYN_nq_open_dev_using_muv2_1krandord \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/nq_open_llm_expl/dev.tsv \
    --predict_file nq_open_od_ans/dev.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples 1000 \
    --rand_order \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST33_SPANYN_nq_open_train_using_muv2_10krandord \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/nq_open_llm_expl/train.tsv \
    --predict_file nq_open_od_ans/train.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --debug \
    --max_samples 10000 \
    --rand_order \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1



