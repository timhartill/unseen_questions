# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \

cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST28_SPANYN_hpqa_dev_using_muv2_1krandord \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/hpqa_llm_expl/dev.tsv \
    --predict_file hpqa_od_ans/dev.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 1000 \
    --rand_order \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_TEST29_SPANYN_hpqa_train_using_muv2_10krandord \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/hpqa_llm_expl/train.tsv \
    --predict_file hpqa_od_ans/train.tsv \
    --template_file generic_spanmadeup_hpqa_csqa2_weicot_withinstruction_muv2.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 10000 \
    --rand_order \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


