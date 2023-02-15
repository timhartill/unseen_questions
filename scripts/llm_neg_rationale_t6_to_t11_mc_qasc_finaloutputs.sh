# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#     --query_no_mc \
#     --rand_order \


cd ../code

echo "Outputting train/dev based on v3 prompt..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T6_MCFOCUS_QASC_DEV_all_onv3 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v3_onlyneg.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T7_MCFOCUS_QASC_TRAIN_all_onv3 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/train.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v3_onlyneg.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "Outputting train/dev based on v6 prompt..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T8_MCFOCUS_QASC_DEV_all_onv6 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v6_onlyneg_theanswermusteverythingfalse.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T9_MCFOCUS_QASC_TRAIN_all_onv6 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/train.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v6_onlyneg_theanswermusteverythingfalse.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "Outputting train/dev based on v8 prompt..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T10_MCFOCUS_QASC_DEV_all_onv8 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v8_negatepositive.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T11_MCFOCUS_QASC_TRAIN_all_onv8 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/train.tsv \
    --template_file neg_rationale_qasc_multi_fact_sameliuquestions_mconly_anschoices_v8_negatepositive.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory_buffer 12 \
    --max_memory -1



