# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#    --rand_order \


cd ../code

echo "Outputting CREAK..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T24_YN_CREAK_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file creak_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T25_YN_CREAK_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file creak_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T26_YN_CREAK_TRAIN_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file creak_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T27_YN_CREAK_TRAIN_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file creak_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "Outputting QASC..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T28_YN_QASC_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T29_YN_QASC_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T30_YN_QASC_TRAIN_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T31_YN_QASC_TRAIN_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file qasc_mc_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "Outputting WORLDTREE..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T32_YN_WORLDTREE_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T33_YN_WORLDTREE_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T34_YN_WORLDTREE_TRAIN_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T35_YN_WORLDTREE_TRAIN_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples -1 \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


echo "Outputting HOVER..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T36_YN_HOVER_DEV_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hover_od_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples 1000 \
    --rand_order \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T37_YN_HOVER_DEV_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hover_od_ans/dev.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples 1000 \
    --rand_order \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T38_YN_HOVER_TRAIN_onv6_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hover_od_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_onlyneg_theanswermustbe.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples 10000 \
    --rand_order \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1


python llm_infer_output.py \
    --prefix LLM_NEGRAT_T39_YN_HOVER_TRAIN_onv6mod2_sample \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file hover_od_ans/train.tsv \
    --template_file neg_rationale_creak_yn_v6_2_onlyneg_theanswermustbe_singlesentence.txt \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_seq_len_in 1472 \
    --query_no_context \
    --query_no_nl_mc_options \
    --debug \
    --debug_count 10 \
    --max_samples 10000 \
    --rand_order \
    --do_sample \
    --temperature 0.95 \
    --top_p 0.96 \
    --top_k 0 \
    --num_return_sequences 5 \
    --max_memory_buffer 12 \
    --max_memory -1







