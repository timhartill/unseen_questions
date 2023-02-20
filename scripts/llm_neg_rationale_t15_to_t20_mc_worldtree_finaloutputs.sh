# Abduce rationale from LLM
# run on 3 80GB gpus, will take ~75 GB on 2 and ~40GB on the third

#     --generate_train \
#     --query_no_mc \
#     --rand_order \


cd ../code

echo "Outputting train/dev based on v3 prompt..."

python llm_infer_output.py \
    --prefix LLM_NEGRAT_T15_MCFOCUS_WORLDTREE_DEV_all_onv3 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v3_mod3_onlyneg.txt \
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
    --prefix LLM_NEGRAT_T16_MCFOCUS_WORLDTREE_TRAIN_all_onv3 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/train.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v3_mod3_onlyneg.txt \
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
    --prefix LLM_NEGRAT_T17_MCFOCUS_WORLDTREE_DEV_all_onv6 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v6_mod3_theanswermusteverythingfalse.txt \
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
    --prefix LLM_NEGRAT_T18_MCFOCUS_WORLDTREE_TRAIN_all_onv6 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/train.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v6_mod3_theanswermusteverythingfalse.txt \
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
    --prefix LLM_NEGRAT_T19_MCFOCUS_WORLDTREE_DEV_all_onv8 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/dev.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v8_mod3_negatepositive.txt \
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
    --prefix LLM_NEGRAT_T20_MCFOCUS_WORLDTREE_TRAIN_all_onv8 \
    --output_dir $LDATA/out/mdr/logs \
    --predict_file worldtree_mc_expl_ans/train.tsv \
    --template_file neg_rationale_worldtree_mconly_anschoices_v8_mod3_negatepositive.txt \
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



