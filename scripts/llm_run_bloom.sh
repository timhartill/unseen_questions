# Abduce rationale from LLM


cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_inital_contextv2_fullwiki_bs150/dev.tsv \
    --predict_file $UQA_DIR/musique_mu_dev_inital_contextv2/dev.tsv \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --generate_train
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 3 \
    --num_beams 4 \
    --num_return_sequences 1 \
    --max_memory -1

python llm_infer_output.py \
    --prefix LLM_TEST \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_inital_contextv2_fullwiki_bs150/dev.tsv \
    --predict_file $UQA_DIR/musique_mu_dev_inital_contextv2/dev.tsv \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --generate_train
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 3 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_memory -1

python llm_infer_output.py \
    --prefix LLM_TEST \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_inital_contextv2_fullwiki_bs150/dev.tsv \
    --predict_file $UQA_DIR/musique_mu_dev_inital_contextv2/dev.tsv \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --generate_train
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 3 \
    --do_sample \
    --temperature 0.7 \
    --top_k 50 \
    --num_return_sequences 2 \
    --max_memory -1

python llm_infer_output.py \
    --prefix LLM_TEST \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_inital_contextv2_fullwiki_bs150/dev.tsv \
    --predict_file $UQA_DIR/musique_mu_dev_inital_contextv2/dev.tsv \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --generate_train
    --generate_dev \
    --generate_eval \
    --debug \
    --max_samples 3 \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.92 \
    --top_k 0 \
    --num_return_sequences 2 \
    --max_memory -1



