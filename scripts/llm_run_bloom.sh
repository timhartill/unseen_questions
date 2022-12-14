# Abduce rationale from LLM


cd ../code

python llm_infer_output.py \
    --prefix LLM_TEST_ \
    --output_dir $LDATA/out/mdr/logs \
    --output_dataset $UQA_DIR/musique_mu_dev_inital_contextv2_fullwiki_bs150/dev.tsv \
    --predict_file $UQA_DIR/musique_mu_dev_inital_contextv2/dev.tsv \
    --model_name bigscience/bloom \
    --max_new_tokens 128 \
    --max_memory -1



