
cd $UDATA/gitrepos/transformers

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /data/LLaMA/13B --model_size 13B --output_dir $UDATA/ckpts/llama13b

