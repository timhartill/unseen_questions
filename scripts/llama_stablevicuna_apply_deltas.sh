
cd ../code

python llama_stablevicuna_apply_delta.py \
    --base-model-path $UDATA/ckpts/llama13b --target-model-path $UDATA/ckpts/stable-vicuna-13b --delta-path CarperAI/stable-vicuna-13b-delta

