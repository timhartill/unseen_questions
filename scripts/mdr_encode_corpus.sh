# Encode and save corpus paragraphs
# After completing base and/or momentum training
# On 3 GPUS the below params only take ~8GB GPU RAM on each GPU. Time to encode 5M HPQA abstracts = 1hr 15mins
#CUDA_VISIBLE_DEVICES=0,1,2,3 

# Set predict_file to the name of the corpus file containing title and paragraph text for each entry.
# Set init_checkpoint to the base trained or the base+momentum trained ckpt.
# Set embed_save_path to the location to save embeddings (index.npy) and the corresponding text (id2doc.json) to.

cd ../code

python mdr_encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name roberta-base \
    --predict_file /home/thar011/data/mdr/hotpot/hpqa_abstracts_tim.jsonl \
    --init_checkpoint /large_data/thar011/out/mdr/logs/03-23-2022/varinitialtest_-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --embed_save_path /home/thar011/data/mdr/hotpot/hpqa_tim_no_momentum \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20


#python scripts/encode_corpus.py \
#    --do_predict \
#    --predict_batch_size 1000 \
#    --model_name roberta-base \
#    --predict_file /home/thar011/data/mdr/hotpot/hpqa_abstracts_tim.jsonl \
#    --init_checkpoint logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
#    --embed_save_path /home/thar011/data/mdr/hotpot/hpqa_tim_momentum \
#    --fp16 \
#    --max_c_len 300 \
#    --num_workers 20


