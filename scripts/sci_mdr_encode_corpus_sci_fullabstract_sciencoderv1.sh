# Encode and save corpus paragraphs
# After completing base and/or momentum training
# On 3 GPUS bs 1000 only takes ~8GB GPU RAM on each GPU. Time to encode 5M HPQA abstracts = 1hr 15mins
# On 1 gpu bs 500 takes ~12GB. Time approx 5hrs on gpu running 2 other jobs.
# On 1 gpu bs 250 takes ~8.5GB. Time approx 3.5 - 6hrs on gpu running 1 other job. (~3 hrs with dedicated gpu)

# Set predict_file to the name of the corpus file containing title and paragraph text for each entry.
# HPQA: /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_abstracts_with_sent_spans.jsonl \ 
# BQA: /home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl \
# Set init_checkpoint to the base trained or the base+momentum trained ckpt.
# Set embed_save_path to the directory to save embeddings (index.npy) and the corresponding text (id2doc.json) to.


# set --update_id2doc_only to only update the id2doc file eg to add sentence span info without encoding para embeddings.

# encode hpqa abstracts using hpqa-trained model which used query title:sents enc: 
#    --predict_file /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_abstracts_with_sent_spans.jsonl \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/hpqa_sent_annots_test1-04-18-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
#    --embed_save_path /large_data/thar011/out/mdr/encoded_corpora/hpqa_sent_annots_test1_04-18_bs24_no_momentum_cenone_ckpt_best \


# encode hpqa abstracts using hpqa+hover trained model which used query para enc:
#    --predict_file /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_abstracts_with_sent_spans.jsonl \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/hover_hpqa_paras_test3-07-12-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
#    --embed_save_path /large_data/thar011/out/mdr/encoded_corpora/hover_hpqa_paras_test3-07-12-2022-nomom \


# encode full wikipedia using bqa_nosquad+nq+tqa traine dmodel which used query para enc:
#    --predict_file /home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl \
#    --init_checkpoint /large_data/thar011/out/mdr/logs/bqa_nosquad_nq_tqa_test3-04-14-2022-nomom-seed16-bsz24-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
#    --embed_save_path /large_data/thar011/out/mdr/encoded_corpora/bqa_nosquad_nq_tqa_test3-04-14_bs24_no_momentum_cenone_ckpt_best \

#     --predict_file $LDATA/out/mdr/encoded_corpora/hotpot/hpqa_abstracts_with_sent_spans.jsonl \
#    --predict_file $HDATA/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl \

#   --init_checkpoint $LDATA/out/mdr/logs/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250_hgx2-09-02-2022-mom-seed16-bsz250-fp16True-lr1e-05-decay0.0-warm0.1-valbsz100-m0.999-k76800-t1.0-ga1-varTrue-cenone/checkpoint_q_best.pt \
# --init_checkpoint $LDATA/out/mdr/logs/hpqa_mdr_orig_ckpt_8gpu_bs150/q_encoder.pt


cd ../code

python mdr_encode_corpus_nativeamp.py \
    --do_predict \
    --predict_batch_size 500 \
    --model_name roberta-base \
    --predict_file $HDATA/data/SCI/sci_corpus_with_sent_spans.jsonl \
    --init_checkpoint $LDATA/out/mdr/logs/scifact_orig_test1_6gpus_bs150_from_hover_hpqa_nq_mu_paras_test12_mom-01-06-2023-nomom-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
    --embed_save_path $LDATA/out/mdr/encoded_corpora/sci_full_abstract_c512_sciencoderv1_test1_mom_6gpubs250 \
    --use_var_versions \
    --fp16 \
    --max_c_len 510 \
    --num_workers 10




