# Continue training of q_encoder with momentum
# Original non-varstep version:bs 100 On 4 GPUS GPU loading is 18GB each and training/eval 1 epoch takes ~12mins
#varstep version bs 50 1 GPU takes ~36GB. 1 epoch approx 30mins

# Note: original momentum trainer script did NOT apply warmup

cd ../code

python mdr_train_mhop_nativeamp.py \
    --do_train \
    --prefix hover_hpqa_nq_mu_paras_test11_mom_6gpubs150_hgx2 \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file $LDATA/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_train_with_neg_v0.jsonl \
    --predict_file $LDATA/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_dev_with_neg_v0.jsonl \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --shared-encoder \
    --gradient_accumulation_steps 1 \
    --use_var_versions \
    --output_dir $LDATA/out/mdr/logs \
    --momentum \
    --reduction none \
    --retrieve_loss_multiplier 1.0 \
    --max_hops 4 \
    --num_negs 2 \
    --random_multi_seq \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init_retriever $LDATA/out/mdr/logs/hover_hpqa_nq_mu_paras_test10_6gpubs150-08-30-2022-nomom-seed16-bsz150-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-ga1-varTrue-cenone/checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --num_train_epochs 75 \
    --warmup-ratio 0.1
    
