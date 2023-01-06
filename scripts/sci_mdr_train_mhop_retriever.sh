#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# Orig non-var step: mdr hpqa train/dev files with train/pred bs 100 on 4 GPUS takes 44GB on 1st gpu and 43.1GB on remaining 3. Takes 13.5hrs for 50 epochs. best at 48:
#export CUDA_VISIBLE_DEVICES=1,2,4,5

# bs100 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
# bs75 on 5 gpus: 1st one total free, others ~26GB free: OOM but maybe ~30GB free on extras would have done it...
#export CUDA_VISIBLE_DEVICES=2
# bs50 on 1 gpu with 42.5GB free fails (also fails in this config for orig mdr version without _var routines)
# bs24 on one gpu with 42.5GB free (6639mb taken) just fits! (initially gets up to 42GB taken then falls to ~38GB taken as fp16 scaling kicks in) (pred bs 100 here works). Stopped after 26 epochs. Best=epoch 21: MRRS: {'mrr_1': 0.9389304208307683, 'mrr_2': 0.9643570317425275, 'mrr_avg': 0.951643726286648} NOTE loss NAN!: Step 82918 Train loss nan MRR-AVG 95.16 on epoch=21. Last=MRRS: {'mrr_1': 0.9364797357582563, 'mrr_2': 0.9660725050908262, 'mrr_avg': 0.9512761204245412} Step 101763 Train loss nan MRR-AVG 95.13 on epoch=26
# with torch.cuda.amp: bs25 starts on 1 gpu using ~43GB. No gradient overflow msgs or scaler scale msgs and seems to run slightly faster approx 41 mins per epoch..

#hpqa train_file /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0.json \
#hpqa predict_file /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0.json \
#hpqa sentence annots train /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl
#hpqa sentence annots predict /large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl
#beerqa train /home/thar011/data/beerqa/beerqa_train_v1.0_with_neg_v0.jsonl \
#beerqa predict /home/thar011/data/beerqa/beerqa_dev_v1.0_with_neg_v0.jsonl \
#bqa_nq_tqa train /home/thar011/data/DPR/bqa_nq_tqa_train_v1.0_with_neg_v0.jsonl
#bqa_nq_tqa predict /home/thar011/data/DPR/bqa_nq_tqa_dev_v1.0_with_neg_v0.jsonl
#bqa no squad nq tqa train /home/thar011/data/DPR/bqa_nosquad_nq_tqa_train_v1.0_with_neg_v0.jsonl
#bqa no squad nq tqa predict /home/thar011/data/DPR/bqa_nosquad_nq_tqa_dev_v1.0_with_neg_v0.jsonl
#hover train /home/thar011/data/baleen_downloads/hover/hover_train_with_neg_and_sent_annots.jsonl
#hover dev   /home/thar011/data/baleen_downloads/hover/hover_dev_with_neg_and_sent_annots.jsonl
# hpqa+hover train /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_train_with_neg_v0_sentannots.jsonl
# hpqa+hover dev /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_dev_with_neg_v0_sentannots.jsonl
# hpqa+hover_nq train  /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_train_with_neg_v0.jsonl
# hpqa+hover_nq dev  /large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_dev_with_neg_v0.jsonl


#bqa ~134k training samples vs ~90k hpqa. On 1 gpu bs24 est 1hr 15mins per epoch vs ~45mins per epoch.  

#     --query_add_titles \
# --random_multi_seq \  # (randomises para ordering in each step)
#    --query_use_sentences \
#    --query_add_titles \

#    --train_file $LDATA/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_train_with_neg_v0.jsonl \
#    --predict_file $LDATA/out/mdr/encoded_corpora/hotpot/hpqa_hover_nq_mu_dev_with_neg_v0.jsonl \
# /home/thar011/data/scifact/data/scifact_orig_dev_with_neg_and_sent_annots.jsonl

cd ../code

python mdr_train_mhop_nativeamp.py \
    --do_train \
    --prefix scifact_orig_test1_6gpus_bs150_from_hover_hpqa_nq_mu_paras_test12_mom \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file $HDATA/data/scifact/data/scifact_orig_train_with_neg_and_sent_annots.jsonl \
    --predict_file $HDATA/data/scifact/data/scifact_orig_dev_with_neg_and_sent_annots.jsonl \
    --init_checkpoint $LDATA/out/mdr/logs/hover_hpqa_nq_mu_paras_test12_mom_6gpubs250_hgx2-09-02-2022-mom-seed16-bsz250-fp16True-lr1e-05-decay0.0-warm0.1-valbsz100-m0.999-k76800-t1.0-ga1-varTrue-cenone/checkpoint_q_best.pt \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 435 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --shared-encoder \
    --gradient_accumulation_steps 1 \
    --use_var_versions \
    --reduction none \
    --retrieve_loss_multiplier 1.0 \
    --max_hops 1 \
    --num_negs 6 \
    --random_multi_seq \
    --output_dir $LDATA/out/mdr/logs \
    --num_train_epochs 50 \
    --warmup-ratio 0.1





    
