# training command - train stage 2 without idt from stage 1 best model over 1m steps also trained without idt

#        --train_batch_size 32 \
#        --fp16 \
#         --dont_pretokenize \
#        --error_based_sampling \
#        --indiv_digits \
#         --num_workers 10 \
#         --num_workers 0 \    default is 0 so no need to include
#         --add_mask_char NONE \  made this the default so excluding here

cd ../code

python cli.py --do_train --output_dir $LDATA/out/mdr/logs/UQA_s11_v12_all_g1_qa_g2_numlit_wikissvise_from_s9_v6_NOidt_addretds \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --checkpoint $LDATA/out/mdr/logs/UQA_s9_v6_numlit_wikissvise_NOidt_errsamp_fixdecode/best-model.pt \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --gradient_accumulation_steps 4 \
        --wait_step 100 \
        --num_scheduler_steps 250000 \
        --learning_rate 2e-5 \
        --model facebook/bart-large \
        --seed 42 \
        --ssm_prob 0.65 \
        --add_mask_char NONE \
        --max_output_length 130 \
        --fp16 \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --approx_dev_samples 1250 \
        --g2_prob 0.2 \
        --error_based_ssvise_prob 0.05 \
        --g1_type err \
        --g2_type uni \
        --g2_datasets q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised \
        --mixture q_paras_all,q_paras_noanswer_all,q_mc_all,q_mc_paras_all,q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised,q_ret_paras_all,q_ret_paras_maxp4_all
        

