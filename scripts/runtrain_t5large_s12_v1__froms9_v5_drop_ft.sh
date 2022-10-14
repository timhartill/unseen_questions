# training command

#        --train_batch_size 32 \
#        --fp16 \
#         --dont_pretokenize \
#        --error_based_sampling \
#        --indiv_digits \
#         --num_workers 10 \
#         --num_workers 0 \    default is 0 so no need to include
#         --add_mask_char NONE \  made this the default so excluding here

cd ../code

python cli.py --do_train --output_dir $LDATA/out/mdr/logs/UQA_s12_v1__froms9_v5_drop_ft_t5l \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --checkpoint $LDATA/out/mdr/logs/UQA_s9_v5_numlit_wikissvise_idt_errsamp_fixdecode_t5large/best-model.pt \
        --train_batch_size 30 \
        --predict_batch_size 30 \
        --do_lowercase \
        --eval_period 10000 --verbose \
        --gradient_accumulation_steps 4 \
        --wait_step 10 \
        --num_scheduler_steps 150000 \
        --learning_rate 5e-5 \
        --model t5-large \
        --strip_single_quotes \
        --seed 42 \
        --ssm_prob 0.65 \
        --add_mask_char NONE \
        --max_output_length 130 \
        --fp16 \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --indiv_digits \
        --mixture drop
        

