# training command t5 large on stage 1 abstractable knowledge tasks
# 1e-4 lr from Yoran/PreasM for t5 large (uqa used 1e-3)
# under fp16 train bs 32 just ran in 80GB but reducing to 30 just to be safe


#        --train_batch_size 32 \
#        --fp16 \
#         --dont_pretokenize \
#        --error_based_sampling \
#        --indiv_digits \
#         --num_workers 10 \
#         --num_workers 0 \    default is 0 so no need to include
#         --add_mask_char NONE \  made this the default so excluding here

cd ../code

python cli.py --do_train --output_dir $LDATA/out/mdr/logs/UQA_s9_v5_numlit_wikissvise_idt_errsamp_fixdecode_t5large \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --train_batch_size 30 \
        --predict_batch_size 30 \
        --do_lowercase \
        --eval_period 10000 --verbose \
        --gradient_accumulation_steps 4 \
        --wait_step 10 \
        --num_scheduler_steps 250000 \
        --learning_rate 1e-4 \
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
        --error_based_sampling \
        --error_based_ssvise_prob 0.35 \
        --approx_dev_samples 1250 \
        --mixture tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised
        

