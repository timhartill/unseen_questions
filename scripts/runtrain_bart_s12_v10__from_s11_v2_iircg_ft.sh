# training command ft iirc-G from Base model




cd ../code

python cli.py --do_train --output_dir $LDATA/out/mdr/logs/UQA_s12_v10__from_s11_v2_iircg_ft_bart \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --checkpoint $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/best-model.pt \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --gradient_accumulation_steps 4 \
        --wait_step 10 \
        --num_scheduler_steps 150000 \
        --learning_rate 2e-5 \
        --model facebook/bart-large \
        --seed 42 \
        --ssm_prob 0.65 \
        --add_mask_char NONE \
        --max_output_length 130 \
        --fp16 \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --indiv_digits \
        --mixture iirc_gold_context
        

