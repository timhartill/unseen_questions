# training command

cd ../code

python cli.py --do_train --output_dir /data/thar011/out/unifiedqa_bart_large_s6_v6_musique_qa_paras_plus_all_decomps \
        --is_unifiedqa \
        --train_file /data/thar011/data/unifiedqa/train.tsv \
        --predict_file /data/thar011/data/unifiedqa/dev.tsv \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --num_train_epochs 10000 \
        --gradient_accumulation_steps 2 \
        --wait_step 10 \
        --num_scheduler_steps 250000 \
        --learning_rate 2e-5 \
        --model facebook/bart-large \
        --seed 42 \
        --max_output_length 130 \
        --mixture musique_qa_paras,musique_decomp_all_dev_in_train

