# training command

cd ../code

python cli.py --do_train --output_dir /data/thar011/out/unifiedqa_bart_large_s7_v3_uqa_ms_explans_msw \
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
        --mixture unifiedqa,musique_full_qa_od_ans,strategy_qa_od_ans,musique_full_qa_expl_ans,strategy_qa_expl_ans,worldtree_mc_ans
        

