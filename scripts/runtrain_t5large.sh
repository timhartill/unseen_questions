# training command

cd ../code

python cli.py --do_train --output_dir $LDATA/out/unifiedqa_t5_large \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --train_batch_size 16 \
        --predict_batch_size 16 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --num_train_epochs 10000 \
        --gradient_accumulation_steps 4 \
        --wait_step 10 \
        --num_scheduler_steps 250000 \
        --learning_rate 1e-3 \
        --model t5-large \
        --mixture unifiedqa \
        --strip_single_quotes

