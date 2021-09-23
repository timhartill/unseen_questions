#!/bin/bash

cd ../code

python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_TEST \
   --predict_file /data/thar011/data/unifiedqa/drop/dev.tsv \
   --predict_batch_size 32 \
   --append_another_bos --do_lowercase \
   --verbose \
   --prefix dev_drop_ \
   --model facebook/bart-large \
   --indiv_digits \
   --calc_metrics       

