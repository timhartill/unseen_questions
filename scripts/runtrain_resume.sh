# resume training command
# other options:  --do_predict --skip_inference --debug --checkpoint ${unifiedqa_checkpoint}
# --prefix dev_  --prefix test_
#         --checkpoint /data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt \
# --mixture unifiedqa,newdataset1,newdataset2  will add newdataset1 and newdataset2 to the training and validation mixtures along with unifiedqa ds's 
# IMPORTANT! Take a copy of best-model if there is one since it will be overwritten by the first validation step!
# IMPORTANT 2! Set --checkpoint_steps, --model and --mixture etc to match the original training... 
# $1 should be /data/thar011/out/name_of_output_dir 


cd ../code


python cli.py --do_train --output_dir $1 \
        --is_unifiedqa \
        --train_file $UDATA/data/unifiedqa/train.tsv \
        --predict_file $UDATA/data/unifiedqa/dev.tsv \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --num_train_epochs 10000 \
        --gradient_accumulation_steps 2 \
        --wait_step 10 \
        --num_scheduler_steps 250000 \
        --learning_rate 2e-5 \
        --checkpoint ${1}/best-model.pt \
        --checkpoint_step 80100 \
        --model facebook/bart-large \
        --mixture unifiedqa

