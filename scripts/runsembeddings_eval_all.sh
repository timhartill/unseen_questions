# Output Main results for our memorisation paper to csv files
# Set up for single model comparison. If you have made extra runs of UQA and/or UQA+TDND,  Expand --uqa_run_dirs and --uqatdnd_run_dirs as in eg: uqa,uqa_run2,uqa_run3


cd ../code


python calc_eval_lowsim_highsim.py \
        --in_log_dir $LDATA \
        --uqa_run_subdirs uqa \
        --uqatdnd_run_subdirs uqa_plus_tdnd \
        --output_dir $LDATA/eval_outputs/uqa_1run \
        --sim_file $LDATA/uqa_plus_tdnd/eval_test_train_similarities_semb_thresh-100.1.json



