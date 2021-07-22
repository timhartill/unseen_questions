#!/bin/bash
# add information for incremental eval datasets, in this case no answer overlap versions, for analysis
# (0) add each new dataset to the dataset_attribs dict in eval_metrics.py 
# run (1) predictions + (2) calc metrics -update the runevalall files for each model below then create version below, copy/paste/run it, 
# once this done (3) create sentence embeddings for each new dataset, 
# then (4) update similarity json file for new datasets to existing model outputs
# /data/thar011/out/unifiedqa_bart_large_v3                             script: runevalall_bartlarge.sh
# /data/thar011/out/unifiedqa_bart_large_v4indiv_digits        script: runevalall_bartlarge_indivdigits.sh & runevalall_bartlarge_indivdigits_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td     script: runevalall_bartlarge_indivdigits.sh & runevalall_bartlarge_indivdigits_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd     script: runevalall_bartlarge_indivdigits.sh & runevalall_bartlarge_indivdigits_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd   script: runevalall_bartlarge_indivdigits.sh & runevalall_bartlarge_indivdigits_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e           script: runevalall_bartlarge_nnorm10e_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd      script: runevalall_bartlarge_nnorm10e_pick_ckpt150k.sh
# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd_200k  script: runevalall_bartlarge_indivdigits_pick_ckpt200k.sh
# /data/thar011/out/unifiedqa_2gputest_from_uqackpt              script: runevalall_bartlarge_indivdigits.sh
# /data/thar011/out/unifiedqa_allenai_bartlarge_eval                    script: runevalall_allenai_bartlarge.sh
# /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos               script: runevalall_allenai_t5large.sh
# /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos                script: runevalall_allenai_t5base.sh
# /data/thar011/out/unifiedqa_t5_base                                   script: runevalall_t5base.sh
# /data/thar011/out/unifiedqa_t5base_290ksteps                          script: runevalall_t5base_290ksteps.sh

cd ../code


# /data/thar011/out/unifiedqa_bart_large_v3 update
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v3 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v3 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --calc_metrics
done

# /data/thar011/out/unifiedqa_bart_large_v4indiv_digits
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v4indiv_digits \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v4indiv_digits/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v4indiv_digits \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v4indiv_digits/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v5indiv_digits_td/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v6indiv_digits_nd/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done


# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-150000.pt \
        --indiv_digits \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v12_nnorm10e/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v15_nnorm10e_tdnd/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd_200k
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd_200k \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-200000.pt \
        --indiv_digits \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd_200k \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint /data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-200000.pt \
        --indiv_digits \
        --calc_metrics
done

# /data/thar011/out/unifiedqa_2gputest_from_uqackpt
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_2gputest_from_uqackpt \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_2gputest_from_uqackpt \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_allenai_bartlarge_eval
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_bartlarge_eval \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --checkpoint /data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt \
        --model facebook/bart-large \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_bartlarge_eval \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --checkpoint /data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt \
        --model facebook/bart-large \
        --calc_metrics
done



# /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model allenai/unifiedqa-t5-large \
        --calc_metrics \
        --strip_single_quotes
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_t5large_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model allenai/unifiedqa-t5-large \
        --calc_metrics \
        --strip_single_quotes
done



# /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model allenai/unifiedqa-t5-base \
        --calc_metrics \
        --strip_single_quotes
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_allenai_t5base_eval_no_bos \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model allenai/unifiedqa-t5-base \
        --calc_metrics \
        --strip_single_quotes
done



# /data/thar011/out/unifiedqa_t5_base
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_t5_base \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model t5-base \
        --calc_metrics \
        --strip_single_quotes
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_t5_base \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model t5-base \
        --calc_metrics \
        --strip_single_quotes
done



# /data/thar011/out/unifiedqa_t5base_290ksteps
#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_t5base_290ksteps \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model t5-base \
        --checkpoint /data/thar011/out/unifiedqa_t5_base/best-model-290000.pt \
        --calc_metrics \
        --strip_single_quotes
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir /data/thar011/out/unifiedqa_t5base_290ksteps \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model t5-base \
        --checkpoint /data/thar011/out/unifiedqa_t5_base/best-model-290000.pt \
        --calc_metrics \
        --strip_single_quotes
done




echo Finished!

