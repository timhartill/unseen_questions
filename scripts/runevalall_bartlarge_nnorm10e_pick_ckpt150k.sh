#!/bin/bash

cd ../code

echo "Running Eval for best model in $1 ..."

# eval on dev set
for ds in newsqa quoref contrast_sets_quoref ropes contrast_sets_ropes drop contrast_sets_drop boolq_np contrast_sets_boolq multirc natural_questions natural_questions_with_dpr_para physical_iqa social_iqa ambigqa squad1_1 squad2 boolq commonsenseqa qasc qasc_with_ir winogrande_xl mctest_corrected_the_separator
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done

#eval on test set where available
for ds in openbookqa openbookqa_with_ir arc_easy arc_easy_with_ir arc_hard arc_hard_with_ir ai2_science_elementary ai2_science_middle race_string narrativeqa
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done

#eval on mmlu test sets
for ds in mmlu_electrical_engineering_test mmlu_high_school_statistics_test mmlu_college_chemistry_test mmlu_econometrics_test mmlu_high_school_world_history_test mmlu_management_test mmlu_business_ethics_test mmlu_jurisprudence_test mmlu_world_religions_test mmlu_global_facts_test mmlu_college_medicine_test mmlu_computer_security_test mmlu_us_foreign_policy_test mmlu_international_law_test mmlu_nutrition_test mmlu_philosophy_test mmlu_virology_test mmlu_high_school_government_and_politics_test mmlu_clinical_knowledge_test mmlu_college_computer_science_test mmlu_college_physics_test mmlu_anatomy_test mmlu_college_biology_test mmlu_high_school_us_history_test mmlu_moral_scenarios_test mmlu_public_relations_test mmlu_high_school_psychology_test mmlu_professional_psychology_test mmlu_high_school_chemistry_test mmlu_high_school_geography_test mmlu_medical_genetics_test mmlu_college_mathematics_test mmlu_high_school_microeconomics_test mmlu_machine_learning_test mmlu_professional_law_test mmlu_miscellaneous_test mmlu_moral_disputes_test mmlu_high_school_computer_science_test mmlu_high_school_macroeconomics_test mmlu_elementary_mathematics_test mmlu_professional_accounting_test mmlu_astronomy_test mmlu_high_school_physics_test mmlu_logical_fallacies_test mmlu_human_aging_test mmlu_high_school_european_history_test mmlu_sociology_test mmlu_security_studies_test mmlu_high_school_mathematics_test mmlu_high_school_biology_test mmlu_prehistory_test mmlu_conceptual_physics_test mmlu_professional_medicine_test mmlu_marketing_test mmlu_abstract_algebra_test mmlu_human_sexuality_test mmlu_formal_logic_test
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on dedup dev sets
for ds in contrast_sets_drop_dedup drop_dedup contrast_sets_boolq_dedup boolq_np_dedup ambigqa_dedup social_iqa_dedup quoref_dedup contrast_sets_quoref_dedup contrast_sets_ropes_dedup
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on dedup test sets
for ds in mmlu_us_foreign_policy_test_dedup mmlu_college_physics_test_dedup mmlu_public_relations_test_dedup mmlu_high_school_psychology_test_dedup mmlu_professional_psychology_test_dedup mmlu_elementary_mathematics_test_dedup mmlu_elementary_to_college_math_test
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on lowsim dev sets
for ds in drop_dedup_lowsim_uqa contrast_sets_drop_dedup_lowsim_uqa physical_iqa_lowsim_uqa social_iqa_dedup_lowsim_uqa commonsenseqa_lowsim_uqa qasc_lowsim_uqa qasc_with_ir_lowsim_uqa ropes_lowsim_uqa newsqa_lowsim_uqa drop_dedup_lowsim_tdnd contrast_sets_drop_dedup_lowsim_tdnd physical_iqa_lowsim_tdnd social_iqa_dedup_lowsim_tdnd commonsenseqa_lowsim_tdnd qasc_lowsim_tdnd qasc_with_ir_lowsim_tdnd ropes_lowsim_tdnd newsqa_lowsim_tdnd
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on lowsim test sets
for ds in mmlu_elementary_to_college_math_test_lowsim_uqa mmlu_elementary_to_college_math_test_lowsim_tdnd
do
    echo "Running eval for $ds on test..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/test.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix test_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


#eval on strategy_qa etc dev sets
for ds in strategy_qa cwwv atomic
do
    echo "Running eval for $ds on dev..."
    python cli.py --do_predict --output_dir $1 \
        --predict_file /data/thar011/data/unifiedqa/${ds}/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --prefix dev_${ds}_ \
        --model facebook/bart-large \
        --checkpoint $1/best-model-150000.pt \
        --norm_numbers --norm_10e \
        --calc_metrics
done


echo Finished!

