#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:19:09 2021

@author: Tim Hartill

Edit this file to add new datasets to evaluation/similarity routines:
    
- Add to dev_eval to produce preds, metrics for a dataset's dev.tsv - each dataset must be in either dev_eval or test_eval but not both!
- Add to test_eval to produce preds, metrics for a dataset's test.tsv - each dataset must be in either dev_eval or test_eval but not both!
- Add to dataset_attribs to include in eval and/or similarity calc plus to configure the set of metrics to be produced for a given dataset
- Add to replace_sim_with to use sembs from another dataset as proxy in calculating similarity for a given dataset
- Edit the following to add/remove datasets from sets of current output reports:
    unifiedqa_unseen_4
    unifiedqa_unseen_4_map  # must configure this to identify whether dev.tsv or test.tsv is the file to be used for calculation
    unifiedqa_unseen_6
    unifiedqa_seen_1
- Edit UQA_DIR to point to base directory for unified-qa formatted datasets.
- Edit create_datasets_dynamic to add new datasets to dynamically create explations for (i.e from q[+mc]->a make q[+mc]+e->a). 
    Datasets added here must be in dev_eval/test_eval and in dataset_attribs..
    Dynamically created versions i.e /UQA_DIR/qasc_svised_expl_ans_modeloutputdir_timestamp will be added to dev_eval/test_eval and dataset_attribs when this module is loaded..


"""
import os
import fnmatch

SVISED_EXPL_ANS = '_dyn_expl_ans_'
selfsupervisedkey = '_selfsvised'   # dataset names ending in this will be processed as self supervised
add_explanationkey = 'Add Explanation:'
EXPL_COMP_KEY = '_expl_components'


UQA_DIR = '/data/thar011/data/unifiedqa/' # datasets base directory


#Add to this list to create predictions/calc metrics for corresponding dev.tsv:
dev_eval = ['newsqa', 'quoref', 'contrast_sets_quoref', 'ropes', 'contrast_sets_ropes', 
            'drop', 'contrast_sets_drop', 'boolq_np', 'contrast_sets_boolq', 'multirc', 
            'natural_questions', 'natural_questions_with_dpr_para', 'physical_iqa', 
            'social_iqa', 'squad1_1', 'squad2', 'boolq', 'commonsenseqa', 
            'qasc', 'qasc_with_ir', 'winogrande_xl', 'mctest_corrected_the_separator', 
            'contrast_sets_drop_dedup', 'drop_dedup', 'contrast_sets_boolq_dedup', 
            'boolq_np_dedup', 'social_iqa_dedup', 'quoref_dedup', 
            'contrast_sets_quoref_dedup', 'contrast_sets_ropes_dedup', 
            'drop_dedup_lowsim_tdnd', 
            'contrast_sets_drop_dedup_lowsim_tdnd', 'physical_iqa_lowsim_tdnd', 
            'social_iqa_dedup_lowsim_tdnd', 'commonsenseqa_lowsim_tdnd', 
            'qasc_lowsim_tdnd', 'qasc_with_ir_lowsim_tdnd', 'ropes_lowsim_tdnd', 
            'newsqa_lowsim_tdnd', 'strategy_qa', 'cwwv', 'atomic',
            'musique_qa', 'musique_qa_paras', 'musique_mu_dev_qa', 'musique_mu_dev_qa_paras',
            'musique_qa_decomp_ans', 'musique_qa_paras_decomp_ans', 'musique_mu_dev_qa_decomp_ans',
            'musique_mu_dev_qa_paras_decomp_ans', 'musique_mu_dev_qa_decomp_context', 
            'musique_mu_dev_qa_expl_ans', 'qasc_mc_ans', 'strategy_qa_expl_ans', 'strategy_qa_od_ans']

#Add to this list to create predictions/calc metrics for corresponding test.tsv:
test_eval = ['openbookqa', 'openbookqa_with_ir', 'arc_easy', 'arc_easy_with_ir', 'arc_hard', 
             'arc_hard_with_ir', 'ai2_science_elementary', 'ai2_science_middle', 'race_string',  
             'mmlu_elementary_to_college_math_test',  
             'mmlu_elementary_to_college_math_test_lowsim_tdnd', 'worldtree_mc_ans', 
             'nq_open_od_ans', 'arc_da_expl_ans', 'arc_da_od_ans']


#Unused, just to keep complete list of everything ever used in eval handy..
dev_eval_all = ['newsqa', 'quoref', 'contrast_sets_quoref', 'ropes', 'contrast_sets_ropes', 
            'drop', 'contrast_sets_drop', 'boolq_np', 'contrast_sets_boolq', 'multirc', 
            'natural_questions', 'natural_questions_with_dpr_para', 'physical_iqa', 
            'social_iqa', 'ambigqa', 'squad1_1', 'squad2', 'boolq', 'commonsenseqa', 
            'qasc', 'qasc_with_ir', 'winogrande_xl', 'mctest_corrected_the_separator', 
            'contrast_sets_drop_dedup', 'drop_dedup', 'contrast_sets_boolq_dedup', 
            'boolq_np_dedup', 'ambigqa_dedup', 'social_iqa_dedup', 'quoref_dedup', 
            'contrast_sets_quoref_dedup', 'contrast_sets_ropes_dedup', 
            'drop_dedup_lowsim_uqa', 'contrast_sets_drop_dedup_lowsim_uqa', 
            'physical_iqa_lowsim_uqa', 'social_iqa_dedup_lowsim_uqa', 
            'commonsenseqa_lowsim_uqa', 'qasc_lowsim_uqa', 'qasc_with_ir_lowsim_uqa', 
            'ropes_lowsim_uqa', 'newsqa_lowsim_uqa', 'drop_dedup_lowsim_tdnd', 
            'contrast_sets_drop_dedup_lowsim_tdnd', 'physical_iqa_lowsim_tdnd', 
            'social_iqa_dedup_lowsim_tdnd', 'commonsenseqa_lowsim_tdnd', 
            'qasc_lowsim_tdnd', 'qasc_with_ir_lowsim_tdnd', 'ropes_lowsim_tdnd', 
            'newsqa_lowsim_tdnd', 'strategy_qa', 'cwwv', 'atomic',
            'musique_qa', 'musique_qa_paras', 'musique_mu_dev_qa', 'musique_mu_dev_qa_paras',
            'musique_qa_decomp_ans', 'musique_qa_paras_decomp_ans', 'musique_mu_dev_qa_decomp_ans',
            'musique_mu_dev_qa_paras_decomp_ans', 'musique_mu_dev_qa_decomp_context']

test_eval_all = ['openbookqa', 'openbookqa_with_ir', 'arc_easy', 'arc_easy_with_ir', 'arc_hard', 
             'arc_hard_with_ir', 'ai2_science_elementary', 'ai2_science_middle', 'race_string', 
             'narrativeqa', 'mmlu_electrical_engineering_test', 'mmlu_high_school_statistics_test', 
             'mmlu_college_chemistry_test', 'mmlu_econometrics_test', 'mmlu_high_school_world_history_test', 
             'mmlu_management_test', 'mmlu_business_ethics_test', 'mmlu_jurisprudence_test', 
             'mmlu_world_religions_test', 'mmlu_global_facts_test', 'mmlu_college_medicine_test', 
             'mmlu_computer_security_test', 'mmlu_us_foreign_policy_test', 'mmlu_international_law_test', 
             'mmlu_nutrition_test', 'mmlu_philosophy_test', 'mmlu_virology_test', 
             'mmlu_high_school_government_and_politics_test', 'mmlu_clinical_knowledge_test', 
             'mmlu_college_computer_science_test', 'mmlu_college_physics_test', 'mmlu_anatomy_test', 
             'mmlu_college_biology_test', 'mmlu_high_school_us_history_test', 'mmlu_moral_scenarios_test', 
             'mmlu_public_relations_test', 'mmlu_high_school_psychology_test', 
             'mmlu_professional_psychology_test', 'mmlu_high_school_chemistry_test', 
             'mmlu_high_school_geography_test', 'mmlu_medical_genetics_test', 'mmlu_college_mathematics_test', 
             'mmlu_high_school_microeconomics_test', 'mmlu_machine_learning_test', 
             'mmlu_professional_law_test', 'mmlu_miscellaneous_test', 'mmlu_moral_disputes_test', 
             'mmlu_high_school_computer_science_test', 'mmlu_high_school_macroeconomics_test', 
             'mmlu_elementary_mathematics_test', 'mmlu_professional_accounting_test', 'mmlu_astronomy_test', 
             'mmlu_high_school_physics_test', 'mmlu_logical_fallacies_test', 'mmlu_human_aging_test', 
             'mmlu_high_school_european_history_test', 'mmlu_sociology_test', 'mmlu_security_studies_test', 
             'mmlu_high_school_mathematics_test', 'mmlu_high_school_biology_test', 'mmlu_prehistory_test', 
             'mmlu_conceptual_physics_test', 'mmlu_professional_medicine_test', 'mmlu_marketing_test', 
             'mmlu_abstract_algebra_test', 'mmlu_human_sexuality_test', 'mmlu_formal_logic_test', 
             'mmlu_us_foreign_policy_test_dedup', 'mmlu_college_physics_test_dedup', 
             'mmlu_public_relations_test_dedup', 'mmlu_high_school_psychology_test_dedup', 
             'mmlu_professional_psychology_test_dedup', 'mmlu_elementary_mathematics_test_dedup', 
             'mmlu_elementary_to_college_math_test', 'mmlu_elementary_to_college_math_test_lowsim_uqa', 
             'mmlu_elementary_to_college_math_test_lowsim_tdnd']


#Map dataset types to relevant metrics to calculate and preferred reporting metric
metric_groups = {
    'EX': {'compute':['EM', 'F1'], 'prefer':'F1'},
    'AB': {'compute':['EM', 'F1', 'RL'], 'prefer':'RL'},
    'MC': {'compute':['EM', 'F1', 'SS'], 'prefer':'SS'},
    'YN': {'compute':['EM', 'F1', 'YN'], 'prefer':'YN'},
    'DC': {'compute':['EM', 'F1A', 'F1DA', 'SARID', 'SARIDA'], 'prefer':'F1A'}  #F1A = F1 on answer only so comparable to other datasets
}

########################################################
#Map datasets to dataset types and optional override for preferred reporting metric (must be one in above 'compute' key)
#NOTE: Any dataset used in evaluation metrics calculation and/or similarity calculation must be added to dataset_attribs
########################################################

dataset_attribs = {
    'ai2_science_elementary': {'type':'MC', 'prefer':''},
    'ai2_science_middle': {'type':'MC', 'prefer':''},
    'ambigqa': {'type':'AB', 'prefer':''},
    'ambigqa_dedup': {'type':'AB', 'prefer':''},
    'arc_easy': {'type':'MC', 'prefer':''},
    'arc_easy_dev': {'type':'MC', 'prefer':''},
    'arc_easy_with_ir': {'type':'MC', 'prefer':''},
    'arc_easy_with_ir_dev': {'type':'MC', 'prefer':''},
    'arc_hard': {'type':'MC', 'prefer':''},
    'arc_hard_dev': {'type':'MC', 'prefer':''},
    'arc_hard_with_ir': {'type':'MC', 'prefer':''},
    'arc_hard_with_ir_dev': {'type':'MC', 'prefer':''},
    'boolq': {'type':'YN', 'prefer':''},
    'boolq_np': {'type':'YN', 'prefer':''},
    'boolq_np_dedup': {'type':'YN', 'prefer':''},
    'commonsenseqa': {'type':'MC', 'prefer':''},
    'commonsenseqa_test': {'type':'MC', 'prefer':''},
    'contrast_sets_boolq': {'type':'YN', 'prefer':''},
    'contrast_sets_boolq_dedup': {'type':'YN', 'prefer':''},
    'contrast_sets_drop': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_quoref': {'type':'EX', 'prefer':''},
    'contrast_sets_quoref_dedup': {'type':'EX', 'prefer':''},
    'contrast_sets_ropes': {'type':'EX', 'prefer':''},
    'contrast_sets_ropes_dedup': {'type':'EX', 'prefer':''},
    'drop': {'type':'AB', 'prefer':'F1'},
    'drop_dedup': {'type':'AB', 'prefer':'F1'},
    'mctest': {'type':'MC', 'prefer':''},
    'mctest_corrected_the_separator': {'type':'MC', 'prefer':''},
    'multirc': {'type':'YN', 'prefer':''},
    'narrativeqa': {'type':'AB', 'prefer':''},
    'narrativeqa_dev': {'type':'AB', 'prefer':''},
    'natural_questions': {'type':'AB', 'prefer':'EM'},
    'natural_questions_direct_ans': {'type':'AB', 'prefer':'EM'},
    'natural_questions_direct_ans_test': {'type':'AB', 'prefer':'EM'},
    'natural_questions_with_dpr_para': {'type':'AB', 'prefer':'EM'},
    'natural_questions_with_dpr_para_test': {'type':'AB', 'prefer':'EM'},
    'newsqa': {'type':'EX', 'prefer':''},
    'openbookqa': {'type':'MC', 'prefer':''},
    'openbookqa_dev': {'type':'MC', 'prefer':''},
    'openbookqa_with_ir': {'type':'MC', 'prefer':''},
    'openbookqa_with_ir_dev': {'type':'MC', 'prefer':''},
    'physical_iqa': {'type':'MC', 'prefer':''},
    'qasc': {'type':'MC', 'prefer':''},
    'qasc_test': {'type':'MC', 'prefer':''},
    'qasc_with_ir': {'type':'MC', 'prefer':''},
    'qasc_with_ir_test': {'type':'MC', 'prefer':''},
    'quoref': {'type':'EX', 'prefer':''},
    'quoref_dedup': {'type':'EX', 'prefer':''},
    'race_string': {'type':'MC', 'prefer':''},
    'race_string_dev': {'type':'MC', 'prefer':''},
    'ropes': {'type':'EX', 'prefer':''},
    'social_iqa': {'type':'MC', 'prefer':''},
    'social_iqa_dedup': {'type':'MC', 'prefer':''},
    'squad1_1': {'type':'EX', 'prefer':''},
    'squad2': {'type':'EX', 'prefer':''},
    'winogrande_l': {'type':'MC', 'prefer':''},
    'winogrande_m': {'type':'MC', 'prefer':''},
    'winogrande_s': {'type':'MC', 'prefer':''},
    'winogrande_test': {'type':'MC', 'prefer':''},
    'winogrande_xl': {'type':'MC', 'prefer':''},
    'winogrande_xs': {'type':'MC', 'prefer':''},
    'mmlu_elementary_mathematics_test': {'type':'MC', 'prefer':''},
    'mmlu_elementary_mathematics_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_business_ethics_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_accounting_test': {'type':'MC', 'prefer':''},
     'mmlu_college_mathematics_test': {'type':'MC', 'prefer':''},
     'mmlu_public_relations_test': {'type':'MC', 'prefer':''},
     'mmlu_public_relations_test_dedup': {'type':'MC', 'prefer':''}, 
     'mmlu_philosophy_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_government_and_politics_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_medicine_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_biology_test': {'type':'MC', 'prefer':''},
     'mmlu_moral_disputes_test': {'type':'MC', 'prefer':''},
     'mmlu_moral_scenarios_test': {'type':'MC', 'prefer':''},
     'mmlu_clinical_knowledge_test': {'type':'MC', 'prefer':''},
     'mmlu_college_computer_science_test': {'type':'MC', 'prefer':''},
     'mmlu_jurisprudence_test': {'type':'MC', 'prefer':''},
     'mmlu_logical_fallacies_test': {'type':'MC', 'prefer':''},   
     'mmlu_us_foreign_policy_test': {'type':'MC', 'prefer':''},
     'mmlu_us_foreign_policy_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_high_school_statistics_test': {'type':'MC', 'prefer':''},
     'mmlu_virology_test': {'type':'MC', 'prefer':''},
     'mmlu_formal_logic_test': {'type':'MC', 'prefer':''},
     'mmlu_security_studies_test': {'type':'MC', 'prefer':''},
     'mmlu_machine_learning_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_us_history_test': {'type':'MC', 'prefer':''},
     'mmlu_world_religions_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_chemistry_test': {'type':'MC', 'prefer':''},
     'mmlu_prehistory_test': {'type':'MC', 'prefer':''},
     'mmlu_electrical_engineering_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_european_history_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_psychology_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_psychology_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_high_school_world_history_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_geography_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_computer_science_test': {'type':'MC', 'prefer':''},
     'mmlu_human_aging_test': {'type':'MC', 'prefer':''},
     'mmlu_marketing_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_mathematics_test': {'type':'MC', 'prefer':''},
     'mmlu_conceptual_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_abstract_algebra_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_psychology_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_psychology_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_management_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_macroeconomics_test': {'type':'MC', 'prefer':''},
     'mmlu_sociology_test': {'type':'MC', 'prefer':''},
     'mmlu_nutrition_test': {'type':'MC', 'prefer':''},
     'mmlu_college_biology_test': {'type':'MC', 'prefer':''},
     'mmlu_professional_law_test': {'type':'MC', 'prefer':''},
     'mmlu_astronomy_test': {'type':'MC', 'prefer':''},
     'mmlu_college_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_physics_test_dedup': {'type':'MC', 'prefer':''},
     'mmlu_miscellaneous_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_microeconomics_test': {'type':'MC', 'prefer':''},
     'mmlu_computer_security_test': {'type':'MC', 'prefer':''},
     'mmlu_international_law_test': {'type':'MC', 'prefer':''},
     'mmlu_global_facts_test': {'type':'MC', 'prefer':''},
     'mmlu_human_sexuality_test': {'type':'MC', 'prefer':''},
     'mmlu_econometrics_test': {'type':'MC', 'prefer':''},
     'mmlu_anatomy_test': {'type':'MC', 'prefer':''},
     'mmlu_medical_genetics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_medicine_test': {'type':'MC', 'prefer':''},
     'mmlu_high_school_physics_test': {'type':'MC', 'prefer':''},
     'mmlu_college_chemistry_test': {'type':'MC', 'prefer':''},
     'mmlu_elementary_to_college_math_test': {'type':'MC', 'prefer':''},
     'synthetic_numeric': {'type':'AB', 'prefer':'EM'},
     'synthetic_textual': {'type':'AB', 'prefer':'EM'},
    'drop_dedup_lowsim_uqa': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup_lowsim_uqa': {'type':'AB', 'prefer':'F1'},
    'mmlu_elementary_to_college_math_test_lowsim_uqa': {'type':'MC', 'prefer':''},
    'physical_iqa_lowsim_uqa': {'type':'MC', 'prefer':''},
    'social_iqa_dedup_lowsim_uqa': {'type':'MC', 'prefer':''},
    'commonsenseqa_lowsim_uqa': {'type':'MC', 'prefer':''},
    'qasc_lowsim_uqa': {'type':'MC', 'prefer':''},
    'qasc_with_ir_lowsim_uqa': {'type':'MC', 'prefer':''},
    'ropes_lowsim_uqa': {'type':'EX', 'prefer':''},
    'newsqa_lowsim_uqa': {'type':'EX', 'prefer':''},
    'drop_dedup_lowsim_tdnd': {'type':'AB', 'prefer':'F1'},
    'contrast_sets_drop_dedup_lowsim_tdnd': {'type':'AB', 'prefer':'F1'},
    'mmlu_elementary_to_college_math_test_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'physical_iqa_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'social_iqa_dedup_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'commonsenseqa_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'qasc_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'qasc_with_ir_lowsim_tdnd': {'type':'MC', 'prefer':''},
    'ropes_lowsim_tdnd': {'type':'EX', 'prefer':''},
    'newsqa_lowsim_tdnd': {'type':'EX', 'prefer':''},
    'strategy_qa': {'type':'MC', 'prefer':''},
    'cwwv': {'type':'MC', 'prefer':''},
    'cskg': {'type':'MC', 'prefer':''},
    'atomic': {'type':'MC', 'prefer':''},
    'musique_qa': {'type':'EX', 'prefer':''},
    'musique_qa_paras': {'type':'EX', 'prefer':''},
    'musique_mu_dev_qa': {'type':'EX', 'prefer':''},
    'musique_mu_dev_qa_paras': {'type':'EX', 'prefer':''},
    'musique_qa_decomp_ans': {'type':'DC', 'prefer':''},
    'musique_qa_paras_decomp_ans': {'type':'DC', 'prefer':''},
    'musique_mu_dev_qa_decomp_ans': {'type':'DC', 'prefer':''},
    'musique_mu_dev_qa_paras_decomp_ans': {'type':'DC', 'prefer':''},
    'musique_qa_plus_qa_decomp_ans': {'type':'EX', 'prefer':''},
    'musique_qa_plus_qa_decomp_ans_full': {'type':'EX', 'prefer':''},
    'musique_qa_paras_plus_qa_paras_decomp_ans': {'type':'EX', 'prefer':''},
    'musique_qa_paras_plus_qa_paras_decomp_ans_full': {'type':'EX', 'prefer':''},
    'musique_mu_dev_qa_decomp_context': {'type':'EX', 'prefer':''},
    'strategy_qa_od_ans': {'type':'YN', 'prefer':''},
    'strategy_qa_expl_ans': {'type':'YN', 'prefer':''},
    'qasc_mc_ans': {'type':'MC', 'prefer':''},
    'worldtree_mc_ans': {'type':'MC', 'prefer':''},
    'musique_mu_dev_qa_expl_ans': {'type':'EX', 'prefer':''},
    'nq_open_od_ans': {'type':'EX', 'prefer':'EM'}, 
    'arc_da_expl_ans': {'type':'EX', 'prefer':''}, 
    'arc_da_od_ans': {'type':'EX', 'prefer':''},
    }

unifiedqa_base_train = ["narrativeqa", "ai2_science_middle", "ai2_science_elementary",
                        "arc_hard", "arc_easy", "mctest_corrected_the_separator",
                        "squad1_1", "squad2", "boolq", "race_string", "openbookqa"]

########################################################
# where same train dataset in difft formats, just calc sim against one format and map similarity for others against that...
########################################################

replace_sim_with = {'cwwv_selfsvised': 'cwwv', 
                'atomic_selfsvised': 'atomic', 
                'cwwv_premask_selfsvised': 'cwwv', 
                'atomic_premask_selfsvised': 'atomic',
                'musique_qa_decomp_ans': 'musique_qa',
                'musique_qa_plus_qa_decomp_ans': 'musique_qa',
                'musique_qa_paras_decomp_ans': 'musique_qa_paras',
                'musique_qa_paras_plus_qa_paras_decomp_ans': 'musique_qa_paras',
                'musique_qa_plus_qa_decomp_ans_full': 'musique_qa_full',
                'musique_qa_paras_plus_qa_paras_decomp_ans_full': 'musique_qa_paras_full',
                'musique_mu_dev_qa_decomp_ans': 'musique_mu_dev_qa',
                'musique_mu_dev_qa_paras_decomp_ans': 'musique_mu_dev_qa_paras',
                'strategy_qa_od_ans': 'strategy_qa',
                'strategy_qa_expl_ans': 'strategy_qa',
                'qasc_mc_ans': 'qasc',
                'musique_mu_dev_qa_expl_ans': 'musique_mu_dev_qa',
                }


########################################################
# Sets of Eval datasets used in generating reports...
########################################################

# Not used
unifiedqa_unseen_1 = [
    'newsqa',
    'quoref',
    'contrast_sets_quoref',
    'ropes',
    'contrast_sets_ropes',
    'drop',
    'contrast_sets_drop',
    'qasc',
    'commonsenseqa',
    'boolq_np',
    'contrast_sets_boolq',
    'multirc'
    ]   

# Not used
unifiedqa_unseen_2 = [
    'qasc_with_ir',
    'winogrande_xl',
    'physical_iqa',
    'social_iqa',
    'natural_questions_with_dpr_para'
    ]

# Not used
unifiedqa_unseen_3 = [
    'drop_dedup',
    'contrast_sets_drop_dedup',
    'mmlu_elementary_mathematics_test_dedup',
    'mmlu_high_school_mathematics_test',
    'mmlu_high_school_statistics_test',
    'mmlu_college_mathematics_test',
    'physical_iqa',
    'social_iqa_dedup',
    'commonsenseqa',
    'qasc',
    'qasc_with_ir',
    'ropes',
    'newsqa'
    ]

# The 10 unseen evaluation datasets used in our paper
unifiedqa_unseen_4 = [
    'drop_dedup',
    'contrast_sets_drop_dedup',
    'mmlu_elementary_to_college_math_test',
    'physical_iqa',
    'social_iqa_dedup',
    'commonsenseqa',
    'qasc',
    'qasc_with_ir',
    'ropes',
    'newsqa',
    'strategy_qa',
    'musique_qa',
    'musique_qa_decomp_ans',
    'musique_mu_dev_qa',
    'musique_mu_dev_qa_decomp_ans',
    'musique_qa_paras',
    'musique_qa_paras_decomp_ans',
    'musique_mu_dev_qa_paras',
    'musique_mu_dev_qa_paras_decomp_ans',
    'musique_mu_dev_qa_decomp_context',
    'strategy_qa_od_ans',
    'nq_open_od_ans',
    'arc_da_od_ans',
    'strategy_qa_expl_ans',
    'qasc_mc_ans',
    'musique_mu_dev_qa_expl_ans',
    'worldtree_mc_ans',
    'arc_da_expl_ans',
    ]


# Note: This is only used in create_least_similar_versions.py and check_least_similar_answer.py
# These two py files have now been modified s.t. if a datasets isn't in this map, the file defaults to 'dev.tsv'..
# So only need to add datasets to this map if 'test.tsv' is the one needed..
unifiedqa_unseen_4_map = {
    'drop_dedup': 'dev.tsv',
    'contrast_sets_drop_dedup': 'dev.tsv',
    'mmlu_elementary_to_college_math_test': 'test.tsv',
    'physical_iqa': 'dev.tsv',
    'social_iqa_dedup': 'dev.tsv',
    'commonsenseqa': 'dev.tsv',
    'qasc': 'dev.tsv',
    'qasc_with_ir': 'dev.tsv',
    'ropes': 'dev.tsv',
    'newsqa': 'dev.tsv',
    'strategy_qa': 'dev.tsv',
    'musique_qa': 'dev.tsv',
    'musique_qa_paras': 'dev.tsv',
    'musique_mu_dev_qa': 'dev.tsv',
    'musique_mu_dev_qa_paras': 'dev.tsv',
    'musique_qa_decomp_ans': 'dev.tsv',
    'musique_qa_paras_decomp_ans': 'dev.tsv',
    'musique_mu_dev_qa_decomp_ans': 'dev.tsv',
    'musique_mu_dev_qa_paras_decomp_ans': 'dev.tsv',
    'musique_mu_dev_qa_decomp_context': 'dev.tsv',
    'worldtree_mc_ans': 'test.tsv',
    'nq_open_od_ans': 'test.tsv',
    'arc_da_od_ans': 'test.tsv',
    'arc_da_expl_ans': 'test.tsv',
    }



# The filtered versions of the 10 unseen datasets used in our paper
unifiedqa_unseen_6 = [
    'drop_dedup_lowsim_tdnd',
    'contrast_sets_drop_dedup_lowsim_tdnd',
    'mmlu_elementary_to_college_math_test_lowsim_tdnd',
    'physical_iqa_lowsim_tdnd',
    'social_iqa_dedup_lowsim_tdnd',
    'commonsenseqa_lowsim_tdnd',
    'qasc_lowsim_tdnd',
    'qasc_with_ir_lowsim_tdnd',
    'ropes_lowsim_tdnd',
    'newsqa_lowsim_tdnd'
    ]


# datasets unifiedqa trained on (only the non-IR versions but treating w/IR as "seen") 
# but only those with a separate labelled test set
unifiedqa_seen_1 = [
    'openbookqa',
    'openbookqa_with_ir',
    'arc_easy',
    'arc_easy_with_ir',
    'arc_hard',
    'arc_hard_with_ir',
    'race_string',
    'ai2_science_elementary',
    'ai2_science_middle',
    'strategy_qa',
    'cwwv',
    'atomic'
    ]

# The 57 mmlu datasets. 
mmlu_unseen_1 = [
     'mmlu_elementary_mathematics_test',
     'mmlu_business_ethics_test',
     'mmlu_professional_accounting_test',
     'mmlu_college_mathematics_test',
     'mmlu_public_relations_test',
     'mmlu_philosophy_test',
     'mmlu_high_school_government_and_politics_test',
     'mmlu_professional_medicine_test',
     'mmlu_high_school_biology_test',
     'mmlu_moral_disputes_test',
     'mmlu_moral_scenarios_test',
     'mmlu_clinical_knowledge_test',
     'mmlu_college_computer_science_test',
     'mmlu_jurisprudence_test',
     'mmlu_logical_fallacies_test',
     'mmlu_us_foreign_policy_test',
     'mmlu_high_school_statistics_test',
     'mmlu_virology_test',
     'mmlu_formal_logic_test',
     'mmlu_security_studies_test',
     'mmlu_machine_learning_test',
     'mmlu_high_school_us_history_test',
     'mmlu_world_religions_test',
     'mmlu_high_school_chemistry_test',
     'mmlu_prehistory_test',
     'mmlu_electrical_engineering_test',
     'mmlu_high_school_european_history_test',
     'mmlu_high_school_psychology_test',
     'mmlu_high_school_world_history_test',
     'mmlu_high_school_geography_test',
     'mmlu_high_school_computer_science_test',
     'mmlu_human_aging_test',
     'mmlu_marketing_test',
     'mmlu_high_school_mathematics_test',
     'mmlu_conceptual_physics_test',
     'mmlu_abstract_algebra_test',
     'mmlu_professional_psychology_test',
     'mmlu_management_test',
     'mmlu_high_school_macroeconomics_test',
     'mmlu_sociology_test',
     'mmlu_nutrition_test',
     'mmlu_college_biology_test',
     'mmlu_professional_law_test',
     'mmlu_astronomy_test',
     'mmlu_college_physics_test',
     'mmlu_miscellaneous_test',
     'mmlu_high_school_microeconomics_test',
     'mmlu_computer_security_test',
     'mmlu_international_law_test',
     'mmlu_global_facts_test',
     'mmlu_human_sexuality_test',
     'mmlu_econometrics_test',
     'mmlu_anatomy_test',
     'mmlu_medical_genetics_test',
     'mmlu_college_medicine_test',
     'mmlu_high_school_physics_test',
     'mmlu_college_chemistry_test']


########################################################
# Eval Datasets q[+mc]->a to take as input and output to new dir as q[+mc]+e->a i.e to generate e for
# New dynamically created datasets will be created as e.g /UQA_DIR/qasc_dyn_expl_ans_modeloutputdir_timestamp
# NOTE: Each of these datasets must be in dataset_attribs and in one of dev_eval or test_eval as using the dev.tsv or test.tsv will be inferred from this
# NOTE2: The dynamically created versions will be added "on the fly" to dev_eval and test_eval and to a special "unseen" dataset
########################################################

create_datasets_dynamic = ['musique_mu_dev_qa', 'strategy_qa_od_ans', 'qasc', 
                           'arc_easy', 'arc_hard', 'nq_open_od_ans', 'arc_da_od_ans' ]

# Not used
unifiedqa_unseen_5 = []

def list_files_pattern(dirtolist, pattern='*'):
    """ Returns a list of files in a dictionary matching a pattern
    """
    return [file for file in os.listdir(dirtolist) if fnmatch.fnmatch(file, pattern)]


for dset in create_datasets_dynamic:
    curr_dir = os.path.join(UQA_DIR, dset)
    if not os.path.exists(curr_dir):
        print(f"ERROR: dataset_attributes.py: {curr_dir} doesn't exist! Skipping...")
        continue
    attrib = dataset_attribs.get(dset)
    if attrib is None:
        print(f"ERROR: dataset_attributes.py: {attrib} hasn't been added to dataset_attribs! Skipping...")
        continue
    if dset in dev_eval:
        evaltype = 'dev'
    elif dset in test_eval:
        evaltype = 'test'
    else:
        print(f"ERROR: dataset_attributes.py: {dset} hasn't been added to one of dev_eval or test_eval! Skipping...")
        continue
    dyn_dsets = list_files_pattern(UQA_DIR, dset+SVISED_EXPL_ANS+'*')  
    if dyn_dsets == []:
        print(f"WARNING: dataset_attributes.py: No dynamically created datasets from {dset} were found in {UQA_DIR}! Skipping...")
        continue
    for dyn_ds in dyn_dsets:
        dataset_attribs[dyn_ds] = attrib
        if evaltype == 'dev':
            if dyn_ds not in dev_eval:
                dev_eval.append(dyn_ds)
        else:
            if dyn_ds not in test_eval:
                test_eval.append(dyn_ds)
        unifiedqa_unseen_5.append(dyn_ds)    
unifiedqa_unseen_5.sort()        

