#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:58:08 2023

@author: tim hartill

Extract results files into one place then run significance tests

"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

import eval_metrics


results_dict = {'BASE': 'out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/eval_metrics.json',
                'BASE_RATD': 'out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/eval_metrics.json',
                'BASE_GR': 'out/mdr/logs/UQA_s14_v3_from_s9_v2_base_add_exp_gold_only/eval_metrics.json',
                'BASE_RATD_GR': 'out/mdr/logs/UQA_s14_v4_from_s9_v2_base_ratd_add_exp_gold_only/eval_metrics.json'}

# StrategyQA, CommonsenseQA, ARC-DA, IIRC, Musique dev
# RATD_Iter_only, GR_RATD_Iter_only, GR_RATD_Rat_only, GR_RATD_Rat_Iter_Naive, GR_RATD_Rat_Iter_Best, GR_Iter_only, GR_Rat_only, GR_Rat_Iter_Naive
datasets_dict = {'BASE_RATD': {'Iter_only': ['strategy_qa_bigbench_fullwiki_bs150_noimplrel',
                                             'commonsenseqa_fullwiki_bs150_noimplrel',
                                             'arc_da_od_ans_fullwiki_bs150',
                                             'iirc_initial_context_fullwiki_bs150',
                                             'musique_mu_dev_odv2_fullwiki_bs150']
                               },  #iterator only
                'BASE_GR': {'Iter_only': ['strategy_qa_bigbench_fullwiki_bs150_noimplrel',
                                             'commonsenseqa_fullwiki_bs150_noimplrel',
                                             'arc_da_od_ans_fullwiki_bs150',
                                             'iirc_initial_context_fullwiki_bs150',
                                             'musique_mu_dev_odv2_fullwiki_bs150'],
                            'Rat_only': ['strategy_qa_bigbench_llm_expl_svint8',
                                         'commonsenseqa_llm_expl_svint8',
                                         'arc_da_od_ans_llm_expl_svint8',
                                         'iirc_initial_context_llm_expl_svint8',
                                         'musique_mu_dev_odv2_llm_expl_svint8'],
                            'Iter_Rat_naive': ['strategy_qa_bigbench_llm_expl_fullwiki_bs150_noimplrel_svint8',
                                               'commonsenseqa_llm_expl_fullwiki_bs150_noimplrel_svint8',
                                               'arc_da_od_ans_llm_expl_fullwiki_bs150_svint8',
                                               'iirc_initial_context_llm_expl_fullwiki_bs150_svint8',
                                               'musique_mu_dev_odv2_llm_expl_fullwiki_bs150_svint8'],
                            'Iter_Rat_bestrr': ['strategy_qa_bigbench_svint8_v3t8_iterthresh_llm_expl_rr_fullwiki_over_rr0.75',
                                                'commonsenseqa_svint8_v3t8_iterthresh_llm_expl_rr_fullwiki_over_rr0.75',
                                                'arc_da_od_ans_svint8_v3t8_iterthresh_llm_expl_rr_fullwiki_over_rr0.75',
                                                'iirc_initial_context_svint8_v3t8_iterthresh_llm_expl_rr_fullwiki_over_rr0.75',
                                                'musique_mu_dev_odv2_svint8_v3t8_iterthresh_llm_expl_rr_fullwiki_over_rr0.75',]
                            },
                'BASE_RATD_GR': {'Iter_only': ['strategy_qa_bigbench_fullwiki_bs150_noimplrel',
                                             'commonsenseqa_fullwiki_bs150_noimplrel',
                                             'arc_da_od_ans_fullwiki_bs150',
                                             'iirc_initial_context_fullwiki_bs150',
                                             'musique_mu_dev_odv2_fullwiki_bs150'],
                            'Rat_only': ['strategy_qa_bigbench_llm_expl_svint8',
                                         'commonsenseqa_llm_expl_svint8',
                                         'arc_da_od_ans_llm_expl_svint8',
                                         'iirc_initial_context_llm_expl_svint8',
                                         'musique_mu_dev_odv2_llm_expl_svint8'],
                            'Iter_Rat_naive': ['strategy_qa_bigbench_llm_expl_fullwiki_bs150_noimplrel_svint8',
                                               'commonsenseqa_llm_expl_fullwiki_bs150_noimplrel_svint8',
                                               'arc_da_od_ans_llm_expl_fullwiki_bs150_svint8',
                                               'iirc_initial_context_llm_expl_fullwiki_bs150_svint8',
                                               'musique_mu_dev_odv2_llm_expl_fullwiki_bs150_svint8'],
                            'Iter_Rat_bestrr': ['strategy_qa_bigbench_svint8_v3t8_llm_expl_rr0.9_fullwiki_rr0.9',
                                                'commonsenseqa_svint8_v3t8_llm_expl_rr0.9_fullwiki_rr0.9',
                                                'arc_da_od_ans_svint8_v3t8_llm_expl_rr0.9_fullwiki_rr0.9',
                                                'iirc_initial_context_svint8_v3t8_llm_expl_rr0.9_fullwiki_rr0.9',
                                                'musique_mu_dev_odv2_svint8_v3t8_llm_expl_rr0.9_fullwiki_rr0.9']
                            }
                }



outdir = os.path.join(eval_metrics.LDATA, 'out/mdr/logs/eval_outputs/main_eval_metrics')
os.makedirs(outdir, exist_ok=True)

results = {}
results_agg = {}  # concatenate all scores for a model/context type together
for s in results_dict:
    results_dict[s] = os.path.join(eval_metrics.LDATA, results_dict[s] )
    assert os.path.exists(results_dict[s])
    outfile = os.path.join(outdir, s + '_eval_metrics.json')
    shutil.copyfile(results_dict[s], outfile)
    print(f"Copied: {results_dict[s]} to {outfile}")
     
    #Copied: /large_data/thar011/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/eval_metrics.json to /large_data/thar011/out/mdr/logs/eval_outputs/main_eval_metrics/BASE_eval_metrics.json
    #Copied: /large_data/thar011/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/eval_metrics.json to /large_data/thar011/out/mdr/logs/eval_outputs/main_eval_metrics/BASE_RATD_eval_metrics.json
    #Copied: /large_data/thar011/out/mdr/logs/UQA_s14_v3_from_s9_v2_base_add_exp_gold_only/eval_metrics.json to /large_data/thar011/out/mdr/logs/eval_outputs/main_eval_metrics/BASE_GR_eval_metrics.json
    #Copied: /large_data/thar011/out/mdr/logs/UQA_s14_v4_from_s9_v2_base_ratd_add_exp_gold_only/eval_metrics.json to /large_data/thar011/out/mdr/logs/eval_outputs/main_eval_metrics/BASE_RATD_GR_eval_metrics.json
    if s != 'BASE':
        metrics = eval_metrics.DatasetMetrics(outfile)
        results[s] = {}
        #results_agg[s] = {}
        for ct in datasets_dict[s]:
            results[s][ct] = {}
            model_context = s + '_' + ct
            results_agg[model_context] = []
            for ds in datasets_dict[s][ct]:
                metricstr = metrics.get_pref_metric(ds)
                assert metricstr != 'NA'
                scores = metrics.get_value(ds, metricstr, key='scores')  #List of metric scores for this dataset on this model
                results[s][ct][ds] = [float(score) for score in scores]
                print(f"{s} {ct} {ds} count: {len(results[s][ct][ds])}")
                results_agg[model_context] += results[s][ct][ds]

print(results_agg.keys())     # dict_keys(['BASE_RATD_Iter_only', 'BASE_GR_Iter_only', 'BASE_GR_Rat_only', 'BASE_GR_Iter_Rat_naive', 'BASE_GR_Iter_Rat_bestrr', 'BASE_RATD_GR_Iter_only', 'BASE_RATD_GR_Rat_only', 'BASE_RATD_GR_Iter_Rat_naive', 'BASE_RATD_GR_Iter_Rat_bestrr'])           

# 	Add BLOOM_Ans_only	SV_Ans_only	BLOOM_COT	SV_COT	

df = pd.DataFrame.from_dict(results_agg)

sig = autorank(df, alpha=0.05, verbose=True)  # confidence level % = 1.0 - alpha
print(sig)

plot_stats(sig, allow_insignificant=False)
plot_stats(sig, allow_insignificant=True)

latex_table(sig)

create_report(sig)
