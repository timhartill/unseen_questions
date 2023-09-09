#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:29:20 2023

@author: tim hartill

Autorank significance testing

https://sherbold.github.io/autorank/

@article{Herbold2020,
  doi = {10.21105/joss.02173},
  url = {https://doi.org/10.21105/joss.02173},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {48},
  pages = {2173},
  author = {Steffen Herbold},
  title = {Autorank: A Python package for automated ranking of classifiers},
  journal = {Journal of Open Source Software}
}

data_all format:

	Model_Context, model1, model2, ...
0	SQA, ...
1	CSQA, ...
2	ARCDA, ...
3	IIRC, ...
4	Musique, ...
5	Mean, ...



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

data_all_file = '/large_data/thar011/out/mdr/logs/eval_outputs/s11/combining_datasources_main_table.csv'

data_all = pd.read_csv(data_all_file)
data = data_all.drop(5,axis='index')
data = data.drop('Model_Context', axis='columns')
pd.set_option('display.max_columns', 15)


#np.random.seed(42)
#pd.set_option('display.max_columns', 7)
#std = 0.3
#means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
#sample_size = 5
#data = pd.DataFrame()
#for i, mean in enumerate(means):
#    data['popxxxx_%i' % i] = np.random.normal(mean, std, sample_size).clip(0, 1)

result = autorank(data, alpha=0.05, verbose=True)  # confidence level % = 1.0 - alpha
print(result)

plot_stats(result, allow_insignificant=False)
plot_stats(result, allow_insignificant=True)

latex_table(result)

data_2_difft = data[['RATD_Iter_only', 'GR_RATD_Rat_Iter_BestPerDS']]

result = autorank(data_2_difft, alpha=0.05, verbose=True)
print(result)

plot_stats(result, allow_insignificant=False)
