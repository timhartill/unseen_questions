#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:39:12 2021

@author: tim hartill

Test different prompting formats with GPT-J


Notes: 
    Liu params: few-shot (k)=5, generate inferences (m)=20, nucleus p = 0.5, max 64 tokens or when hit \n
    West params: few-shot (k)=10, generate inferences (m)=10, nucleus p = 0.9, number examples, freq penalty=0.5

"""
import os
import numpy as np

from utils import load_model, run_model, empty_cache, load_uqa_supervised, return_sublist
from utils import load_prompt_template, load_templates, fill_prompt_template, generate_continuations
from text_processing import format_sentence
import eval_metrics

UQA_DIR = '/data/thar011/data/unifiedqa/'
PROMPT_DIR = os.path.join(UQA_DIR, 'prompts')
cuda_device = 0


model_name = "EleutherAI/gpt-j-6B"
#tokenizer, model = load_model(model_name, checkpoint=None)
tokenizer, model = load_model(model_name, checkpoint=None, cuda_device=cuda_device)


eval_model_name = 'facebook/bart-large'
eval_model_ckpt = '/data/thar011/out/unifiedqa_bart_large_v3/best-model.pt'
tokenizer_eval, model_eval = load_model(eval_model_name, checkpoint=eval_model_ckpt, cuda_device=cuda_device)

tstin = "Answer yes or no: Is 16 greater than 6? Answer:"
tstin = [tstin, tstin]
res = run_model(tstin, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=50, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=2, temperature=0.7,
                output_scores=False, return_dict_in_generate=True)

res_eval = run_model(tstin, model_eval, tokenizer_eval, indiv_digits=False, norm_numbers=False, 
                max_input_length=50, verbose=True,
                lower=True, 
                num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=130,
                output_scores=True, return_dict_in_generate=True)



###############
# QASC
###############

QASC_DEV = os.path.join(UQA_DIR, 'qasc', 'dev.tsv')
QASC_EXPLANATION_FILE = os.path.join(UQA_DIR, 'qasc_mc_ans', 'train.tsv')
qasc_dev = load_uqa_supervised(QASC_DEV, return_parsed=True)
qasc_train_expl = load_uqa_supervised(QASC_EXPLANATION_FILE, ans_lower=False, return_parsed=True)
qasc_train_questions = [format_sentence(s['q_only'].replace('Add Explanation:', '', 1), endchar='') for s in qasc_train_expl]
qasc_train_explanations = [s['context'] for s in qasc_train_expl]
qasc_train_answers = [s['answer'] for s in qasc_train_expl]

num_q = len(qasc_train_expl)
np.random.seed(42)
prompt_indices = np.random.choice(num_q, 100, replace=False)
# array([5914, 5425, 1430, 7324, 4028, 1009, 3172, 2892, 3985, 5023, 4074,
#       1302, 4471, 7541,  554, 6864,  483, 6908, 6159, 5057, 5170, 2199,
#       3837, 2345, 5137, 7331, 4825, 1242, 1882, 5519, 4525, 1730, 5861,
#       6091, 2406, 2302,  233,  794,  866, 3333, 1400, 1744, 7937, 6224,
#       4510, 4922,  932, 3567, 4151, 1737,  318, 2995, 2338, 5513, 7743,
#       1926, 3012, 1575, 4113,  349, 3355, 7716, 4606, 3942, 1010, 3844,
#        239, 6438, 3238, 6879,  748, 6218, 5324, 3149, 1295, 7685, 2867,
#       7977, 3217, 6642, 4270, 7165, 6968, 5815, 6594, 3018, 4394, 2663,
#       6084,  453, 3995, 7780, 6612, 6075, 4668, 5548, 2348, 8088, 4674,
#       6195])
#prompt_rand = prompt_indices[np.random.choice(prompt_indices.shape[0], 20, replace=False)]

qasc_1_fact = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_single_fact_liu_v1.txt')
qasc_2_fact = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_multi_fact_sameliuquestions_v1.txt')
qasc_2_fact_numbered = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_multi_fact_numbered_sameliuquestions_v1.txt')
qasc_5_cleaned = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_5_cleaned.txt')

qasc_2_fact_numbered_vark = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_var_numbered_examples_v1.txt')

p_qasc_1 = fill_prompt_template(qasc_1_fact, query=qasc_dev[0]['q_only'])
p_qasc_2 = fill_prompt_template(qasc_2_fact, query=qasc_dev[0]['q_only'])
p_qasc_2_numbered = fill_prompt_template(qasc_2_fact_numbered, query=qasc_dev[0]['q_only'])
p_qasc_5_cleaned = fill_prompt_template(qasc_2_fact_numbered, query=qasc_dev[0]['q_only'])



res = run_model(p_qasc_1, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

res = run_model(p_qasc_2_numbered, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)
"""
['A climate is described as having wet and dry periods.',
 'Climate is described in terms of temperature and precipitation. It also depends on the interaction between the land and the sea.',
 'Climate is generally described in terms of average weather patterns. E.g. average monthly temperatures.',
 'Humans generally describe climate in terms of temperature and precipitation.  ',
 'Climate is described in terms of temperature, rainfall, and the sea level.',
 'Extreme changes in temperature cause all living things to change their state of life. Warm air is a force of climate change.',
 'It is more common to talk about climate in terms of weather patterns and the atmosphere.',
 'The environment or conditions experienced by a particular organism.',
 'Weather is generally described in terms of temperature, precipitation, and winds. The climate is generally described in terms of temperature and precipitation.',
 'Temperate and oceanic climates are generally defined by seasonal temperature differences. Oceanic climates are warm during the summer and cold during the winter. Temperate climates are warmer during the summer and colder during the winter.']
"""
res = run_model(p_qasc_2_numbered, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.5, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

# test k = 5 random
inputs = return_sublist(qasc_train_questions, prompt_indices[:5])
outputs = return_sublist(qasc_train_explanations, prompt_indices[:5])
p_qasc_2_numbered_k5 = fill_prompt_template(qasc_2_fact_numbered_vark, query=qasc_dev[0]['q_only'], 
                                            example_inputs=inputs, example_outputs=outputs)

res = run_model(p_qasc_2_numbered_k5, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)
"""
['Climate is generally described in terms of temperature, precipitation, and wind.',
 'Climates are the sets of conditions such as atmospheric pressure, water supply, humidity, air temperature,    precipitation, and solar radiation that affect the physical, biological, chemical, and human systems of an area over an extended period of time, such as the present day or the past.',
 'climate is generally described in terms of temperature, pressure, and composition of atmosphere',
 'Climate is generally described in terms of precipitation.',
 'Climate is generally described in terms of the average temperature. However it also involves the pressure of the atmosphere and other factors.',
 'A general descriptions of climate is moisture and temperature.',
 'Change in air temperature. ',
 'A variable climate requires some kind of condition that varies from one location to another. Climate cannot be considered a single thing. There are so many variables like temperature and precipitation that makes climate hard to consider a single thing.',
 'The seasons of the year and long periods of time is the climate of the earth.',
 'Temperature temperature is generally measured in degrees.']
"""

# Not better:
res = run_model(p_qasc_5_cleaned, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)

inputs = return_sublist(qasc_train_questions, prompt_indices[:10])
outputs = return_sublist(qasc_train_explanations, prompt_indices[:10])
p_qasc_2_numbered_k10 = fill_prompt_template(qasc_2_fact_numbered_vark, query=qasc_dev[0]['q_only'], 
                                            example_inputs=inputs, example_outputs=outputs)

res = run_model(p_qasc_2_numbered_k10, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True) #input 442 tokens
"""
["As the atmosphere cools, the polar ice caps shrink and melt. This is called the ice cap's albedo effect.",
 "Climate refers to  Earth's average state of temperature, temperature, precipitation, water vapor pressure, winds, radiation, gases, etc. over a given period of time.",
 'Generally, climatic conditions are described in terms of: temp, humidity, rainfall.',
 'Climatology is a field that studies climate and is used in forecasting.',
 'Temperature extremes. High temperature and low temperature are main characteristics of climate.',
 '',
 'Average temperature averages the weather over a long time period. ',
 'Climate is generally described in terms of temperature, precipitation,  vegetation, drought, storm, typhoon, or tornado.',
 'Climate is generally described in terms of temperature, precipitation and average humidity.',
 'and are key environmental variables. Climate is the average weather conditions of a region. Climate is long-term and temporal and region-specific.']
"""

inputs = return_sublist(qasc_train_questions, prompt_indices[:32])
outputs = return_sublist(qasc_train_explanations, prompt_indices[:32])
p_qasc_2_numbered_k32 = fill_prompt_template(qasc_2_fact_numbered_vark, query=qasc_dev[0]['q_only'], 
                                            example_inputs=inputs, example_outputs=outputs)

res = run_model(p_qasc_2_numbered_k32, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=1500, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True) #input 1332 tokens OOM

inputs = return_sublist(qasc_train_questions, prompt_indices[:15])
outputs = return_sublist(qasc_train_explanations, prompt_indices[:15])
p_qasc_2_numbered_k15 = fill_prompt_template(qasc_2_fact_numbered_vark, query=qasc_dev[0]['q_only'], 
                                            example_inputs=inputs, example_outputs=outputs)

res = run_model(p_qasc_2_numbered_k15, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=1000, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True) #input 660 tokens Output not obviously better


inputs = return_sublist(qasc_train_questions, prompt_indices[:7])
outputs = return_sublist(qasc_train_explanations, prompt_indices[:7])
test_questions = return_sublist(qasc_train_questions, prompt_indices[50:])  #create small "test" set from unused train for evaluating explanation quality based on answer accuracy
test_answers = return_sublist(qasc_train_answers, prompt_indices[50:])
p_qasc_2_numbered_k7 = fill_prompt_template(qasc_2_fact_numbered_vark, query=qasc_dev[0]['q_only'], 
                                            example_inputs=inputs, example_outputs=outputs)

res = run_model(p_qasc_2_numbered_k7, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=1000, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)  #317 tokens

"""
q: 'Climate is generally described in terms of what?'

["Climatology is a scientific and statistical discipline concerned with general circulation of the atmosphere in the Earth's atmosphere.",
 'Climate is generally described in terms of temperature and precipitation.',
 'Climate is generally described in terms of temperature and precipitation.',
 'Cold weather is typically cold temperatures and precipitation. Hot weather is typically hot temperatures and precipitation.',
 'Warm weather will produce flowering.',
 "'Climate is generally described in terms of change in temperature'",
 'It is generally described in terms of heat and moisture.',
 'Climate is generally described in terms of temperature.',
 'Climate is the average of all weather patterns that occur during a year or over a decade or the temperature of the atmosphere surrounding the earth.',
 'The basic facts of climate are: Temperature has a large influence over most weather.']
"""
preds = eval_metrics.preds_basic_filter(res.preds)

# test multi template:
qasc_2_templates = load_templates(['/data/thar011/data/unifiedqa/prompts/qasc_var_numbered_examples_v1.txt',
                                            '/data/thar011/data/unifiedqa/prompts/qasc_5_cleaned.txt'])
test_samples = return_sublist(qasc_train_expl, prompt_indices[98:])
test_questions = [format_sentence(s['q_only'], endchar='') for s in test_samples]
test_answers = [s['answer'] for s in test_samples]
example_inputs = return_sublist(qasc_train_expl, prompt_indices[:7], key='q_only')
example_outputs = return_sublist(qasc_train_expl, prompt_indices[:7], key='context')
qasc_completions = generate_continuations(qasc_2_templates, model, tokenizer, test_questions, verbose=True,
                                          example_inputs=example_inputs, example_outputs=example_outputs, max_input_length=1000, 
                                          do_sample=True, max_new_tokens=64, top_k=0, top_p=0.9, temperature=0.7,
                                          num_return_sequences=10, output_scores=False, return_dict_in_generate=True)

scores0 = eval_metrics.calc_measure_basic(qasc_completions[0][0], compareto=test_questions[0])
scores1 = eval_metrics.calc_measure_basic(qasc_completions[0][1], compareto=test_questions[0])

#Beam search gives error:
qasc_completions = generate_continuations(qasc_2_templates, model, tokenizer, test_questions, verbose=True,
                                          example_inputs=example_inputs, example_outputs=example_outputs, max_input_length=1000, 
                                          num_beams=4, early_stopping=True, min_length=1, max_new_tokens=64,
                                          num_return_sequences=1, output_scores=False, return_dict_in_generate=True)



#test beam search:
p_qasc_5_cleaned = fill_prompt_template(qasc_2_templates[1], query=test_questions[0])
res = run_model(p_qasc_5_cleaned, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=64,
                output_scores=True, return_dict_in_generate=True) 

res = run_model(p_qasc_5_cleaned, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=512, verbose=True, 
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.5, num_return_sequences=10, temperature=0.7,
                output_scores=False, return_dict_in_generate=True) 

preds = eval_metrics.preds_basic_filter(res.preds)  
scores = eval_metrics.calc_measure_basic(preds, compareto=test_questions[0])


#TODO Having both facts seems to work better than just one fact.
#TODO Adding example numbering seems to work better at keeping outputs centered on the query topic
#TODO 0.9 seems to work a bit better than 0.5 - 0.5 has > proportion of blank knowledge and not obviously more diversity
#TODO Need to test with having more examples in the prompt - settle on k=7. Not obviously better using k > 5.

#TODO Can we use the cosine similarity of each sentence in order to determine which ones to select?

#######################
# WORLDTREE
#######################

TEST = os.path.join(UQA_DIR, 'worldtree_mc_ans', 'test.tsv')
EXPLANATION_FILE = os.path.join(UQA_DIR, 'worldtree_mc_ans', 'train.tsv')
dset_dev = load_uqa_supervised(TEST, return_parsed=True)
question = dset_dev[0]['q_only']
gold_expl = dset_dev[0]['context']
dset_train_expl = load_uqa_supervised(EXPLANATION_FILE, ans_lower=False, return_parsed=True)
dset_train_questions = [format_sentence(s['q_only'].replace('Add Explanation:', '', 1), endchar='') for s in dset_train_expl]
dset_train_explanations = [s['context'] for s in dset_train_expl]
dset_train_answers = [s['answer'] for s in dset_train_expl]
num_q = len(dset_train_expl)
np.random.seed(42)
prompt_indices = np.random.choice(num_q, 100, replace=False)
template_fact_numbered_var_k = load_prompt_template('/data/thar011/data/unifiedqa/prompts/qasc_var_numbered_examples_v1.txt')
template_7_clean = load_prompt_template('/data/thar011/data/unifiedqa/prompts/worldtree_7.txt')

inputs = return_sublist(dset_train_questions, prompt_indices[:7])
outputs = return_sublist(dset_train_explanations, prompt_indices[:7])
prompt = fill_prompt_template(template_fact_numbered_var_k, query=question, 
                                            example_inputs=inputs, example_outputs=outputs)
prompt_clean = fill_prompt_template(template_7_clean, query=question, 
                                            example_inputs=inputs, example_outputs=outputs)


res = run_model(prompt, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=1000, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)  # 695 tokens

"""
q: 'What remains in the same location in the sky of the Northern Hemisphere each night?'
gold_expl = 'The North Star does not move every night in the sky in the Northern Hemisphere. Moving changes position. To remain means to not change. If something does not move then that something can be found in the same location. Place; location is synonymous with position.'

["The northern hemisphere is defined as the area of the Earth that is north of the equator. It is the hemisphere that is north of the plane of the equator; the plane that is the intersection of the Earth's equator with the horizon.  The Earth rotates on its axis; also called the Earth's",
 'The Southern hemisphere has a different appearance on Earth. A transition occurs between light and dark during a day. There are patterns in the sky of the Southern Hemisphere. They are called sunsets and sunrises. A sun is a kind of object. A sunrise is the appearance of the Sun in the sky. Dark is the',
 'The 12-hour cycle of the day, is a characteristic of the sky, not the Earth. Clouds are the most noticeable feature of the night sky. The sun rises in the East and sets in the West.',
 'Which of the following factors is the most important to keep the weather outside pleasant? A dry atmosphere; the temperature of the air; cold temperatures.',
 '_____ does not experience variation in intensity over the course of a year. There is no change in the position of stars in the sky over the course of a year. The Great Bear, the Archer and the Pleiades never change position in the sky over the course of a year. There are certain stars which seem to',
 'A star is a kind of object. Objects can be celestial or terrestrial. A star is a kind of celestial object. celestial is the opposite of terrestrial. The heavens are the upper part of the atmosphere; the celestial sphere. The Northern Hemisphere is on the northern side of Earth; the Northern Hemisphere is the portion of Earth',
 'The sun rises in the east and sets in the west. Summer and winter are seasons in the northern hemisphere. North is the direction of the north pole. The north is the direction of the polar axis.',
 '????. If a planet has seasons then it is a kind of planet that rotates. In the Northern Hemisphere the sun sets. It is night. What remains in the same location is the night sky. A planet is an object. The night sky is a place.',
 'The stars are mostly space dust; heat radiating from the sun. The Sun goes around the Earth every night. Diurnal is derived from the Greek word di, meaning day and vernal means spring. The spring is the season when the Sun is closer to the Earth.',
 'The fixed stars are the stars that do not move and rotate with the earth. The constellations are the patterns of stars that change each night in the sky.']
"""
 
res = run_model(prompt_clean, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                max_input_length=1000, verbose=True,
                lower=False, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                max_new_tokens=64, do_sample=True, top_k=0, top_p=0.9, num_return_sequences=10,
                output_scores=True, return_dict_in_generate=True)  # 696 tokens

preds = eval_metrics.preds_basic_filter(res.preds)
scores = eval_metrics.calc_measure_basic(preds, compareto=question)
print(*zip(preds,scores))
"""
('An astronomical object may be moved with respect to the sky, but it always remains in the same place in the sky (or in the same part of the sky). The Sun, Moon, and planets are astronomical objects. A location in the sky is a location in the sky where something is located. A property of something', 0.2456140350877193) ('At noon the sun is directly overhead and is said to be at its highest point in the sky. The crescent moon is above the horizon and is directly above the sun. The sun rises in the east and sets in the west. The moon moves east to west because it is the moon and the earth is spinning.', 0.1) ('The stars are fixed in place, and in the sky. Because it is fixed, it cannot be displaced or moved. Because it cannot be moved, it remains in the same place in the sky each night. Some of the fixed objects that do not move are the sun, moon, stars, and planets. Planets', 0.2711864406779661) ('_______ moon comes up in the east at sunset and goes down in the west at dawn, giving rise to the ______ ___ phenomena. _______', 0.13333333333333333) ('For at least two hundred years, two celestial bodies have had a daily motion in the sky. These are called the Sun and the Moon. What remains in the same location is the Sun. The Sun does not move in the sky. As a result, the motion is called the diurnal (or day) motion', 0.2545454545454546) ('The stars rise in the east,', 0.125) ('An astronomical object that rises in the morning and sets in the evening is called an _______ object. A nautical object that rises in the morning and sets in the evening is called a _______ object. Skyline, i.e. horizon line. a meteor, a shooting star. Sunset, i.e.', 0.0851063829787234) ('The north star is a fixed point of reference used to navigate the sky. Throughout the night, the north star remains in the same place. Throughout the year, the north star remains in the same place. Throughout the night, the north star is a constant reference point in the sky. What remains in the same place is', 0.2909090909090909) ('A spotlight shines over the same location in the sky at night; a spotlight light only. As the light of the spotlight gets lower and lower, then more and more of the sky is left in darkness. Shadow is the opposite of light. Light is a property of something. Shine means to direct energy.', 0.2456140350877193) ('Nights are the opposite of days. The word diurnal means things that occur during the night (e.g., sleep, danger, and darkness). The rotation of the Earth around the sun causes all the Earth to change from night to day (daytime) and then back to night (nighttime). In the Northern', 0.15094339622641506)
"""

#TODO F1 between question and each pred seems to correlate with usefulness
#TODO WORLDTREE Facts don't seem to be terribly useful. Try hand-authoring explanations that are better
#TODO SQA explanations from facts

