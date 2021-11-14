#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:52:23 2021

@author: tim hartill

generate arbitrary predictions from a model


"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining
from bart import MyBart
from data import normalize_num, split_digits_special
from utils import load_model, run_model, get_single_result

model_name = "facebook/bart-large"
#'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
#checkpoint = None
#checkpoint = "/data/thar011/ckpts/unifiedqa-bart-large-allenai/unifiedQA-uncased/best-model.pt" # path to the downloaded checkpoint
checkpoint = "/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-150000.pt"
#checkpoint = "/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/best-model-200000.pt"
special='Ġ'
indiv_digits = True  # individual digit tokenization
norm_numbers = True # normalize numbers. if norm='10e' normalize to 10e format
norm=''



tokenizer, model = load_model(model_name, checkpoint)

input_string = 'In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.'

input_string = "<s> What is the population of Albany, Georgia? \\n"  # <no answer>
input_string = "<s> What is the population of Albany, Georgia? \\n Albany, GA has around 75,000 people"  # 75000! ie [' 75000\n']
# correctly returns [' 77434\n']
input_string = "<s> What is the population of Albany, Georgia? \\n Albany (/\u0254\u02d0b\u0259ni/ AW-b\u0259-nee) is a city in the U.S. state of Georgia. Located on the Flint River, it is the seat of Dougherty County, and is the sole incorporated city in that county. Located in southwest Georgia, it is the principal city of the Albany, Georgia metropolitan area. The population was 77,434 at the 2010 U.S. Census, making it the eighth-largest city in the state. It became prominent in the nineteenth century as a shipping and market center, first served by riverboats. Scheduled steamboats connected Albany with the busy port of Apalachicola, Florida. They were replaced by railroads. Seven lines met in Albany, and it was a center of trade in the Southeast. It is part of the Black Belt, the extensive area in the Deep South of cotton plantations. From the mid-20th century, it received military investment during World War II and after, that helped develop the region. Albany and this area were prominent during the civil rights era, particularly during the early 1960s as activists worked to regain voting and other civil rights. Railroad restructuring and reduction in the military here caused job losses, but the city has developed new businesses."  
# 2 paras both with same direct answer: still correct
input_string = "<s> What is the population of Albany, Georgia? \\n Albany (/\u0254\u02d0b\u0259ni/ AW-b\u0259-nee) is a city in the U.S. state of Georgia. Located on the Flint River, it is the seat of Dougherty County, and is the sole incorporated city in that county. Located in southwest Georgia, it is the principal city of the Albany, Georgia metropolitan area. The population was 77,434 at the 2010 U.S. Census, making it the eighth-largest city in the state. It became prominent in the nineteenth century as a shipping and market center, first served by riverboats. Scheduled steamboats connected Albany with the busy port of Apalachicola, Florida. They were replaced by railroads. Seven lines met in Albany, and it was a center of trade in the Southeast. It is part of the Black Belt, the extensive area in the Deep South of cotton plantations. From the mid-20th century, it received military investment during World War II and after, that helped develop the region. Albany and this area were prominent during the civil rights era, particularly during the early 1960s as activists worked to regain voting and other civil rights. Railroad restructuring and reduction in the military here caused job losses, but the city has developed new businesses. As of the census of 2010, there were 77,434 people, 29,781 households, and 18,515 families residing in the city. The population density was 1,385.5 people per square mile (535.0/km\u00b2). There were 33,436 housing units at an average density of 577.3 per square mile (222.9/km\u00b2)."  

# 3 paras, 2 with same direct answer: still returns correct answer
input_string = "<s> What is the population of Albany, Georgia? \\n As the world ocean is the principal component of Earth's hydrosphere, it is integral to life, forms part of the carbon cycle, and influences climate and weather patterns. The World Ocean is the habitat of 230,000 known species, but because much of it is unexplored, the number of species that exist in the ocean is much larger, possibly over two million. The origin of Earth's oceans is unknown; oceans are thought to have formed in the Hadean eon and may have been the cause for the emergence of life. Albany (/\u0254\u02d0b\u0259ni/ AW-b\u0259-nee) is a city in the U.S. state of Georgia. Located on the Flint River, it is the seat of Dougherty County, and is the sole incorporated city in that county. Located in southwest Georgia, it is the principal city of the Albany, Georgia metropolitan area. The population was 77,434 at the 2010 U.S. Census, making it the eighth-largest city in the state. It became prominent in the nineteenth century as a shipping and market center, first served by riverboats. Scheduled steamboats connected Albany with the busy port of Apalachicola, Florida. They were replaced by railroads. Seven lines met in Albany, and it was a center of trade in the Southeast. It is part of the Black Belt, the extensive area in the Deep South of cotton plantations. From the mid-20th century, it received military investment during World War II and after, that helped develop the region. Albany and this area were prominent during the civil rights era, particularly during the early 1960s as activists worked to regain voting and other civil rights. Railroad restructuring and reduction in the military here caused job losses, but the city has developed new businesses. As of the census of 2010, there were 77,434 people, 29,781 households, and 18,515 families residing in the city. The population density was 1,385.5 people per square mile (535.0/km\u00b2). There were 33,436 housing units at an average density of 577.3 per square mile (222.9/km\u00b2)."  

# 4 paras, 2 with same direct answer but one probably truncated: still returns correct answer
input_string = "<s> What is the population of Albany, Georgia? \\n Maine, Vermont, New Hampshire, and Interior northern Massachusetts have a humid continental climate (Dfb in K\u00f6ppen climate classification). In this region, the winters are long, cold, and heavy snow is common (most locations receive 60 to 120 inches (1,500 to 3,000\u00a0mm) of snow annually in this region). The summer months are moderately warm, though summer is rather short. Annual rainfall is spread evenly throughout the year. Cities like Bangor, Maine, Portland, Maine, Manchester, New Hampshire, Burlington, Vermont, and Pittsfield, Massachusetts average around 45 inches (1,100\u00a0mm) of rainfall and 60 to 90 inches (1,500 to 2,300\u00a0mm) of snow annually. As the world ocean is the principal component of Earth's hydrosphere, it is integral to life, forms part of the carbon cycle, and influences climate and weather patterns. The World Ocean is the habitat of 230,000 known species, but because much of it is unexplored, the number of species that exist in the ocean is much larger, possibly over two million. The origin of Earth's oceans is unknown; oceans are thought to have formed in the Hadean eon and may have been the cause for the emergence of life. Albany (/\u0254\u02d0b\u0259ni/ AW-b\u0259-nee) is a city in the U.S. state of Georgia. Located on the Flint River, it is the seat of Dougherty County, and is the sole incorporated city in that county. Located in southwest Georgia, it is the principal city of the Albany, Georgia metropolitan area. The population was 77,434 at the 2010 U.S. Census, making it the eighth-largest city in the state. It became prominent in the nineteenth century as a shipping and market center, first served by riverboats. Scheduled steamboats connected Albany with the busy port of Apalachicola, Florida. They were replaced by railroads. Seven lines met in Albany, and it was a center of trade in the Southeast. It is part of the Black Belt, the extensive area in the Deep South of cotton plantations. From the mid-20th century, it received military investment during World War II and after, that helped develop the region. Albany and this area were prominent during the civil rights era, particularly during the early 1960s as activists worked to regain voting and other civil rights. Railroad restructuring and reduction in the military here caused job losses, but the city has developed new businesses. As of the census of 2010, there were 77,434 people, 29,781 households, and 18,515 families residing in the city. The population density was 1,385.5 people per square mile (535.0/km\u00b2). There were 33,436 housing units at an average density of 577.3 per square mile (222.9/km\u00b2)."  

# returns [' 3\n'] might be right!
input_string = "<s> How many kids did Julius Caesar have? \\n" 

# No single para with answer, 2 paras, requires approx counting: returns [' one\n'] correct answer is 2 or 3 sqa model returned 2
input_string = "<s> How many kids did Julius Caesar have? \\n Caesarion was the eldest son of Cleopatra and possibly the only biological son of Julius Caesar, after whom he was named. He was the last sovereign member of the Ptolemaic dynasty of Egypt. In addition to being co-ruler of Egypt as Pharaoh with his mother, he was expected to be his father's successor as the leader of the Romans. Julia (c. 76 BC \u2013 54 BC) was the daughter of Roman dictator Julius Caesar by his first or second wife Cornelia, and his only child from his marriages. Julia became the fourth wife of Pompey the Great and was renowned for her beauty and virtue."

# No single para with answer, 3 paras, requires approx counting: returns [' 2\n'] correct answer is 2 or 3 sqa model returned 2
input_string = "<s> How many kids did Julius Caesar have? \\n Caesarion was the eldest son of Cleopatra and possibly the only biological son of Julius Caesar, after whom he was named. He was the last sovereign member of the Ptolemaic dynasty of Egypt. In addition to being co-ruler of Egypt as Pharaoh with his mother, he was expected to be his father's successor as the leader of the Romans. Julia (c. 76 BC \u2013 54 BC) was the daughter of Roman dictator Julius Caesar by his first or second wife Cornelia, and his only child from his marriages. Julia became the fourth wife of Pompey the Great and was renowned for her beauty and virtue. Parents Father Gaius Julius Caesar the Elder (proconsul of Asia in 90s BC) Mother Aurelia (one of the Aurelii Cottae) Sisters Julia Major Julia Minor Wives First marriage to Cornelia (Cinnilla), from 84\u00a0BC until her death in 69 or 68\u00a0BC Second marriage to Pompeia, from 67\u00a0BC until he divorced her around 61\u00a0BC over the Bona Dea scandal Third marriage to Calpurnia, from 59\u00a0BC until Caesar's death Children Julia, by Cornelia, born in 83 or 82 BC Caesarion, by Cleopatra VII, born 47 BC, and killed at age 17 by Caesar's adopted son Octavianus. Posthumously adopted: Gaius Julius Caesar Octavianus, his great-nephew by blood (grandson of Julia, his sister), who later became Emperor Augustus. Suspected Children Marcus Junius Brutus (born 85 BC): The historian Plutarch notes that Caesar believed Brutus to have been his illegitimate son, as his mother Servilia had been Caesar's lover during their youth. Caesar would have been 15 years old when Brutus was born. Junia Tertia (born ca. 60s BC), the daughter of Caesar's lover Servilia was believed by Cicero among other contemporaries, to be Caesar's natural daughter. Decimus Junius Brutus Albinus (born ca. 85\u201381 BC): On several occasions Caesar expressed how he loved Decimus Brutus like a son. This Brutus was also named an heir of Caesar in case Octavius had died before the latter. Ronald Syme argued that if a Brutus was the natural son of Caesar, Decimus was more likely than Marcus. Grandchildren Grandchild from Julia and Pompey, dead at several days, unnamed. Lovers Cleopatra VII, mother of Caesarion Servilia, mother of Brutus Euno\u00eb, queen of Mauretania and wife of Bogudes Notable relatives Gaius Marius (married to his paternal aunt Julia) Mark Antony (his relative through Antony's mother Julia) Lucius Julius Caesar (his third-cousin) Roman society viewed the passive role during sexual activity, regardless of gender, to be a sign of submission or inferiority. Indeed, Suetonius says that in Caesar's Gallic triumph, his soldiers sang that, \"Caesar may have conquered the Gauls, but Nicomedes conquered Caesar.\" According to Cicero, Bibulus, Gaius Memmius, and others (mainly Caesar's enemies), he had an affair with Nicomedes IV of Bithynia early in his career. The stories were repeated, referring to Caesar as the Queen of Bithynia, by some Roman politicians as a way to humiliate him. Caesar himself denied the accusations repeatedly throughout his lifetime, and according to Cassius Dio, even under oath on one occasion. This form of slander was popular during this time in the Roman Republic to demean and discredit political opponents."

# Single salient fact plus 3 paras, requires approx counting: returns [' 3\n'] correct answer is 2 or 3 sqa model returned 2
input_string = "<s> How many kids did Julius Caesar have? \\n Julius Caesar had three children. Caesarion was the eldest son of Cleopatra and possibly the only biological son of Julius Caesar, after whom he was named. He was the last sovereign member of the Ptolemaic dynasty of Egypt. In addition to being co-ruler of Egypt as Pharaoh with his mother, he was expected to be his father's successor as the leader of the Romans. Julia (c. 76 BC \u2013 54 BC) was the daughter of Roman dictator Julius Caesar by his first or second wife Cornelia, and his only child from his marriages. Julia became the fourth wife of Pompey the Great and was renowned for her beauty and virtue. Parents Father Gaius Julius Caesar the Elder (proconsul of Asia in 90s BC) Mother Aurelia (one of the Aurelii Cottae) Sisters Julia Major Julia Minor Wives First marriage to Cornelia (Cinnilla), from 84\u00a0BC until her death in 69 or 68\u00a0BC Second marriage to Pompeia, from 67\u00a0BC until he divorced her around 61\u00a0BC over the Bona Dea scandal Third marriage to Calpurnia, from 59\u00a0BC until Caesar's death Children Julia, by Cornelia, born in 83 or 82 BC Caesarion, by Cleopatra VII, born 47 BC, and killed at age 17 by Caesar's adopted son Octavianus. Posthumously adopted: Gaius Julius Caesar Octavianus, his great-nephew by blood (grandson of Julia, his sister), who later became Emperor Augustus. Suspected Children Marcus Junius Brutus (born 85 BC): The historian Plutarch notes that Caesar believed Brutus to have been his illegitimate son, as his mother Servilia had been Caesar's lover during their youth. Caesar would have been 15 years old when Brutus was born. Junia Tertia (born ca. 60s BC), the daughter of Caesar's lover Servilia was believed by Cicero among other contemporaries, to be Caesar's natural daughter. Decimus Junius Brutus Albinus (born ca. 85\u201381 BC): On several occasions Caesar expressed how he loved Decimus Brutus like a son. This Brutus was also named an heir of Caesar in case Octavius had died before the latter. Ronald Syme argued that if a Brutus was the natural son of Caesar, Decimus was more likely than Marcus. Grandchildren Grandchild from Julia and Pompey, dead at several days, unnamed. Lovers Cleopatra VII, mother of Caesarion Servilia, mother of Brutus Euno\u00eb, queen of Mauretania and wife of Bogudes Notable relatives Gaius Marius (married to his paternal aunt Julia) Mark Antony (his relative through Antony's mother Julia) Lucius Julius Caesar (his third-cousin) Roman society viewed the passive role during sexual activity, regardless of gender, to be a sign of submission or inferiority. Indeed, Suetonius says that in Caesar's Gallic triumph, his soldiers sang that, \"Caesar may have conquered the Gauls, but Nicomedes conquered Caesar.\" According to Cicero, Bibulus, Gaius Memmius, and others (mainly Caesar's enemies), he had an affair with Nicomedes IV of Bithynia early in his career. The stories were repeated, referring to Caesar as the Queen of Bithynia, by some Roman politicians as a way to humiliate him. Caesar himself denied the accusations repeatedly throughout his lifetime, and according to Cassius Dio, even under oath on one occasion. This form of slander was popular during this time in the Roman Republic to demean and discredit political opponents."


# returns [' <no answer>\n']
input_string = "<s> How many kids did Genghis Khan have? \\n"

# 1 para partial answer returns [' <no answer>\n']
input_string = "<s> How many kids did Genghis Khan have? \\n Alakhai Bekhi (Alagai B\u00e4ki; born c. 1191, died after 1230) was a daughter of Genghis Khan and his first wife B\u00f6rte. She played significant role behind the scenes during her father\u2019s lifetime."

# 2 para partial answer returns [' four\n'] sqa model returns "fourth"
input_string = "<s> How many kids did Genghis Khan have? \\n Alakhai Bekhi (Alagai B\u00e4ki; born c. 1191, died after 1230) was a daughter of Genghis Khan and his first wife B\u00f6rte. She played significant role behind the scenes during her father\u2019s lifetime. Tolui (c.1191\u20131232) was the fourth son of Genghis Khan by his chief khatun B\u00f6rte. His ulus, or territorial inheritance, at his father's death in 1227 was the homelands in Mongolia, and it was he who served as civil administrator until 1229, the time it took to confirm \u00d6gedei as second Great Khan of the Mongol Empire (1206\u20131368). Before that he had served with distinction in the campaigns against the Jin dynasty, the Western Xia and the Khwarezmid Empire, where he was instrumental in the capture and massacre at Merv and Nishapur. He is a direct ancestor of most of the Ilkhanids."

# 3 para partial answer truncated returns [' four\n'] sqa model returns "fourth"
input_string = "<s> How many kids did Genghis Khan have? \\n Alakhai Bekhi (Alagai B\u00e4ki; born c. 1191, died after 1230) was a daughter of Genghis Khan and his first wife B\u00f6rte. She played significant role behind the scenes during her father\u2019s lifetime. Tolui (c.1191\u20131232) was the fourth son of Genghis Khan by his chief khatun B\u00f6rte. His ulus, or territorial inheritance, at his father's death in 1227 was the homelands in Mongolia, and it was he who served as civil administrator until 1229, the time it took to confirm \u00d6gedei as second Great Khan of the Mongol Empire (1206\u20131368). Before that he had served with distinction in the campaigns against the Jin dynasty, the Western Xia and the Khwarezmid Empire, where he was instrumental in the capture and massacre at Merv and Nishapur. He is a direct ancestor of most of the Ilkhanids. Soon after the marriage between them took place, the Three Merkits attacked their family camp at dawn and kidnapped B\u00f6rte. She was given to one of their warriors as a spoil of war. Tem\u00fcjin was deeply distressed by the abduction of his wife and remarked that his \"bed was made empty\" and his \"breast was torn apart\". Tem\u00fcjin rescued her several months later with the aid of his allies Wang Khan and Jamukha. Many scholars describe this event as one of the key crossroads in Tem\u00fcjin\u2019s life, which moved him along the path towards becoming a conqueror. \u201cAs the pillaging and plundering went on, Tem\u00fcjin moved among the people that were hurriedly escaping, calling, \u2018B\u00f6rte, B\u00f6rte!\u2019 And so he came upon her, for Lady B\u00f6rte was among those fleeing people. She heard the voice of Tem\u00fcjin and, recognizing it, she got off the cart and came running towards him. Although it was still night, Lady B\u00f6rte and Qo\u2019aq\u010din both recognized Tem\u00fcjin\u2019s reins and tether and grabbed them. It was moonlight; he looked at them, recognized Lady B\u00f6rte, and they fell into each other\u2019s arms.\u201d -The Secret History of the Mongols B\u00f6rte was held captive for eight months, and gave birth to Jochi soon after she was rescued. This left doubt as to who the father of the child was, because her captor took her as a \"wife\" and could have possibly impregnated her. Despite this, Tem\u00fcjin let Jochi remain in the family and claimed him as his own son. B\u00f6rte had three more sons, Chagatai (1183\u20131242), \u00d6gedei (1186\u20131241), and Tolui (1191\u20131232). Tem\u00fcjin had many other children with other wives, but they were excluded from succession, only B\u00f6rte's sons could be considered to be his heirs."

#correctly returns [' yes\n']. sqa model final. sqa model Incorrectly returns no
input_string = "<s> Is fourth greater than two? \\n"

#uqa+Tdnd results returns [' 3\n'] with norm numbers and [' four\n'] without. sqa model final. sqa model Incorrectly returns no
input_string = "<s> Is four greater than 2? \\n"

#uqa+Tdnd results returns [' yes\n'] with norm numbers and [' yes\n'] without. sqa model final. sqa model Incorrectly returns no
input_string = "<s> yes or no. Is four greater than 2? \\n"

#uqa+Tdnd results incorrectly returns [' yes\n'] with norm numbers and [' yes\n'] without again... 
input_string = "<s> yes or no. Is 2 greater than four? \\n"


input_string = "<s> what is the difference between 100,000 and 97,856? \\n"  # 1244 should be 2144
input_string = "<s> what is 100,000 - 97,856? \\n"   #2344

input_string = "<s> what is the difference between 100,000 and 77,434? \\n"  # 2266 should be 22566
input_string = "<s> what is 100,000 - 77,434? \\n"   #Also 2266

input_string = "<s> is 2244 greater than 1166? \\n (A) yes (b) no"  # 1078 ie 2244-1166 with " (A) yes (b) no" answers yes or no but often incorrectly
input_string = "<s> is 2244 greater than 1166? \\n"  # 1078 ie 2244-1166 with " (A) yes (b) no" answers yes or no but often incorrectly

input_string = "<s> is 16 greater than 3? \\n (A) yes (b) no"  # yes
input_string = "<s> is 3 greater than 16? \\n (A) yes (b) no"  # also yes instead of no
input_string = "<s> 3 is greater than 16?  \\n"  # [' 2\n']
input_string = "<s> is 3 more than 16? \\n"  # [' 2\n']
input_string = "<s> is 3 higher than 16? \\n"  # [' 2\n']
input_string = "<s> is 3 less than 16? \\n"  # [' 13\n']
input_string = '<s> is 3 > 16?  \\n'  # [' 6\n']
input_string = "<s> is 3 greater than or less than 16?  \\n"  # [' 16\n']
input_string = "<s> is 3 greater than 16 or 16 greater than 3?  \\n"  # [' 14\n']
input_string = "<s> is it true that 3 greater than 16? \\n"  # also yes instead of no
input_string = "<s> is it the case that is 3 greater than 16? \\n"  # also yes instead of no
input_string = "<s> is it not the case that is 3 greater than 16? \\n"  # no
input_string = "<s> is it not the case that is 16 greater than 3? \\n"  # no
input_string = "<s> is 3 greater than 16 or not? \\n"  # also yes instead of no
input_string = "<s> is sixteen greater than three? \\n"  # yes if norm numbers False
input_string = "<s> is three greater than sixteen? \\n"  # yes if norm numbers False
input_string = "<s> is 3 greater than 16? \\n answer yes or no"  # [' 2\n']

input_string = "<s> What is 1457.0 + 18750 + 17728.24 + 14009.6? \\n"  #[' 51834.94\n'] correct = 51944.84

run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers)
run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=False)  #
run_model(input_string, model, tokenizer, indiv_digits=False, norm_numbers=norm_numbers)  #just makes the calculation worse

# match current inference params:
res = run_model(input_string, model, tokenizer, indiv_digits=indiv_digits, norm_numbers=norm_numbers,
                num_return_sequences=1, num_beams=4, early_stopping=True, min_length=1, max_length=100,
                output_scores=True, return_dict_in_generate=True)  # res.keys(): odict_keys(['sequences', 'sequences_scores', 'scores', 'preds'])
pred, score = get_single_result(res)

res = run_model(input_string, model, tokenizer, temperature=0.9, num_return_sequences=4, num_beams=20, 
                output_scores=True, return_dict_in_generate=True)
#res.preds has ["pred1", "pred2"] if input_string is a list
#res.sequences has output tokens as ids
#res.sequences_scores returns overall score (final beam score) of each returned seq [num_sequences]
#res.scores returns tuple of scores - detailed score for each sequence: (output num_toks entries of [num_beams, vocab_size])

run_model("<s> is plastic harder than metal? \\n", model, tokenizer)

res = run_model([input_string,input_string], model, tokenizer, indiv_digits=False, norm_numbers=False, 
                lower=False, num_return_sequences=1, num_beams=4, early_stopping=True, max_length=150,
                output_scores=True, return_dict_in_generate=True)
#res.preds has ["pred1 sentence", "pred2 sentence"] if input_string is a list of #samples. num_return_sequences outputs per sample
#res.sequences has output tokens as ids shape [#samples, max output len]
#res.sequences_scores returns overall score (final beam score) of each returned seq [num_sequences]
#res.scores is tuple of output num_toks entries of [#beams*#samples, vocab size] if input_string is a list of #samples

# GPT2 tests: added append_eos=False, prepend_bos=False,
res = run_model([input_string,input_string], model, tokenizer, indiv_digits=False, norm_numbers=False, 
                append_eos=False, prepend_bos=False,
                lower=False, temperature=0.9, num_return_sequences=1, num_beams=4,  max_length=150, no_repeat_ngram_size=2,
                output_scores=True, return_dict_in_generate=True)
#res.sequences_scores returns overall score (final beam score) of each returned seq [num_sequences]

res = run_model(input_string, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                lower=False, append_eos=False, prepend_bos=False,
                max_length=150, do_sample=True, top_k=50, #top_p=0.92, 
                output_scores=True, return_dict_in_generate=True)
res.preds
