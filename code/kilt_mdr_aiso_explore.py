#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 11:38:21 2022

@author: tim hartill

Explore KILT processed wiki dump without MongoDB
Also Explore MDR HPQA data
And AISO data

1. git clone kilt and cd into kilt dir
2. mkdir .../kilt/data
3. pip install -r requirements.txt then pip install . otherwise get_triviaqa_input.py download script doesnt work
4. cd into .../kilt/data then wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
5. cd .../kilt then python scripts/download_all_kilt_data.py
6.                  python scripts/get_triviaqa_input.py

"""
import os
import pandas as pd
import utils

#KILT:
KILT_DATA_DIR = '/data/thar011/gitrepos/KILT/data/'
KILT_WIKI = 'kilt_knowledgesource.json'
KILT_HPQA_DEV = 'hotpotqa-dev-kilt.jsonl'
KILT_FEVER_DEV = 'fever-dev-kilt.jsonl'
KILT_NQ_DEV = 'nq-dev-kilt.jsonl'

#MDR:
HPQA_DATA_DIR = '/data/thar011/gitrepos/multihop_dense_retrieval/data/hotpot/'
HPQA_DEV = 'hotpot_dev_with_neg_v0.json'  # retrieval training dev set

#AISO:
AISO_DATA_DIR = '/data/thar011/gitrepos/AISO/data/'
AISO_STEP_TRAIN = 'hotpot-step-train.strict.refined.jsonl'
AISO_STEP_DEV = 'hotpot-step-dev.strict.refined.jsonl'
AISO_CORPUS = 'corpus/hotpot-paragraph-5.strict.refined.tsv'


wiki = utils.load_jsonl(os.path.join(KILT_DATA_DIR, KILT_WIKI))  # 5903531 rows -> 5903530 final

print(wiki[0].keys()) # dict_keys(['_id', 'wikipedia_id', 'wikipedia_title', 'text', 'anchors', 'categories', 'history', 'wikidata_info'])
wiki[0]['_id']  # '290'
wiki[0]['wikipedia_id']  # '290'
wiki[0]['wikipedia_title']  # 'A'
wiki[0]['text'] # hyperlinks not in text..
# ['A\n',
# 'A (named , plural "As", "A\'s", "a"s, "a\'s" or "aes") is the first letter and the first vowel of the modern English alphabet and the ISO basic Latin alphabet. It is similar to the Ancient Greek letter alpha, from which it derives. The uppercase version consists of the two slanting sides of a triangle, crossed in the middle by a horizontal bar. The lowercase version can be written in two forms: the double-storey a and single-storey ɑ. The latter is commonly used in handwriting and fonts based on it, especially fonts intended to be read by children, and is also found in italic type.\n',
# 'In the English grammar, "a", and its variant "an", is an indefinite article.\n',
# 'Section::::History.\n', ... , 'Section::::History.:Typographic variants.\n', ...
# 'Section::::Use in writing systems.\n',  'Section::::Use in writing systems.:English.\n', 'In modern English orthography, the letter represents at least seven different vowel sounds:\n',
# 'BULLET::::- the near-open front unrounded vowel as in "pad";\n',
# 'BULLET::::- the open back unrounded vowel as in "father", which is closer to its original Latin and Greek sound;\n',
wiki[0]['anchors'][0] # {'text': 'named', #para text
#                         'href': 'English%20alphabet%23Letter%20names', #hyperlink href target https://en.wikipedia.org/wiki/English_alphabet#Letter_names
#                         'paragraph_id': 1, #['text'][1]
#                         'start': 3, #['text'][1][3] ...
#                         'end': 8, #['text'][1][8]
#                         'wikipedia_title': 'English alphabet', #wikipedia page title target
#                         'wikipedia_id': '378194'} #wikipedia target id
wiki[0]['categories'] # 'ISO basic Latin letters'
# 'url' is the version of the page the data in this record is from:
wiki[0]['history'] # {'revid': 907008348, 'timestamp': '2019-07-19T20:25:53Z', 'parentid': 906725792, 'pre_dump': True, 'pageid': 290, 'url': 'https://en.wikipedia.org/w/index.php?title=A&oldid=907008348'}
wiki[0]['wikidata_info'] # {'wikidata_id': 'Q9659'}

hpqa = utils.load_jsonl(os.path.join(KILT_DATA_DIR, KILT_HPQA_DEV)) #5600
hpqa[0]
#{'id': '5a8b57f25542995d1e6f1371',
# 'input': 'Were Scott Derrickson and Ed Wood of the same nationality?',
# 'output': [{'answer': 'yes',
#   'provenance': [{'wikipedia_id': '2816539',
#     'title': 'Scott Derrickson',
#     'start_paragraph_id': 1,
#     'start_character': 0,
#     'end_paragraph_id': 1,
#     'end_character': 81,
#     'bleu_score': 0.6964705665515707,
#     'section': 'Section::::Abstract.'},
#    {'wikipedia_id': '10520',
#     'title': 'Ed Wood',
#     'start_paragraph_id': 1,
#     'start_character': 0,
#     'end_paragraph_id': 1,
#     'end_character': 106,
#     'bleu_score': 0.7784290264326612,
#     'section': 'Section::::Abstract.'}]}]}

fever = utils.load_jsonl(os.path.join(KILT_DATA_DIR, KILT_FEVER_DEV))
fever[0]
#{'id': '137334',
# 'input': 'Fox 2000 Pictures released the film Soul Food.',
# 'output': [{'answer': 'SUPPORTS',
#   'provenance': [{'wikipedia_id': '1073955',
#     'title': 'Soul Food (film)',
#     'start_paragraph_id': 1,
#     'start_character': 0,
#     'end_paragraph_id': 1,
#     'end_character': 153,
#     'bleu_score': 0.8482942955247808,
#     'meta': {'fever_page_id': 'Soul_Food_-LRB-film-RRB-',
#      'fever_sentence_id': 0},
#     'section': 'Section::::Abstract.'}]}]}

nq = utils.load_jsonl(os.path.join(KILT_DATA_DIR, KILT_NQ_DEV))
nq[0]
"""
{'id': '6915606477668963399',
 'input': 'what do the 3 dots mean in math',
 'output': [{'answer': 'the therefore sign',
   'provenance': [{'wikipedia_id': '10593264',
     'title': 'Therefore sign',
     'start_paragraph_id': 1,
     'start_character': 44,
     'end_paragraph_id': 1,
     'end_character': 62,
     'bleu_score': 1.0,
     'section': 'Section::::Abstract.'}]},
  {'answer': 'therefore sign',
   'provenance': [{'wikipedia_id': '10593264',
     'title': 'Therefore sign',
     'start_paragraph_id': 1,
     'start_character': 48,
     'end_paragraph_id': 1,
     'end_character': 62,
     'bleu_score': 1.0,
     'section': 'Section::::Abstract.'}]},
  {'answer': 'a logical consequence , such as the conclusion of a syllogism'},
  {'answer': 'the therefore sign ( ∴ ) is generally used before a logical consequence , such as the conclusion of a syllogism'},
  {'provenance': [{'wikipedia_id': '10593264',
     'title': 'Therefore sign',
     'start_paragraph_id': 1,
     'start_character': 0,
     'end_paragraph_id': 1,
     'end_character': 375,
     'bleu_score': 0.6816476074249925,
     'meta': {'yes_no_answer': 'NONE', 'annotation_id': 13591449469826568799},
     'section': 'Section::::Abstract.'}]},
  {'provenance': [{'wikipedia_id': '10593264',
     'title': 'Therefore sign',
     'section': 'Section::::Abstract.',
     'start_paragraph_id': 1,
     'end_paragraph_id': 1,
     'meta': {'evidence_span': ['The symbol consists of three dots placed in an upright triangle and is read "therefore".',
       'The symbol consists of three dots placed in an upright triangle and is read "therefore".',
       'The symbol consists of three dots placed in an upright triangle and is read "therefore".',
       'In logical argument and mathematical proof, the therefore sign () is generally used before a logical consequence, such as the conclusion of a syllogism. The symbol consists of three dots placed in an upright triangle and is read "therefore". It is encoded at .']}}]}]}
"""

hpqa = utils.load_jsonl(os.path.join(KILT_DATA_DIR, KILT_HPQA_DEV)) #5600
hpqa[0]
#{'id': '5a8b57f25542995d1e6f1371',
# 'input': 'Were Scott Derrickson and Ed Wood of the same nationality?',
# 'output': [{'answer': 'yes',
#   'provenance': [{'wikipedia_id': '2816539',
#     'title': 'Scott Derrickson',
#     'start_paragraph_id': 1,
#     'start_character': 0,
#     'end_paragraph_id': 1,
#     'end_character': 81,
#     'bleu_score': 0.6964705665515707,
#     'section': 'Section::::Abstract.'},
#    {'wikipedia_id': '10520',
#     'title': 'Ed Wood',
#     'start_paragraph_id': 1,
#     'start_character': 0,
#     'end_paragraph_id': 1,
#     'end_character': 106,
#     'bleu_score': 0.7784290264326612,
#     'section': 'Section::::Abstract.'}]}]}


########### MDR ####################

mdr_hpqa = utils.load_jsonl(os.path.join(HPQA_DATA_DIR, HPQA_DEV) ) # 7405
mdr_hpqa[0].keys() # dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id'])  comparison type
mdr_hpqa[0]['question'] # 'Were Scott Derrickson and Ed Wood of the same nationality?'
mdr_hpqa[0]['answers'] # ['yes']
mdr_hpqa[0]['type'] # 'comparison'
mdr_hpqa[0]['_id'] # '5a8b57f25542995d1e6f1371' # matches KILT 'id'
mdr_hpqa[0]['pos_paras'] # hyperlinks, wiki ids etc removed. NOT CHUNKED LIKE DPR
#[{'title': 'Ed Wood',
#  'text': 'Edward Davis Wood Jr. (October 10, 1924\xa0– December 10, 1978) was an American filmmaker, actor, writer, producer, and director.'},
# {'title': 'Scott Derrickson',
#  'text': 'Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."'}]
mdr_hpqa[0]['neg_paras'] # same format as pos long list of irrelevant paras
len(mdr_hpqa[0]['neg_paras'])  # 19
mdr_hpqa[0]['neg_paras'][0]
#{'title': 'That Darn Cat (1997 film)',
# 'text': 'That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug. It is a remake of the 1965 film "That Darn Cat!", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon. It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.'}
# NO 'bridge' KEY - ORDERING in mhop_dataset.py is random


mdr_hpqa[1].keys() # dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', 'bridge', '_id']) bridge type
mdr_hpqa[1]['question'] # 'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?'
mdr_hpqa[1]['answers'] # ['Chief of Protocol']
mdr_hpqa[1]['type'] # 'bridge'
mdr_hpqa[1]['_id'] # '5a8c7595554299585d9e36b6'
mdr_hpqa[1]['pos_paras']
#[{'title': 'Kiss and Tell (1945 film)',
#  'text': "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves."},
# {'title': 'Shirley Temple',
#  'text': "Shirley Temple Black (April 23, 1928\xa0– February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."}]
len(mdr_hpqa[1]['neg_paras'])  # 19
mdr_hpqa[1]['neg_paras'][0]
#{'title': 'A Kiss for Corliss',
# 'text': 'A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film "Kiss and Tell". "A Kiss for Corliss" was retitled "Almost a Bride" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists.'}
mdr_hpqa[1]['bridge'] # 'Shirley Temple'  TITLE Of SECOND PARA FOR BRIDGE QUESTIONS FOR ORDERING IN mhop_dataset.py


#### AISO #######################

aiso_corpus = pd.read_csv(os.path.join(AISO_DATA_DIR, AISO_CORPUS), sep='\t')   # (249461, 4). Header corrupted and should be 5M+ rows...
# 0   id               249460 non-null  object
# 1   text             249460 non-null  object
# 2   itle\hyperlinks  249458 non-null  object
# 3   sentence_spans   249458 non-null  object

aiso_dev = utils.load_jsonl(os.path.join(AISO_DATA_DIR, AISO_STEP_DEV))     #7405
aiso_train = utils.load_jsonl(os.path.join(AISO_DATA_DIR, AISO_STEP_TRAIN)) #90447

aiso_dev[0].keys()   # dict_keys(['_id', 'question', 'answer', 'sp_facts', 'hard_negs', 'hn_scores', 'state2action'])
aiso_train[0].keys() # dict_keys(['_id', 'question', 'answer', 'sp_facts', 'hard_negs', 'hn_scores', 'state2action'])

aiso_dev[0]['_id']      # '5a8b57f25542995d1e6f1371'  #matches MDR/KILT/HPQA
aiso_dev[0]['question'] # 'Were Scott Derrickson and Ed Wood of the same nationality?'
aiso_dev[0]['answer']   # 'yes'
aiso_dev[0]['sp_facts'] # {'Scott Derrickson': [0], 'Ed Wood': [0]}
len(aiso_dev[0]['hard_negs'])  # 10
aiso_dev[0]['hard_negs'][0] # '528464_0'
aiso_dev[0]['hn_scores']  # [0.0018630551639944315, 0.0005404593539424241, 0.000537772080861032, 0.0005160804139450192, 0.0005122957518324256, 0.0005065873847343028, 0.0005059774266555905, 0.0004960809019394219, 0.0004910272546112537, 0.0004873361031059176]
aiso_dev[0]['state2action'] # {'initial': {   'query': 'Scott Derrickson', 'action': 'MDR', # take action MDR(q+'Scott Derrickson')
#                                          'sp_ranks': {'BM25': {'2816539_0': 0, '10520_0': 2000},
#                                                  'BM25+Link': {'2816539_0': 1, '10520_0': 2000},
#                                                        'MDR': {'2816539_0': 1, '10520_0': 0},
#                                                   'MDR+Link': {'2816539_0': 2000, '10520_0': 34}}},
#                     'Scott Derrickson': {   'query': 'Ed Wood', 'action': 'BM25', #take action BM25(q + 'Scott Derrickson'+'Ed Wood')
#                                         'sp2_ranks': {'BM25': 0, 'BM25+Link': 1, 'MDR': 0, 'MDR+Link': 53}},
#                              'Ed Wood': {   'query': 'Ed Wood', 'action': 'MDR',  #actually at the answer, "action MDR(ed wood)' seems to be a leftover - or is it the final action of 3 ie get to final ed wood through intermediate para 10520
#                                         'sp2_ranks': {'BM25': 2000, 'BM25+Link': 2000, 'MDR': 0, 'MDR+Link': 2000}}}

aiso_dev[1]['_id']      # '5a8c7595554299585d9e36b6'  #matches MDR/KILT/HPQA
aiso_dev[1]['question'] # 'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?'
aiso_dev[1]['answer']   # 'Chief of Protocol'
aiso_dev[1]['sp_facts'] # {'Kiss and Tell (1945 film)': [0], 'Shirley Temple': [0, 1]}  # sentence # in para relevance ie sent 0 in 'Kiss and Tell (1945 film)' is relevant and sents 0 & 1 in 'Shirley Temple' are relevant
len(aiso_dev[1]['hard_negs'])  # 10
aiso_dev[1]['hard_negs'][0] # '43034001_0'
aiso_dev[1]['hn_scores']  # [... desc order like 1st example]
aiso_dev[1]['state2action'] # {'initial': {'query': 'Corliss Archer in the film Kiss and Tell', 'action': 'BM25',
#                                       'sp_ranks': {'BM25': {'804602_0': 2000, '33022480_0': 0},
#                                               'BM25+Link': {'804602_0': 0, '33022480_0': 2},
#                                                     'MDR': {'804602_0': 2000, '33022480_0': 0},
#                                                'MDR+Link': {'804602_0': 0, '33022480_0': 417}}},
#                       'Shirley Temple': {'query': 'Corliss Archer in the film Kiss and Tell', 'action': 'BM25',
#                                      'sp2_ranks': {'BM25': 0, 'BM25+Link': 2, 'MDR': 0, 'MDR+Link': 331}},
#            'Kiss and Tell (1945 film)': {'query': 'Shirley Temple', 'action': 'LINK',
#                                      'sp2_ranks': {'BM25': 0, 'BM25+Link': 2, 'MDR': 0, 'MDR+Link': 66}}}

"""
ADDITIONAL_SPECIAL_TOKENS
{'YES': '[unused0]', # 1
 'NO': '[unused1]',  # 2
 'SOP': '[unused2]', # 3
 'NONE': '[unused3]'} # 4 = "answerable score"
BOS/CLS = 101, EOS/SEP both 102
FUNCTIONS = ("ANSWER", "BM25", "MDR", "LINK")
FUNC2ID = {func: idx for idx, func in enumerate(FUNCTIONS)}
NA_POS = 3

eval_dataset[0]: Each time call evaldataset[0] get difft para combination for same question
{'q_id': '5a8b57f25542995d1e6f1371',
 'context_ids': ['2816539_0'],  # 'Scott Derrickson'. Can be list of at least 3 ids
 'context': 'Scott Derrickson [unused2] Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."',
 'context_token_spans': [(0, 5),  (6, 13),  (13, 16),  (17, 26),  (27, 32),  (33, 40),  (40, 43),  (44, 45),  (45, 49),  (50, 54),  (55, 57),  (57, 58),  (59, 63),  (63, 64),  (65, 67),  (68, 70),  (71, 79),  (80, 88),  (88, 89),  (90, 102),  (103, 106),  (107, 115),  (115, 116),  (117, 119),  (120, 125),  (126, 128),  (129, 132),  (133, 140),  (140, 141),  (142, 152),  (152, 153),
  (154, 156),  (157, 159),  (160, 164),  (165, 170),  (171, 174),  (175, 184),  (185, 191),  (192, 197),  (198, 202),  (203, 205),  (206, 207),  (207, 215),  (215, 216),  (216, 217),  (218, 219),  (219, 222),  (223, 225),  (225, 227),  (227, 230),  (230, 231),  (232, 234),  (235, 240),  (241, 245),  (245, 246),  (246, 247),  (248, 251),  (252, 253),  (253, 260),  (261, 263),  (264, 268),
  (269, 273),  (273, 274),  (274, 275),  (276, 278),  (279, 283),  (284, 286),  (287, 290),  (291, 295),  (296, 302),  (303, 312),  (313, 321),  (322, 333),  (333, 334),  (335, 336),  (336, 342),  (343, 350),  (350, 351),  (351, 352)],
 'sents_map': [('Scott Derrickson', 0),  ('Scott Derrickson', 1),  ('Scott Derrickson', 2)], #3 sentences in context - can have multiple aras in this list. Each sent idx is 0 based from that para not from context start
 'sparse_query': 'Ed Wood',
 'dense_expansion_id': '2816539_0', # 'Scott Derrickson' i.e. para_id to expand from if next action is MDR
 'link_targets': ['[unused3]',  'California',  'Deliver Us from Evil (2014 film)',  'Doctor Strange (2016 film)',
  'Horror film',  'Los Angeles',  'Marvel Cinematic Universe',  'Sinister (film)',  'The Exorcism of Emily Rose'],
 #input_ids has q + context
 'input_ids': tensor([  101,     1,     2,     4,  2020,  3660, 18928,  3385,  1998,  3968,  3536,  1997,  1996,  2168, 10662,  1029,   102,  3660, 18928,  3385,  3,  3660, 18928,  3385,  1006,  2141,  2251,  2385,  1010,  3547, 1007,  2003,  2019,  2137,  2472,  1010, 11167,  1998,  3135,  1012, 2002,  3268,  1999,  3050,  3349,  1010,  2662,  1012,  2002,  2003,
          2190,  2124,  2005,  9855,  5469,  3152,  2107,  2004,  1000, 16491, 1000,  1010,  1000,  1996,  4654,  2953, 22987,  2213,  1997,  6253, 3123,  1000,  1010,  1998,  1000,  8116,  2149,  2013,  4763,  1000, 1010,  2004,  2092,  2004,  1996,  2355,  8348, 21014,  5304, 18932, 1010,  1000,  3460,  4326,  1012,  1000,   102]),
 'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
 # Y N NONE ...title...para:
 'answer_mask': tensor([0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.]),
 'context_token_offset': tensor(17),  # 1st context token in input_ids right after SEP
 'paras_mark': [20],                  # 1st SOP/3 + any additional SOP for multiple paras eg [p1, p2, p3]
 'paras_span': [(17, 95)],
 'paras_label': tensor([1.]),                  # evidence/no evidence, for multiple paras can be list eg [1., 0.]
 'sents_span': [(21, 39), (40, 47), (48, 95)], # 3 sentences
 'sents_label': tensor([1., 0., 0.]),          # sentence is evidence/not evidence (in .jsonl but how derived?) - 1 entry per sent even over multiple paras
 'answer_starts': tensor([-1]),                # if answer in context would contain offset to start eg "Yes" = 1
 'answer_ends': tensor([-1]),
 'sparse_start': tensor(9),   #3968 #span of sparse query to execute if action_label = BM25/1
 'sparse_end': tensor(10),    #3536
 'dense_expansion': tensor(0), # idx of para to expand from in context_ids if action_label = MDR/2 - uses the SOP token associated with this para in the context
 #link spans seem to be first few tokens before any () of each anchor in text
 'links_spans': [[(3, 3)],  [(46, 46)],  [(75, 78)],  [(92, 93)],  [(54, 55)],  [(43, 44)],  [(86, 88)],  [(59, 59)],  [(63, 70)]],
 'link_label': tensor(0),      # idx in link_targets/link_spans to expand if action_label = LINK/3
 'action_label': tensor(1)}      #  ("ANSWER"/0, "BM25"/1, "MDR"/2, "LINK"/3)

b = [eval_dataset[i] for i in range(0,4)]
batch = collate_transitions(b, pad_id=tokenizer.pad_token_id)
batch.keys(): dict_keys(['q_id', 'context_ids', 'context', 'context_token_spans', 'sents_map', 'sparse_query', 'dense_expansion_id', 'link_targets', 'nn_input'])
batch['nn_input'].keys() dict_keys(['input_ids', 'attention_mask', 'token_type_ids', 'answer_mask', 'context_token_offset', 'paras_mark', 'paras_span', 'sents_span', 'sparse_start', 'sparse_end', 'links_spans', 'paras_label', 'sents_label', 'answer_starts', 'answer_ends', 'dense_expansion', 'link_label', 'action_label'])

# union_model forward:
# (B, T, H)  #TJH bs, seq len, hidden_size
seq_hiddens = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))[0]
none_hidden = seq_hiddens[:, 3, :]  # (B, H)  #TJH: 4th tok = NONE = evidence score
para_threshold = self.reranker(none_hidden).squeeze(-1)  # (B,)
state_hidden = seq_hiddens[:, 0, :]  # (B, H)  #TJH: CLS/<s>

MODEL HEADS:
  (reranker): Linear(in_features=1024, out_features=1, bias=True) # in = NONE tok and separately each SOP tok. Then softmax over concat(SOPs + NONE) to determine para to expand or suff evidence -> dense_hidden (B,H)
  (sp_cls): Linear(in_features=1024, out_features=1, bias=True)  # in = sentence toks mean per sentence out = sentence logits (then sigmoid/bce vs sent labels) (B, #sentences)
  (answerer): Answerer(  # in = seq_hidden [B, T, H] + answer mask & topk=1. Out with start_logits [B,T], end_logits [B,T], start idx [B,T], end idx [B,T] and answer_hidden = mean output of answerer [B,H]
    (qa_outputs): Linear(in_features=1024, out_features=2, bias=True)
  )
  (linker): Linker(
    (scorer): Linear(in_features=1024, out_features=1, bias=True)
  )
  (commander): Commander( # input (state_hidden (B,H) of CLS, 
                                   answer_hidden (B,H) of mean answer span, 
                                   sparse_hidden (B,H) of mean sparse span,
                                   dense_hidden (B,H) of max SOP/NONE, 
                                   link_hidden (B,H) of selected link)
                          # output action_logits (B,4) logits for ("ANSWER", "BM25", "MDR", "LINK")
                                   best_action: idx of best action
    (ffn): FFN(
      (dense1): Linear(in_features=3072, out_features=4096, bias=True)
      (dense2): Linear(in_features=4096, out_features=1024, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (act_scorer): Linear(in_features=1024, out_features=1, bias=True)
  )
  (bce_loss): BCEWithLogitsLoss() #sentence logits vs sentence labels then * sp_weight
  (ce_loss): CrossEntropyLoss()  # action logit vs action label with -1 actions excluded and wrong links preds setting action-> -1 and f1(pred ans span, label ans span) < 0.4 also setting action -1
                                   also used for answer span start/end loss
                                   and link span loss
 paras loss: key "memory" = a custom ranking loss between logits for [NONE] + eg [[SOP], [SOP]] vs [0.5] + eg [1., 0.]
            Unclear why this isn't a BCE loss like the sentence loss??
)    
    
Inference model outputs:
        #       (B, 4)         (B, T) Ans    (B, T) Ans  (B, _L)      (B, _P)      (B, _S)
        return (action_logits, start_logits, end_logits, link_logits, para_logits, sent_logits,
                # (B,)NONElogit (B,) TJH:Always 0's
                para_threshold, sent_threshold,
                # (B,) idx   (B, *)Ans  (B, *)Ans (B,)lnkidx (B,) para to expand next idx
                pred_action, top_start, top_end, pred_link, pred_exp,
                # (B,)    (B,)       (B,)para confidence
                ans_conf, link_conf, exp_conf)
"""





