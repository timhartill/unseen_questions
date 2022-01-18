#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 11:38:21 2022

@author: tim hartill

Explore KILT processed wiki dump without MongoDB
Also Explore MDR HPQA data

1. git clone kilt and cd into kilt dir
2. mkdir .../kilt/data
3. pip install -r requirements.txt then pip install . otherwise get_triviaqa_input.py download script doesnt work
4. cd into .../kilt/data then wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
5. cd .../kilt then python scripts/download_all_kilt_data.py
6.                  python scripts/get_triviaqa_input.py

"""
import os
import utils


KILT_DATA_DIR = '/data/thar011/gitrepos/KILT/data/'
KILT_WIKI = 'kilt_knowledgesource.json'
KILT_HPQA_DEV = 'hotpotqa-dev-kilt.jsonl'
KILT_FEVER_DEV = 'fever-dev-kilt.jsonl'
KILT_NQ_DEV = 'nq-dev-kilt.jsonl'

HPQA_DATA_DIR = '/data/thar011/gitrepos/multihop_dense_retrieval/data/hotpot/'
HPQA_DEV = 'hotpot_dev_with_neg_v0.json'  # retrieval training dev set

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








