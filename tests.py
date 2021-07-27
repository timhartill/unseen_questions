#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:07:59 2021

@author: Tim Hartill


"""
import json


# Spacy for NER!!
import spacy

nlp = spacy.load("en_core_web_sm")
txt="john smith is a nice person. apple is looking at buying u.k. startup for $1 billion in July 2020 or perhaps 1/6/23 or failing that 2024\nHello world.\nApples are good fruit to eat\nAre new zealand fruit or australian vegetables better for you? Astronomers look for the bright stars that orbit dark partners in the same way. The North Star can be used to find your way if you're lost in the dark. The north star can be used to find your way if you're lost in the dark"
doc = nlp(txt)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

all_sentences = [sent.text.strip() for sent in doc.sents]  #just splitting on full stop would split on decimal points etc and discard them
print(all_sentences)

token = doc[0]  #'The'

for i, token in enumerate(doc):
    print(i, token.text, token.pos_, token.dep_, "'" +token.text_with_ws+"'", token.idx, token.idx+len(token.text_with_ws))


    
#TODO - how to convert to/from tokenised version...


# look at QASc corpus:
with open('/data/thar011/data/qasc/QASC_Corpus/QASC_Corpus.txt', 'r') as f:
    qasc_corpus = f.read()

qasc_corpus[0:500]  #"Determination is very important.\nListening can never be over-emphasized.\nIf living alone, don t advertise it.\nReadings appear instantly on the LCD display.\nRevolution's search engine is Excite for Web Servers .\nMetallic cerium is found in an alloy with iron that is used in flints for cigarette lighters.\nAnd just about anybody can get pierced - including children.\nFax machines can transmit anything already on paper .\nVintages ratings reflect general district ratings.\nOcean Properties Find brief d"

qry = "Art can find the lost Art can find the lost Art can find the lost."

idx = qasc_corpus.find(qry)

qasc_corpus[idx-100:idx+250]

idx = qasc_corpus.find("what can be used to find your way if you're lost in the dark")
#idx: -1


def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    num_jsons = len(all_json_list)
    if verbose:
        print('JSON as text successfully loaded. Number of json messages in file is ', num_jsons)
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print('Text successfully converted to JSON.')
    return all_json_list

with open('/data/thar011/data/strategyqa/enwiki-20200511-cirrussearch-parasv2.jsonl', 'r') as f:
    cirrus = f.read()
all_json_list = cirrus.split('\n')
all_json_list2 = [json.loads(j) for j in all_json_list if j.strip() != '']



