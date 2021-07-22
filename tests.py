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
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
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

