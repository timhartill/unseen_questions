#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:18:23 2021

@author: Tim Hartill

Sentence embeddings using model from:
    Reimers, Nils, and Iryna Gurevych. 2019. 
    “Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.” 
    In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 
    3982–92. Hong Kong, China: Association for Computational Linguistics.

"""
import re
import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, cuda=True, model_name = 'sentence-transformers/stsb-roberta-large'):
        self.cuda = cuda
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.get_model()

        
    def mean_pooling(self, model_output, attention_mask):
        """
        Aggregating over token embeddings to get sentence embeddings. See for example:
        https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Sum columns
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    
    def get_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.cuda:
            self.model.to(torch.device("cuda"))
        self.model.eval()
        return
    
    
    def encode_inputs(self, sentences):
        """ sentences = ['string of input words 1', 'string of input words 2', ... ]
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, 
                                       max_length=512, return_tensors="pt") 
        if self.cuda:
            encoded_input.to(torch.device("cuda"))
        return encoded_input


    def get_embeddings(self, encoded_input):
        """ Returned sentence embeddings list of num_samples of np shape [1024]
        """
        emb_list = []
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, 
                                                encoded_input['attention_mask']).detach().cpu().numpy()
        for emb in sentence_embeddings:
            emb_list.append(emb)
        return emb_list

    def strip_context(self, questions):
        """ Utility to isolate questions from list of "question \\n context.." strings
        """
        new_questions = []
        for q in questions:
            q_split = q.split('\\n')
            if q_split[0].strip() == '':
                new_q = q
            else:
                new_q = q_split[0]
            new_questions.append( new_q )
        return new_questions


def restate_qa(q, ans):
    """ Restate eval q + a depending on input format with objective of making it more 
        aligned with self supervised training format for similarity purposes.
    Possible input formats:
        question \\n (without label = ssupervised, with label = open domain)
        question \\n MC options
        question \\n MC options \\n context
        question \\n context
    Output: concatenate answer to question (or replace leading wh word with answer and remove '?' if there is a leading wh word), 
            remove MC options if they exist.
    """
    ans = ans.strip()
    if ans == '':  #ssvised so no reformat
        return q
    ans = ans.rstrip('.')
    sentences = q.split('\\n')
    sentences = [s.strip() for s in sentences if s.strip() != '']
        
    # restate question
    new_q = sentences[0]
    #new_q2 = re.sub(r'^(what|who|when|how many|how much)', ans.capitalize(), sentences[0], flags=re.I)
    #if new_q != new_q2:
    #    new_q = new_q2
    #elif new_q.find(' _ ') != -1:
    if new_q.find(' _ ') != -1:
        new_q = new_q.replace(' _ ', ' ' + ans + ' ', 1)
    else:
        if new_q[-1] not in ['.', '?', '!', ':', ';']:
            new_q += '.'
        new_q = new_q + ' ' + ans.capitalize()
    new_q = new_q.rstrip('?')
    if new_q[-1] not in ['.', '?', '!', ':', ';']:
        new_q += '.'
    new_q = re.sub(r'\s+', ' ', new_q)  # remove double spaces
    new_q = new_q.replace(' .', '.')
    new_q += ' '
    if len(sentences) > 1:
        if not sentences[1].startswith('(A)'):
            new_q += sentences[1] 
            if new_q[-1] not in ['.', '?', '!', ':', ';']:
                new_q += '.'
            new_q += ' '
    if len(sentences) > 2:
        new_q += sentences[2]    
        if new_q[-1] not in ['.', '?', '!', ':', ';']:
            new_q += '.'
        new_q += ' '
    new_q += '\\n'
    return new_q


def restate_qa_all(questions, answers):
    """ Reformat all questions to match self-supervised format """
    num_q = len(questions)
    new_questions = []
    for i in range(num_q):
        new_questions.append( restate_qa(questions[i], answers[i]) )
    return new_questions
